try:
    import cupy as np
except ImportError:
    import numpy as np

from game.snake_env import VectorizedSnakeEnv


class search():
    def __init__(self, neural_net, depth=1):
        self.nn = neural_net
        self.depth = depth

    def flood_fill_batch(self, boards, heads):
        """
        Vectorized reachability score for a whole batch of boards at once.

        boards: (N, H, W) array, 0 = free cell, non-zero = occupied
        heads:  (N, 2) int array of (x, y) head positions

        Returns (N,) array: fraction of each board's free cells that are
        reachable from its head.

        Implemented as iterative dilation (BFS "wavefront" expressed as
        array ops) instead of a per-board Python BFS queue, so every board
        in the batch advances together as plain vectorized array math —
        this is what lets it run on the GPU. `max_iters = H * W` is the
        worst-case geodesic distance across a maze-shaped free region, so
        results are exact, matching a true BFS.
        """
        boards = np.asarray(boards)
        heads = np.asarray(heads)
        N, H, W = boards.shape

        free_mask = (boards == 0)
        free_counts = free_mask.reshape(N, -1).sum(axis=1)

        visited = np.zeros((N, H, W), dtype=bool)
        idx = np.arange(N)
        visited[idx, heads[:, 0], heads[:, 1]] = True

        max_iters = H * W
        for _ in range(max_iters):
            up = np.zeros_like(visited)
            up[:, :-1, :] = visited[:, 1:, :]
            down = np.zeros_like(visited)
            down[:, 1:, :] = visited[:, :-1, :]
            left = np.zeros_like(visited)
            left[:, :, :-1] = visited[:, :, 1:]
            right = np.zeros_like(visited)
            right[:, :, 1:] = visited[:, :, :-1]
            visited = visited | ((up | down | left | right) & free_mask)

        counts = visited.reshape(N, -1).sum(axis=1)
        return counts / np.maximum(free_counts, 1)

    def find_top2_actions_batch(self, probs):
        """
        probs: (N, 4) action probabilities.
        Returns one-hot arrays (N, 4) for the best and second-best action
        of every row at once (replaces the old per-node python loop).
        """
        order = np.argsort(probs, axis=1)
        best_idx = order[:, -1]
        second_idx = order[:, -2]

        N = probs.shape[0]
        rows = np.arange(N)

        best_oh = np.zeros_like(probs)
        second_oh = np.zeros_like(probs)
        best_oh[rows, best_idx] = 1
        second_oh[rows, second_idx] = 1

        return best_idx, second_idx, best_oh, second_oh

    def find_best_action(self, state, direction, length, dx_food, dy_food, running, env):
        """
        BFS-style search where every depth level advances as ONE batched
        operation across the whole frontier: one forward prop, one env
        step for each of the two candidate actions, one vectorized
        flood-fill, one prune. There is no per-node Python loop — the
        frontier itself is a single (batched) VectorizedSnakeEnv whose
        size shrinks as branches die or get pruned.
        """

        FLOOD_FILL_THRESHOLD = 0.3  # prune if less than 30% of free space is accessible
        FLOOD_FILL_WEIGHT = 0.5
        VALUE_WEIGHT = 0.5

        batch_env = env          # frontier of speculative envs, num_envs = live branch count
        root_actions = None      # (num_envs, 4) one-hot, aligned with batch_env

        for depth in range(self.depth):
            if batch_env.num_envs == 0:
                break

            b_state, b_dir, b_len, b_dx, b_dy, b_running = batch_env.get_state()

            probs, _ = self.nn.forward_prop_search(
                b_state, b_dir, b_len, b_dx, b_dy, b_running
            )
            probs = np.asarray(probs)

            _, _, best_oh, second_oh = self.find_top2_actions_batch(probs)
            best_idx = np.argmax(best_oh, axis=1)
            second_idx = np.argmax(second_oh, axis=1)

            env_a = batch_env.copy_env()
            env_b = batch_env.copy_env()
            env_a.step(best_idx)
            env_b.step(second_idx)

            branch_env = VectorizedSnakeEnv.concat([env_a, env_b])

            if depth == 0:
                branch_root_actions = np.concatenate([best_oh, second_oh], axis=0)
            else:
                branch_root_actions = np.concatenate([root_actions, root_actions], axis=0)

            alive_mask = branch_env.running.astype(bool)
            fill_scores = self.flood_fill_batch(branch_env.snake_boards, branch_env.heads)
            keep_mask = alive_mask & (fill_scores >= FLOOD_FILL_THRESHOLD)

            keep_idx = np.where(keep_mask)[0]

            batch_env = branch_env.select(keep_idx)
            root_actions = branch_root_actions[keep_idx]

        if batch_env.num_envs == 0:
            # all branches pruned - fall back to policy only
            probs, values = self.nn.forward_prop_search(
                state, direction, length, dx_food, dy_food, running
            )
            probs = np.asarray(probs)
            values = np.asarray(values)
            _, _, best_oh, _ = self.find_top2_actions_batch(probs)
            return best_oh[0], float(values[0, 0])

        # evaluate all leaf nodes in one batch
        leaf_state, leaf_dir, leaf_len, leaf_dx, leaf_dy, leaf_running = batch_env.get_state()

        _, leaf_values = self.nn.forward_prop_search(
            leaf_state, leaf_dir, leaf_len, leaf_dx, leaf_dy, leaf_running
        )
        leaf_values = np.asarray(leaf_values)

        fill_scores = self.flood_fill_batch(batch_env.snake_boards, batch_env.heads)
        combined_scores = VALUE_WEIGHT * leaf_values[:, 0] + FLOOD_FILL_WEIGHT * fill_scores

        best_leaf_idx = int(np.argmax(combined_scores))
        best_action = root_actions[best_leaf_idx]
        best_score = float(combined_scores[best_leaf_idx])

        return best_action, best_score