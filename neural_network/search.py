import numpy as np
from neural_network.nn import NN
from game.snake_env import VectorizedSnakeEnv

class search():
    def __init__(self, neural_net, depth=1):
        self.nn = neural_net
        self.depth = depth

    def flood_fill(self, env, env_idx=0):
        board = env.snake_boards[env_idx]
        head = env.heads[env_idx]
        size = env.size

        free_cells = int(np.sum(board == 0))
        if free_cells == 0:
            return 0.0

        visited = np.zeros((size, size), dtype=bool)
        queue = [tuple(head)]
        visited[head[0], head[1]] = True
        count = 0

        while queue:
            x, y = queue.pop(0)
            count += 1
            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < size and 0 <= ny < size:
                    if not visited[nx, ny] and board[nx, ny] == 0:
                        visited[nx, ny] = True
                        queue.append((nx, ny))

        return count / free_cells  # fraction of free cells that are reachable

    def find_best_action(self, state, direction, length, dx_food, dy_food, running, env):
        """
        BFS search that batches forward props at each depth level.
        Prunes branches where flood fill score is too low.
        """

        # each node stores:
        # (env, state, direction, length, dx_food, dy_food, running, root_action, is_alive)
        initial_node = (env, state, direction, length, dx_food, dy_food, running, None)
        current_level = [initial_node]

        FLOOD_FILL_THRESHOLD = 0.3  # prune if less than 30% of free space is accessible
        FLOOD_FILL_WEIGHT = 0.5
        VALUE_WEIGHT = 0.5

        for depth in range(self.depth):
            if not current_level:
                break

            # batch all states at this level into single forward prop
            batch_states      = np.concatenate([node[1] for node in current_level], axis=0)
            batch_directions  = np.concatenate([node[2] for node in current_level], axis=0)
            batch_lengths     = np.concatenate([node[3].reshape(-1) for node in current_level], axis=0)
            batch_dx          = np.concatenate([node[4].reshape(-1) for node in current_level], axis=0)
            batch_dy          = np.concatenate([node[5].reshape(-1) for node in current_level], axis=0)
            batch_running     = np.concatenate([node[6].reshape(-1) for node in current_level], axis=0)

            probs, values = self.nn.forward_prop_search(
                batch_states,
                batch_directions,
                batch_lengths,
                batch_dx,
                batch_dy,
                batch_running
            )

            next_level = []

            for i, node in enumerate(current_level):
                node_env, _, _, _, _, _, _, root_action = node

                action_1_one_hot, action_2_one_hot = self.find_top_two_actions(probs[i])

                for action_one_hot in [action_1_one_hot, action_2_one_hot]:
                    action_idx = int(action_one_hot.argmax())

                    # set root action on first depth
                    if depth == 0:
                        this_root_action = action_one_hot
                    else:
                        this_root_action = root_action

                    new_env = node_env.copy_env()
                    new_env.step(np.array([action_idx]))

                    # prune dead branches
                    if not new_env.running[0]:
                        continue

                    # flood fill pruning
                    fill_score = self.flood_fill(new_env)
                    if fill_score < FLOOD_FILL_THRESHOLD:
                        continue

                    new_state, new_dir, new_len, new_dx, new_dy, new_running = new_env.get_state()

                    next_level.append((
                        new_env,
                        new_state,
                        new_dir,
                        new_len,
                        new_dx,
                        new_dy,
                        new_running,
                        this_root_action
                    ))

            current_level = next_level

        # evaluate all leaf nodes
        if not current_level:
            # all branches pruned - fall back to policy only
            probs, values = self.nn.forward_prop_search(state, direction, length, dx_food, dy_food, running)
            best_action, _ = self.find_top_two_actions(probs[0])
            return best_action, float(values[0, 0])

        # batch evaluate leaves
        batch_states     = np.concatenate([node[1] for node in current_level], axis=0)
        batch_directions = np.concatenate([node[2] for node in current_level], axis=0)
        batch_lengths    = np.concatenate([node[3].reshape(-1) for node in current_level], axis=0)
        batch_dx         = np.concatenate([node[4].reshape(-1) for node in current_level], axis=0)
        batch_dy         = np.concatenate([node[5].reshape(-1) for node in current_level], axis=0)
        batch_running    = np.concatenate([node[6].reshape(-1) for node in current_level], axis=0)

        _, leaf_values = self.nn.forward_prop_search(
            batch_states,
            batch_directions,
            batch_lengths,
            batch_dx,
            batch_dy,
            batch_running
        )

        best_score = -1e9
        best_action = None

        for i, node in enumerate(current_level):
            node_env = node[0]
            root_action = node[7]

            fill_score = self.flood_fill(node_env)
            value_score = float(leaf_values[i, 0])

            combined_score = VALUE_WEIGHT * value_score + FLOOD_FILL_WEIGHT * fill_score

            if combined_score > best_score:
                best_score = combined_score
                best_action = root_action

        return best_action, best_score
    
    def find_top_two_actions(self, probs):
        top2 = np.argsort(probs)[-2:]
        action_1_one_hot = np.zeros(4)
        action_2_one_hot = np.zeros(4)
        idx1 = int(top2[1])
        idx2 = int(top2[0])
        action_1_one_hot[idx1] = 1
        action_2_one_hot[idx2] = 1
        return action_1_one_hot, action_2_one_hot