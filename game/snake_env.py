try:
    import cupy as np
    USE_GPU = True
except ImportError:
    import numpy as np
    USE_GPU = False

# plain CPU numpy — used only for one-time / inherently-sequential setup
# routines (pregenerating library positions) and for on-disk save/load,
# so the position library always lives on the host regardless of backend.
import numpy as onp


class VectorizedSnakeEnv:
    """
    Fully vectorized Snake environment.

    Directions:
        0 = Up
        1 = Right
        2 = Down
        3 = Left
    """

    DIRS = np.array([
        [0, 1],    # Up
        [1, 0],    # Right
        [0, -1],   # Down
        [-1, 0],   # Left
    ], dtype=np.int32)

    OPPOSITE = np.array([2, 3, 0, 1], dtype=np.int32)

    def __init__(self, num_envs=64, size=20):
        self.num_envs = num_envs
        self.size = size

        # Snake board stores body age
        # 0 = empty
        # larger value = newer body segment
        self.snake_boards = np.zeros(
            (num_envs, size, size),
            dtype=np.int16
        )

        self.food_boards = np.zeros(
            (num_envs, size, size),
            dtype=np.int8
        )

        self.heads = np.zeros((num_envs, 2), dtype=np.int32)
        self.foods = np.zeros((num_envs, 2), dtype=np.int32)

        self.lengths = np.ones(num_envs, dtype=np.int32)

        self.curr_directions = np.zeros(num_envs, dtype=np.int8)

        self.running = np.ones(num_envs, dtype=bool)

        self.reset()

    def reset(self, env_indices=None):

        if env_indices is None:
            env_indices = np.arange(self.num_envs)
        else:
            env_indices = np.asarray(env_indices)

        self.running[env_indices] = True
        self.lengths[env_indices] = 1
        self.curr_directions[env_indices] = 0

        self.snake_boards[env_indices] = 0
        self.food_boards[env_indices] = 0

        x = np.random.randint(3, 6, size=len(env_indices))
        y = np.random.randint(3, 6, size=len(env_indices))

        self.heads[env_indices, 0] = x
        self.heads[env_indices, 1] = y

        self.snake_boards[
            env_indices,
            x,
            y
        ] = 1

        self.spawn_food(env_indices)

    def spawn_food(self, env_indices):
        """
        Fully vectorized food spawning — no per-env Python loop.

        For each env, assigns a random score to every board cell, masks
        out occupied cells to -1, then takes the per-row argmax. Since
        every empty cell gets an iid random score, the argmax lands on
        a uniformly random empty cell (assumes at least one empty cell
        exists per env, which the caller guarantees).
        """

        env_indices = np.asarray(env_indices)

        if env_indices.size == 0:
            return

        self.food_boards[env_indices] = 0

        n = int(env_indices.shape[0])
        s = self.size

        empty_mask = (self.snake_boards[env_indices] == 0).reshape(n, s * s)

        scores = np.random.rand(n, s * s)
        scores = np.where(empty_mask, scores, -1.0)

        flat_idx = np.argmax(scores, axis=1)
        fx = (flat_idx // s).astype(np.int32)
        fy = (flat_idx % s).astype(np.int32)

        self.foods[env_indices, 0] = fx
        self.foods[env_indices, 1] = fy
        self.food_boards[env_indices, fx, fy] = 1

    def step(self, actions):
        actions = np.asarray(actions)

        rewards = np.zeros(self.num_envs, dtype=np.float32)

        alive = self.running.copy()

        rewards[alive] -= 0.001 * (1 + self.lengths[alive] / self.size) # small step penalty

        old_heads = self.heads.copy()

        # prevent reversing
        opposite = self.OPPOSITE[self.curr_directions]

        invalid_turn = (actions == opposite)

        actions = np.where(
            invalid_turn,
            self.curr_directions,
            actions
        )

        self.curr_directions = actions

        # compute new head positions
        deltas = self.DIRS[actions]

        new_heads = self.heads + deltas

        nx = new_heads[:, 0]
        ny = new_heads[:, 1]

        # wall collisions
        wall_collision = (
            (nx < 0) |
            (nx >= self.size) |
            (ny < 0) |
            (ny >= self.size)
        )

        safe_nx = np.clip(nx, 0, self.size - 1)
        safe_ny = np.clip(ny, 0, self.size - 1)

        # snake collision
        body_collision = (
            self.snake_boards[
                np.arange(self.num_envs),
                safe_nx,
                safe_ny
            ] > 0
        )

        # moving into tail is allowed
        tail_cells = (self.snake_boards == 1)

        moving_into_tail = tail_cells[
            np.arange(self.num_envs),
            safe_nx,
            safe_ny
        ]

        body_collision &= ~moving_into_tail

        done = (
            wall_collision |
            body_collision
        ) & alive

        still_alive = alive & (~done)

        # distance shaping
        old_dist = np.abs(
            old_heads[:, 0] - self.foods[:, 0]
        ) + np.abs(
            old_heads[:, 1] - self.foods[:, 1]
        )

        new_dist = np.abs(
            safe_nx - self.foods[:, 0]
        ) + np.abs(
            safe_ny - self.foods[:, 1]
        )

        # rewards[still_alive] += (
        #     0.01 * (1 + self.lengths[still_alive] / self.size) * (
        #         old_dist[still_alive] - new_dist[still_alive]
        #     )
        # )

        # food check
        ate_food = (
            (safe_nx == self.foods[:, 0]) &
            (safe_ny == self.foods[:, 1]) &
            still_alive
        )

        rewards[ate_food] += 3.0 * (1 + self.lengths[ate_food] / self.size)

        # decrement snake ages
        # snakes that eat keep their tail
        decay_mask = still_alive & (~ate_food)

        self.snake_boards[decay_mask] = np.maximum(
            self.snake_boards[decay_mask] - 1,
            0
        )

        # snakes that eat do NOT decay
        self.lengths[ate_food] += 1

        # update heads
        self.heads[still_alive] = new_heads[still_alive]

        # place new heads
        self.snake_boards[
            np.arange(self.num_envs)[still_alive],
            safe_nx[still_alive],
            safe_ny[still_alive]
        ] = self.lengths[still_alive]

        # spawn new food — vectorized, so this is cheap even every step
        ate_idx = np.where(ate_food)[0]
        if ate_idx.size > 0:
            self.spawn_food(ate_idx)

        # death
        rewards[done] -= 1.5 * (1 + 3 * self.lengths[done] / self.size)

        self.running[done] = False

        return rewards

    def get_state(self):

        # normalized snake board
        normalized_snake = (
            self.snake_boards /
            np.maximum(
                self.lengths[:, None, None],
                1
            )
        )

        head_board = np.zeros_like(
            normalized_snake,
            dtype=np.float32
        )

        head_board[
            np.arange(self.num_envs),
            self.heads[:, 0],
            self.heads[:, 1]
        ] = 1.0

        boards = np.stack([
            normalized_snake,
            head_board,
            self.food_boards
        ], axis=1).astype(np.float32)

        direction_onehot = np.eye(
            4,
            dtype=np.float32
        )[self.curr_directions]

        length = (
            self.lengths /
            (self.size * self.size)
        ).astype(np.float32)

        # food deltas
        dx_food = (
            self.heads[:, 0] -
            self.foods[:, 0]
        ) / self.size

        dy_food = (
            self.heads[:, 1] -
            self.foods[:, 1]
        ) / self.size

        running = self.running.astype(np.float32)

        return (
            boards,
            direction_onehot,
            length,
            dx_food.astype(np.float32),
            dy_food.astype(np.float32),
            running
        )

    def snapshot_envs(self, env_indices):
        """
        Extract current internal state for the given env indices,
        in the same format as the pregenerated position library.
        Only meaningful for envs that are currently running.
        """
        env_indices_cpu = env_indices.get() if hasattr(env_indices, "get") else env_indices
        env_indices_cpu = onp.asarray(env_indices_cpu)

        def to_cpu(x):
            return x.get() if hasattr(x, "get") else onp.asarray(x)

        return {
            "snake_boards": to_cpu(self.snake_boards[env_indices_cpu]).copy(),
            "food_boards": to_cpu(self.food_boards[env_indices_cpu]).copy(),
            "heads": to_cpu(self.heads[env_indices_cpu]).copy(),
            "lengths": to_cpu(self.lengths[env_indices_cpu]).copy(),
            "directions": to_cpu(self.curr_directions[env_indices_cpu]).copy(),
            "count": len(env_indices_cpu)
        }

    def seed_from_library(self, env_indices, position_library):
        count = position_library["count"]
        env_indices_cpu = env_indices.get() if hasattr(env_indices, "get") else env_indices

        K = 512  # number of distinct positions per training block
        chosen = onp.random.choice(count, size=K, replace=False)
        # tile across envs: env i gets position chosen[i % K]
        tiled = chosen[onp.arange(len(env_indices_cpu)) % K]

        self.snake_boards[env_indices_cpu] = np.asarray(position_library["snake_boards"][tiled])
        self.food_boards[env_indices_cpu] = np.asarray(position_library["food_boards"][tiled])
        self.heads[env_indices_cpu] = np.asarray(position_library["heads"][tiled])
        self.lengths[env_indices_cpu] = np.asarray(position_library["lengths"][tiled])
        self.curr_directions[env_indices_cpu] = np.asarray(position_library["directions"][tiled])

        # food/running weren't part of the library format, so re-derive them:
        # each seeded env keeps whatever food it currently has if valid,
        # otherwise gets a fresh one, and is marked running.
        self.running[env_indices_cpu] = True
        self.spawn_food(np.asarray(env_indices_cpu))

    def pregenerate_random_snake_envs(self, pregenerated_envs=10000, min_length=10, max_length=375):
        # Deliberately plain CPU numpy: this builds each snake body one
        # cell at a time (inherently sequential), and only runs once at
        # startup — not worth round-tripping to GPU per step.
        snake_boards = onp.zeros((pregenerated_envs, self.size, self.size), dtype=onp.int16)
        heads = onp.zeros((pregenerated_envs, 2), dtype=onp.int32)
        lengths = onp.zeros(pregenerated_envs, dtype=onp.int32)
        directions = onp.zeros(pregenerated_envs, dtype=onp.int8)

        opposite_cpu = onp.array([2, 3, 0, 1], dtype=onp.int32)
        dirs_cpu = onp.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=onp.int32)

        i = 0
        attempts = 0
        max_attempts = pregenerated_envs * 15

        while attempts < max_attempts and i < pregenerated_envs:
            attempts += 1
            target_length = onp.random.randint(min_length, max_length)

            board = onp.zeros((self.size, self.size), dtype=onp.int16)
            x, y = onp.random.randint(0, self.size), onp.random.randint(0, self.size)
            body = [(x, y)]
            board[x, y] = 1
            last_dir = onp.random.randint(4)

            for _ in range(target_length - 1):
                preferred = [last_dir] * 30 + [d for d in range(4) if d != opposite_cpu[last_dir]]
                onp.random.shuffle(preferred)

                moved = False
                for d in preferred:
                    dx, dy = dirs_cpu[d]
                    nx, ny = body[-1][0] + dx, body[-1][1] + dy
                    if 0 <= nx < self.size and 0 <= ny < self.size and board[nx, ny] == 0:
                        moved = True
                        body.append((nx, ny))
                        board[nx, ny] = 1
                        last_dir = d
                        break

                if not moved:
                    break

            if len(body) < min_length:
                continue
                # bias towards longer positions
            if len(body) < 100:
                random_num = onp.random.randint(50)
                if random_num < 49:
                    continue

            actual_length = len(body)
            final_board = onp.zeros((self.size, self.size), dtype=onp.int16)
            for age, (bx, by) in enumerate(body):
                final_board[bx, by] = age + 1

            snake_boards[i] = final_board
            heads[i] = body[-1]
            lengths[i] = actual_length
            directions[i] = last_dir
            i += 1

        actual_count = i
        print(f"Generated {actual_count} environments in {attempts} attempts")
        print(f"Average environment length is {onp.mean(lengths)}")
        print(f"Maximum environment length is {onp.max(lengths)}")

        # return dictionary of arrays
        return {
            "snake_boards": snake_boards[:actual_count],
            "heads": heads[:actual_count],
            "lengths": lengths[:actual_count],
            "directions": directions[:actual_count],
            "count": actual_count
        }

    def save_environments(self, path="snake_positions.npz", **kwargs):
        onp.savez_compressed(path, **kwargs)

    def load_environments(self, path="snake_positions.npz"):
        data = onp.load(path)
        return {
            "snake_boards": data["snake_boards"],
            "heads": data["heads"],
            "lengths": data["lengths"],
            "directions": data["directions"],
            "count": int(data["lengths"].shape[0])
        }

    def copy_env(self):
        env = VectorizedSnakeEnv(self.num_envs, self.size)
        env.snake_boards = self.snake_boards.copy()
        env.food_boards = self.food_boards.copy()
        env.heads = self.heads.copy()
        env.foods = self.foods.copy()
        env.lengths = self.lengths.copy()
        env.curr_directions = self.curr_directions.copy()
        env.running = self.running.copy()
        return env

    # ------------------------------------------------------------------
    # Batch helpers
    #
    # These let callers (e.g. the tree search) treat a whole frontier of
    # speculative envs as a single vectorized VectorizedSnakeEnv instead
    # of looping over individual num_envs=1 copies in Python. They never
    # call __init__/reset — they just wire up the raw field arrays.
    # ------------------------------------------------------------------

    @staticmethod
    def from_fields(snake_boards, food_boards, heads, foods, lengths, curr_directions, running):
        env = VectorizedSnakeEnv.__new__(VectorizedSnakeEnv)
        env.num_envs = int(snake_boards.shape[0])
        env.size = int(snake_boards.shape[1])
        env.snake_boards = snake_boards
        env.food_boards = food_boards
        env.heads = heads
        env.foods = foods
        env.lengths = lengths
        env.curr_directions = curr_directions
        env.running = running
        return env

    @staticmethod
    def concat(envs):
        """Concatenate several VectorizedSnakeEnv instances (any num_envs each) into one batch."""
        return VectorizedSnakeEnv.from_fields(
            snake_boards=np.concatenate([e.snake_boards for e in envs], axis=0),
            food_boards=np.concatenate([e.food_boards for e in envs], axis=0),
            heads=np.concatenate([e.heads for e in envs], axis=0),
            foods=np.concatenate([e.foods for e in envs], axis=0),
            lengths=np.concatenate([e.lengths for e in envs], axis=0),
            curr_directions=np.concatenate([e.curr_directions for e in envs], axis=0),
            running=np.concatenate([e.running for e in envs], axis=0),
        )

    def select(self, idx):
        """Return a new env containing only the sub-batch at integer/bool index `idx`."""
        idx = np.asarray(idx)
        return VectorizedSnakeEnv.from_fields(
            snake_boards=self.snake_boards[idx].copy(),
            food_boards=self.food_boards[idx].copy(),
            heads=self.heads[idx].copy(),
            foods=self.foods[idx].copy(),
            lengths=self.lengths[idx].copy(),
            curr_directions=self.curr_directions[idx].copy(),
            running=self.running[idx].copy(),
        )