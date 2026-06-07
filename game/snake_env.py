import numpy as np

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

        self.food_boards[env_indices] = 0

        for i in env_indices:
            empty = np.argwhere(self.snake_boards[i] == 0)

            idx = np.random.randint(len(empty))

            fx, fy = empty[idx]

            self.foods[i] = [fx, fy]

            self.food_boards[i, fx, fy] = 1

    def step(self, actions):

        if hasattr(actions, "get"):
            actions = actions.get()
        else:
            actions = np.asarray(actions)

        rewards = np.zeros(self.num_envs, dtype=np.float32)

        alive = self.running.copy()

        rewards[alive] += 0.004

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

        rewards[still_alive] += (
            0.04 * (
                old_dist[still_alive] -
                new_dist[still_alive]
            )
        )

        # food check
        ate_food = (
            (safe_nx == self.foods[:, 0]) &
            (safe_ny == self.foods[:, 1]) &
            still_alive
        )

        rewards[ate_food] += 2.0

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

        # spawn new food
        if np.any(ate_food):
            self.spawn_food(np.where(ate_food)[0])

        # death
        rewards[done] -= 2.0

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