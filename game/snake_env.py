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

        rewards[still_alive] += (
            0.04 * (1 + self.lengths[still_alive] / self.size) * (
                old_dist[still_alive] - new_dist[still_alive]
            )
        )

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

        # spawn new food
        if np.any(ate_food):
            self.spawn_food(np.where(ate_food)[0])

        # death
        rewards[done] -= 2.001 * (1 + 3 * self.lengths[done] / self.size)

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
        env_indices_cpu = np.asarray(env_indices_cpu)

        return {
            "snake_boards": np.asarray(self.snake_boards[env_indices_cpu]).copy(),
            "heads": np.asarray(self.heads[env_indices_cpu]).copy(),
            "lengths": np.asarray(self.lengths[env_indices_cpu]).copy(),
            "directions": np.asarray(self.curr_directions[env_indices_cpu]).copy(),
            "count": len(env_indices_cpu)
        }
    
    def seed_from_library(self, env_indices, position_library):
        count = position_library["count"]
        env_indices_cpu = env_indices.get() if hasattr(env_indices, "get") else env_indices
        
        K = 4  # number of distinct positions per training block
        chosen = np.random.choice(count, size=K, replace=False)
        # tile across envs: env i gets position chosen[i % K]
        tiled = chosen[np.arange(len(env_indices)) % K]
        self.snake_boards[env_indices_cpu] = position_library["snake_boards"][tiled]
        self.heads[env_indices_cpu] = position_library["heads"][tiled]
        self.lengths[env_indices_cpu] = position_library["lengths"][tiled]
        self.curr_directions[env_indices_cpu] = position_library["directions"][tiled]
    
    def pregenerate_random_snake_envs(self, pregenerated_envs=10000, min_length=10, max_length=375):
        snake_boards = np.zeros((pregenerated_envs, self.size, self.size), dtype=np.int16)
        heads = np.zeros((pregenerated_envs, 2), dtype=np.int32)
        lengths = np.zeros(pregenerated_envs, dtype=np.int32)
        directions = np.zeros(pregenerated_envs, dtype=np.int8)
        
        i = 0
        attempts = 0
        max_attempts = pregenerated_envs * 15
        
        while attempts < max_attempts and i < pregenerated_envs:
            attempts += 1
            target_length = np.random.randint(min_length, max_length)
            
            board = np.zeros((self.size, self.size), dtype=np.int16)
            x, y = np.random.randint(0, self.size), np.random.randint(0, self.size)
            body = [(x, y)]
            board[x, y] = 1
            last_dir = np.random.randint(4)
            
            for _ in range(target_length - 1):
                preferred = [last_dir] * 30 + [d for d in range(4) if d != self.OPPOSITE[last_dir]]
                np.random.shuffle(preferred)
                
                moved = False
                for d in preferred:
                    dx, dy = self.DIRS[d]
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
                #bias towards longer positions
            if len(body) < 100:
                random_num = np.random.randint(50)
                if random_num < 49:
                    continue
            
            actual_length = len(body)
            final_board = np.zeros((self.size, self.size), dtype=np.int16)
            for age, (bx, by) in enumerate(body):
                final_board[bx, by] = age + 1
            
            snake_boards[i] = final_board
            heads[i] = body[-1]
            lengths[i] = actual_length
            directions[i] = last_dir
            i += 1
            
        actual_count = i
        print(f"Generated {actual_count} environments in {attempts} attempts")
        print(f"Average environment length is {np.mean(lengths)}")
        print(f"Maximum environment length is {np.max(lengths)}")
        
        #return dictionary of arrays
        return {
            "snake_boards": snake_boards[:actual_count],
            "heads": heads[:actual_count],
            "lengths": lengths[:actual_count],
            "directions": directions[:actual_count],
            "count": actual_count
        }
        
    def save_environments(self, path="snake_positions.npz", **kwargs):
        np.savez_compressed(path, **kwargs)
    
    def load_environments(self, path="snake_positions.npz"):
        data = np.load(path)
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
        env.heads = self.heads.copy()
        env.lengths = self.lengths.copy()
        env.curr_directions = self.curr_directions.copy()
        return env