import numpy as np

class Snake_Env():
    def __init__(self, size=20):
        self.size = size
        self.snake_board = np.zeros((size, size), dtype=int)
        start = np.random.randint(3, 6, size=2)
        self.snake_board[start[0]][start[1]] = 1
        # start = [2, 2]
        # start = np.array(start)
        #head and tail pointers to track which parts of snake to update
        self.head = start.copy()
        self.tail = start.copy()
        self.direction = np.zeros(4) # [North, East, South, West]
        self.curr_direction = 0
        self.direction[self.curr_direction] = 1
        self.food_board = np.zeros((size, size), dtype=int)
        self.food = np.zeros(2, dtype=int)
        zeros = np.argwhere(self.snake_board == 0)
        i = np.random.randint(len(zeros))
        x, y = zeros[i]
        # x, y = 1, 1
        self.food_board[x][y] = 1
        self.food[0] = x
        self.food[1] = y
        self.next_tail_dict = {}
        self.length = 1
        self.running = True
    
    def set_direction(self, turn):
        #cannot turn back into snake
        match turn:
            case 0:
                if self.curr_direction == 2:
                    return self.curr_direction
            case 1:
                if self.curr_direction == 3:
                    return self.curr_direction
            case 2:
                if self.curr_direction == 0:
                    return self.curr_direction
            case 3:
                if self.curr_direction == 1:
                    return self.curr_direction
        self.direction[self.curr_direction] = 0
        self.direction[turn] = 1
        self.curr_direction = turn
        return self.curr_direction
    
    def step(self, direction):
        reward = 0
        reward -= 0.0005 #small step penalty
        # reward += 0.01   # survival bonus per step
        turn = np.argmax(direction)
        self.set_direction(turn)
        
        #calculate new_head
        new_head = np.zeros(2, dtype=int)
        match self.curr_direction:
            case 0:
                if self.head[1] == self.size - 1:
                    self.running = False
                    return reward - 1
                else:
                    new_head[0] = self.head[0]
                    new_head[1] = self.head[1] + 1
            case 1:
                if self.head[0] == self.size - 1:
                    self.running = False
                    return reward - 1
                else:
                    new_head[0] = self.head[0] + 1
                    new_head[1] = self.head[1]
            case 2:
                if self.head[1] == 0:
                    self.running = False
                    return reward - 1
                else:
                    new_head[0] = self.head[0]
                    new_head[1] = self.head[1] - 1
            case 3:
                if self.head[0] == 0:
                    self.running = False
                    return reward - 1
                else:
                    new_head[0] = self.head[0] - 1
                    new_head[1] = self.head[1]
        if self.snake_board[new_head[0]][new_head[1]] == 1:
            # allow moving into tail ONLY if tail moves away this step
            if not (new_head[0] == self.tail[0] and new_head[1] == self.tail[1]):
                self.running = False
                return reward - 1
        
        #Manhattan distance calculation to see if snake is closer to food
        new_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
        old_dist = abs(self.head[0] - self.food[0]) + abs(self.head[1] - self.food[1])
        
        if new_dist < old_dist:
            reward += 0.05 #makes stepping in direction of food more positive
        else:
            reward -= 0.05

        #check if new_head is over food
        if new_head[0] == self.food[0] and new_head[1] == self.food[1]:
            reward += 1
        
        #update next_tail_dict
        self.next_tail_dict[tuple(self.head)] = new_head
        #update head
        self.head = new_head
        
        # After updating head:
        self.snake_board[self.head[0]][self.head[1]] = 1
        
        # check if head is over food
        if self.head[0] == self.food[0] and self.head[1] == self.food[1]:
            self.length += 1
            
            self.food_board[self.food[0]][self.food[1]] = 0
            # select new food location that is not a snake body
            zeros = np.argwhere(self.snake_board == 0)
            
            if len(zeros) == 0:
                reward += 1
                self.running = False
                return reward
            
            i = np.random.randint(len(zeros))
            x, y = zeros[i]
            
            self.food[0] = x
            self.food[1] = y
            self.food_board[x][y] = 1
            
            return reward
        
        #update tail pointer if not over food
        old_tail = self.tail  # save before advancing

        # After updating tail (when no food eaten):
        self.tail = self.next_tail_dict[tuple(self.tail)]
        del self.next_tail_dict[tuple(old_tail)]
        self.snake_board[old_tail[0]][old_tail[1]] = 0  # clear old tail
        return reward
        
    def get_state(self):
        # #use tail dict as a sort of linked list to create more encoded snake board
        # #start from tail and go all the way to head and gradually increase value of snake body as you go
        # node = self.tail
        # i = 1
        # while node in self.next_tail_dict: # O(n) n = snake length
        #     self.snake_board[node[0]][node[1]] = i / self.length # gradually increases as it goes up snake length
        #     node = self.next_tail_dict[node]
        #     i += 1
        
        snake_head_board = np.zeros((self.size, self.size))
        snake_head_board[self.head[0]][self.head[1]] = 1
        
        boards = np.stack([self.snake_board, snake_head_board, self.food_board]) #tensor
        length = self.length / (self.size * self.size) # normalize length value between 0 and 1
        
        distance_to_danger_left = 1
        x = self.head[0] - 1
        while x >= 0 and self.snake_board[x][self.head[1]] != 1:
            distance_to_danger_left += 1
            x -= 1
        distance_to_danger_left /= self.size
        
        distance_to_danger_right = 1
        x = self.head[0] + 1
        while x < self.size and self.snake_board[x][self.head[1]] != 1:
            distance_to_danger_right += 1
            x += 1
        distance_to_danger_right /= self.size
        
        distance_to_danger_up = 1
        y = self.head[1] + 1
        while y < self.size and self.snake_board[self.head[0]][y] != 1:
            distance_to_danger_up += 1
            y += 1
        distance_to_danger_up /= self.size
        
        distance_to_danger_down = 1
        y = self.head[1] - 1
        while y >= 0 and self.snake_board[self.head[0]][y] != 1:
            distance_to_danger_down += 1
            y -= 1
        distance_to_danger_down /= self.size
        
        dx_food = (self.head[0] - self.food[0]) / self.size
        dy_food = (self.head[1] - self.food[1]) / self.size
        
        return boards, self.direction, length, distance_to_danger_up, distance_to_danger_right, distance_to_danger_down, distance_to_danger_left, dx_food, dy_food