import numpy as np

class Snake_Env():
    def __init__(self, size=20):
        self.size = size
        self.snake_board = np.zeros((size, size), dtype=int)
        self.snake_board[0][0] = 1
        #head and tail pointers to track which parts of snake to update
        self.head = np.zeros(2, dtype=int)
        self.tail = np.zeros(2, dtype=int)
        self.direction = np.zeros(4)
        self.direction[0] = 1  # 0:North, 1:East, 2:South, 3:West
        self.curr_direction = 0 # index of direction array
        self.food_board = np.zeros((size, size), dtype=int)
        x, y = np.random.randint(2, 10), np.random.randint(2, 10)
        self.food_board[x][y] = 1
        self.food = np.zeros(2, dtype=int)
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
        turn = np.argmax(direction)
        self.set_direction(turn)
        
        #calculate new_head
        new_head = np.zeros(2, dtype=int)
        match self.curr_direction:
            case 0:
                if self.head[1] == self.size - 1:
                    self.running = False
                    return -2 - 0.005
                else:
                    new_head[0] = self.head[0]
                    new_head[1] = self.head[1] + 1
            case 1:
                if self.head[0] == self.size - 1:
                    self.running = False
                    return -2 - 0.005
                else:
                    new_head[0] = self.head[0] + 1
                    new_head[1] = self.head[1]
            case 2:
                if self.head[1] == 0:
                    self.running = False
                    return -2 - 0.005
                else:
                    new_head[0] = self.head[0]
                    new_head[1] = self.head[1] - 1
            case 3:
                if self.head[0] == 0:
                    self.running = False
                    return -2 - 0.005
                else:
                    new_head[0] = self.head[0] - 1
                    new_head[1] = self.head[1]
        
        reward = -0.005 #small step penalty
        #check if new head is snake body and not tail
        if(self.snake_board[new_head[0]][new_head[1]] == 1 and not (new_head[0] == self.tail[0] and new_head[1] == self.tail[1])):
            self.running = False
            return reward - 2
        
        #Manhattan distance calculation to see if snake is closer to food
        new_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
        old_dist = abs(self.head[0] - self.food[0]) + abs(self.head[1] - self.food[1])
        
        if new_dist < old_dist:
            reward += .003 #small reward but still overall negative to encourage faster solves

        #check if new_head is over food
        if new_head[0] == self.food[0] and new_head[1] == self.food[1]:
            reward += 1
        
        #update next_tail_dict
        self.next_tail_dict[tuple(self.head)] = new_head
        #update head
        self.head = new_head
        
        # check if head is over food
        if self.head[0] == self.food[0] and self.head[1] == self.food[1]:
            self.length += 1
            
            self.food_board[self.food[0]][self.food[1]] = 0
            # select new food location that is not a snake body
            zeros = np.argwhere(self.snake_board == 0)
            
            if len(zeros) == 0:
                reward += 10
                state = self.get_state()
                self.running = False
                return state, reward
            
            i = np.random.randint(len(zeros))
            x, y = zeros[i]
            
            self.food[0] = x
            self.food[1] = y
            self.food_board[x][y] = 1
            
            return reward
        
        #update tail pointer if not over food
        self.tail = self.next_tail_dict[tuple(self.tail)]
        
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
        
        boards = np.stack([self.snake_board, snake_head_board, self.food_board])
        length = self.length / (self.size * self.size) # normalize length value between 0 and 1
        return boards, self.direction, length