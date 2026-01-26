import numpy as np

class Snake_env():
    def __init__(self, size=200):
        self.snake_board = np.zeros((size, size))
        self.snake_board[0][0] = 1
        #head and tail pointers to track which parts of snake to update
        self.head = np.zeros((2, 1))
        self.tail = np.zeros((2, 1))
        self.direction = np.zeros(4)
        self.direction[0] = 1  # 0:North, 1:East, 2:South, 3:West
        self.curr_direction = 0 # index of direction array
        self.food_board = np.zeros((size, size))
        x, y = np.random.randint(20, 100), np.random.randint(0, 100)
        self.food_board[x][y] = 1
        self.next_tail_dict = {}
    
    def set_direction(self, turn):
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
        self.set_direction[direction]
        
        