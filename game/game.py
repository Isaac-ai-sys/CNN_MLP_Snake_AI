import pygame
from game.snake_env import VectorizedSnakeEnv
from neural_network.search import search
from neural_network.nn import NN
try:
    import cupy as np
except:
    import numpy as np

class Game():
    def __init__(self):
        self.size = 20
        self.game = VectorizedSnakeEnv(1, self.size)
        pygame.init()
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.width, self.height = min(self.screen.get_width(), self.screen.get_height()), min(self.screen.get_width(), self.screen.get_height())
        self.square_width = self.width / self.size
        self.square_height = self.height / self.size
        self.height_offset = (self.screen.get_height() - self.height) / 2
        self.width_offset = (self.screen.get_width() - self.width) / 2
        self.left_border = self.width_offset
        self.right_border = self.screen.get_width() - self.width_offset
        self.clock = pygame.time.Clock()
        self.direction = np.zeros(4)
        self.direction[0] = 1
        
        BOARD_SIZE = 20
        KERNEL_SIZE = 3
        CONV_DEPTH = 8
        POOL_SIZE = 2

        conv1_out = BOARD_SIZE - KERNEL_SIZE + 1
        pool_out = conv1_out // POOL_SIZE
        conv2_out = pool_out - KERNEL_SIZE + 1

        flat_size = (CONV_DEPTH * 2) * conv2_out * conv2_out
        dense_input = flat_size + 8

        feature_layers = []
        actor_layers = []
        critic_layers = []
        
        self.nn = NN()
        # create feature network
        feature_layers.append(self.nn.create_convolution_layer((3, BOARD_SIZE, BOARD_SIZE), KERNEL_SIZE, CONV_DEPTH)) # output is 18x18x8
        feature_layers.append(self.nn.create_max_pool_layer(2)) # output is 9x9x8
        feature_layers.append(self.nn.create_convolution_layer((CONV_DEPTH, pool_out, pool_out), KERNEL_SIZE, CONV_DEPTH * 2)) # output is 7x7x16
        feature_layers.append(self.nn.create_reshape_layer((CONV_DEPTH * 2, conv2_out, conv2_out), (flat_size, 1)))
        feature_layers.append(self.nn.create_dense_layer(64, dense_input)) # (7x7x16 + 12) x 64 = 50,944 parameters
        self.nn.feature_layers = feature_layers
        
        #create actor network
        actor_layers.append(self.nn.create_dense_layer(32, 64)) # 64x32 = 2048 parameters
        actor_layers.append(self.nn.create_dense_layer(16, 32)) #  16x32 = 512 parameters
        actor_layers.append(self.nn.create_dense_layer(4, 16)) # 16x4 = 64 parameters
        self.nn.actor_layers = actor_layers
        
        #create critic network
        critic_layers.append(self.nn.create_dense_layer(32, 64)) # 64x32 = 2048 parameters
        critic_layers.append(self.nn.create_dense_layer(1, 32)) # 32x1 = parameters
        self.nn.critic_layers = critic_layers

        self.nn.load()
        self.s = search(self.nn)
    
    def run(self):
        running = True
        tick_rate = 10
        while running:
            self.clock.tick(tick_rate)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key == pygame.K_r:
                        self.restart()
                    if event.key == pygame.K_COMMA:
                        tick_rate = max(1, tick_rate - 1)
                    if event.key == pygame.K_PERIOD:
                        tick_rate = min(120, tick_rate + 1)
                    if event.key == pygame.K_UP:
                        self.direction = np.zeros(4)
                        self.direction[0] = 1
                    if event.key == pygame.K_RIGHT:
                        self.direction = np.zeros(4)
                        self.direction[1] = 1
                    if event.key == pygame.K_DOWN:
                        self.direction = np.zeros(4)
                        self.direction[2] = 1
                    if event.key == pygame.K_LEFT:
                        self.direction = np.zeros(4)
                        self.direction[3] = 1
            
            if self.game.running[0]:
                state, direction, length, dx_food, dy_food, env_running = self.game.get_state()
                action, val = self.nn.forward_prop(state, direction, length, dx_food, dy_food, env_running)
                action_idx = np.array([int(action.argmax())])
                self.game.step(action_idx)
                state, direction, length, dx_food, dy_food, env_running = self.game.get_state()
                self.snake_board = self.game.snake_boards[0]
                self.food_board = state[0, 2]
            
            self.screen.fill((0, 0, 0))  # black background
            for i in range(self.size):
                for j in range(self.size):
                    if(self.food_board[i, j] == 1):
                        rect = pygame.Rect(i * self.square_width + self.width_offset, self.height - (j * self.square_height + self.square_height), self.square_width - 10, self.square_height - 10)
                        pygame.draw.rect(self.screen, (255, 0, 0), rect)
                    elif(self.snake_board[i, j] > 0):
                        rect = pygame.Rect(i * self.square_width + self.width_offset, self.height - (j * self.square_height + self.square_height), self.square_width - 10, self.square_height - 10)
                        pygame.draw.rect(self.screen, (0, 255, 0), rect)
            
            #draw borders
            rect = pygame.Rect(self.left_border - 10, 0, 10, self.height)
            pygame.draw.rect(self.screen, (0, 0, 255), rect)
            rect = pygame.Rect(self.right_border - 10, 0, 10, self.height)
            pygame.draw.rect(self.screen, (0, 0, 255), rect)
            
            if(not self.game.running):
                font = pygame.font.Font(None, 300)   # None = default font, 50 = size
                text_surface = font.render("Game Over", True, (255, 255, 255))  # white text
                self.screen.blit(text_surface, (100, 100))
            
            pygame.display.flip() #prints whatever updates you made to screen

        pygame.quit()
        
    def restart(self):
        #start new game
        self.game = VectorizedSnakeEnv(1, self.size)
        self.run()
    
if __name__ == "__main__":
    game = Game()
    game.run()