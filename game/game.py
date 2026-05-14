import pygame
#from neural_network.nn import NN
from snake_env import Snake_Env
import numpy as np

class Game():
    def __init__(self):
        self.size = 20
        self.game = Snake_Env(self.size)
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
            
            if(self.game.running):
                self.game.step(self.direction)
                self.snake_board = self.game.snake_board
                self.food_board = self.game.food_board
            
            self.screen.fill((0, 0, 0))  # black background
            for i in range(self.size):
                for j in range(self.size):
                    if(self.food_board[i, j] == 1):
                        rect = pygame.Rect(i * self.square_width + self.width_offset, self.height - (j * self.square_height + self.square_height), self.square_width - 10, self.square_height - 10)
                        pygame.draw.rect(self.screen, (255, 0, 0), rect)
                    elif(self.snake_board[i, j] == 1):
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
        self.game = Snake_Env(self.size)
        self.run()
    
if __name__ == "__main__":
    game = Game()
    game.run()