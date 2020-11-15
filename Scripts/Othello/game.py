from Board import Board
import pygame

###############################################################################
#                       OTHELLO by Alexis F
###############################################################################
# 
# Click on the grid to place token
# 
# Press RETURN key to play random move
# Press R key to reset the board
# Press BACKSPACE key to undo move
# Press ESCAPE to quit
# 
###############################################################################


# Constants
CELL_SIZE = 32
BOARD_COLORS = {}
BOARD_COLORS["BLACK"] = (0, 0, 0)
BOARD_COLORS["BOARD"] = (0, 128, 0)
BOARD_COLORS["WHITE"] = (255, 255, 255)
BOARD_COLORS["GAME_OVER"] = (128, 128, 0)

class Game():

    def __init__(self):
        self.board = Board()
        self.screen = pygame.display.set_mode((256, 256 + 32))
        self.mousepressed = False
        self.reset()

    def reset(self):
        self.running = True
        self.gameover = False
        self.board.reset()

    def run(self):
        self.running = True
        while self.running:
            self.handle_events()
            self.render()
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    return
                if event.key == pygame.K_RETURN:
                    if not self.gameover:
                        action = self.step_random()
                        self.render()
                if event.key == pygame.K_r:
                    self.reset()
                if event.key == pygame.K_BACKSPACE:
                    self.undo()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.mousepressed: # Mouse already pressed
                    self.mousepressed = True
                else: # Mouse just pressed
                    if not self.gameover:
                        (mouse_x, mouse_y) = pygame.mouse.get_pos()
                        if mouse_x >= 0 and mouse_x <= 256 and mouse_y >= 0 and mouse_y <= 256:
                            mouse_i = mouse_x // CELL_SIZE
                            mouse_j = mouse_y // CELL_SIZE
                            self.cell_clicked(mouse_i, mouse_j)

    def cell_clicked(self, i, j):
        action = (i, j)
        observation, reward, done, info = self.board.step(action)
        if done:
            self.gameover = True
    
    def do(self, action):
        self.memory.append(self.board.state)
    
    def undo(self):
        self.board.undo()
        if self.gameover:
            self.gameover = False

    def step_random(self):
        action = self.board.sample()
        observation, reward, done, info = self.board.step(action)
        if done:
            self.gameover = True

    def render(self):
        if self.gameover:
            self.screen.fill((32, 32, 0))
        else:
            self.screen.fill((0, 64, 0))

        # Draw mouse rect
        (mouse_x, mouse_y) = pygame.mouse.get_pos()
        if mouse_x >= 0 and mouse_x <= 256 and mouse_y >= 0 and mouse_y <= 256:
            mouse_i = mouse_x // CELL_SIZE
            mouse_j = mouse_y // CELL_SIZE
            rect = pygame.Rect(mouse_i * CELL_SIZE, mouse_j * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            color = (255, 0, 0)
            pygame.draw.rect(self.screen, color, rect)

        # Draw board
        cell_x = 0
        cell_y = 0
        for j in range(8):
            cell_x = 0
            for i in range(8):
                cell_id = self.board.get_cell(i, j)
                rect = pygame.Rect(cell_x + 1, cell_y + 1, CELL_SIZE - 2, CELL_SIZE - 2)
                color = (0, 0, 0)
                if cell_id == 0:
                    if self.gameover:
                        color = BOARD_COLORS["GAME_OVER"]
                    else:
                        color = BOARD_COLORS["BOARD"]
                elif cell_id == 1:
                    color = BOARD_COLORS["WHITE"]
                elif cell_id == -1:
                    color = BOARD_COLORS["BLACK"]
                pygame.draw.rect(self.screen, color, rect)

                cell_x += CELL_SIZE
            cell_y += CELL_SIZE
        
        # Display player
        rect = pygame.Rect(2, 256 + 2, 256 - 4, 32 - 4)
        color = (0, 0, 0)
        if self.board.player == 1:
            color = BOARD_COLORS["WHITE"] 
        if self.board.player == -1:
            color = BOARD_COLORS["BLACK"]
        pygame.draw.rect(self.screen, color, rect)
        

        pygame.display.flip()


def main():
    # Init pygame
    pygame.init()

    # Run game
    game = Game()
    game.run()

    # Close pygame
    pygame.quit()
    

if __name__ == "__main__":
    main()