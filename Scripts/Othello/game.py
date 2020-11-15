from Board import Board, Signal
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
BOARD_COLORS["ALT_BOARD"] = (0, 64, 0)
BOARD_COLORS["WHITE"] = (255, 255, 255)
BOARD_COLORS["GAME_OVER"] = (128, 128, 0)
BOARD_COLORS["ALT_GAME_OVER"] = (16, 16, 0)

class Game():

    def __init__(self):
        self.board = Board()
        self.screen = pygame.display.set_mode((256, 256 + 32))
        self.mousepressed = False
        self.font = pygame.font.Font('Scripts/Othello/pixel.ttf', 16)
        self.reset()

    def reset(self):
        self.running = True
        self.gameover = False
        self.info = ""
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
        self.do(action)

    def do(self, action):
        observation, signal = self.board.step(action)
        self.process_signal(signal)
    
    def undo(self):
        self.board.undo()
        self.info = ""
        if self.gameover:
            self.gameover = False

    def step_random(self):
        action = self.board.sample()
        observation, signal = self.board.step(action)
        self.process_signal(signal)
    
    def process_signal(self, signal: Signal):
        if signal is Signal.GAME_OVER:
            self.gameover = True
            winner = self.board.get_winner()
            white_score = self.board.get_white_score()
            black_score = self.board.get_black_score()
            if winner == 1:
                self.info = "Winner is WHITE: {}/{}".format(white_score, black_score)
            elif winner == -1:
                self.info = "Winner is BLACK: {}/{}".format(white_score, black_score)
            else:
                self.info = "Game is DRAW: {}/{}".format(white_score, black_score)
        elif signal is Signal.ILLEGAL_MOVE:
            self.gameover = True
            self.info = "Illegal Move"
            # Current player is the one that didn't play
            if self.board.player == -1:
                self.info += " ! Winner is WHITE"
            elif self.board.player == 1:
                self.info += " ! Winner is BLACK"

    def render(self):
        # Fill screen with color
        if self.gameover:
            self.screen.fill(BOARD_COLORS["ALT_GAME_OVER"])
        else:
            self.screen.fill(BOARD_COLORS["ALT_BOARD"])

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
        altcolor = (1, 1, 1)
        if self.gameover:
            color = BOARD_COLORS["GAME_OVER"]
            altcolor = BOARD_COLORS["ALT_GAME_OVER"]
        else:
            if self.board.player == 1:
                color = BOARD_COLORS["WHITE"] 
                altcolor = BOARD_COLORS["BLACK"]
            if self.board.player == -1:
                color = BOARD_COLORS["BLACK"]
                altcolor = BOARD_COLORS["WHITE"]
        pygame.draw.rect(self.screen, color, rect)
        
        # Display information
        textsurface = self.font.render(self.info, False, altcolor)
        text_x = 128 - textsurface.get_width() / 2
        text_y = 256 + 16 - textsurface.get_height() / 2
        self.screen.blit(textsurface, (text_x, text_y))

        # Render to screen
        pygame.display.flip()


def main():
    # Init pygame
    pygame.init()
    pygame.font.init() 

    # Run game
    game = Game()
    game.run()

    # Close pygame
    pygame.quit()
    

if __name__ == "__main__":
    main()