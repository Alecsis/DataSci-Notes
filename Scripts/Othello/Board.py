import numpy as np
from enum import Enum
import random

class Signal(Enum):
    BEGIN_GAME = 1,
    VALID_MOVE = 2,
    ILLEGAL_MOVE = 3,
    GAME_OVER = 4


class Board():
    """
    Othello is a 8x8 grid
    [Black ; empty; White] -> [-1; 0; 1]
    """

    def __init__(self):
        self.player = 1 # White first
        self.state = np.zeros((8, 8), dtype=int) # Grid representation
        self.memory = [] # To undo moves
        self.reset() # Clear & populates central cells

    def reset(self):
        self.state = np.zeros((8, 8), dtype=int)
        self.set_cell(3, 3, 1)
        self.set_cell(4, 4, 1)
        self.set_cell(3, 4, -1)
        self.set_cell(4, 3, -1)
        self.player = 1 # White first
        begin_signal = Signal.BEGIN_GAME
        self.memory = [(np.copy(self.state), begin_signal)]
        return self.get_observation(), begin_signal
    
    def get_cell(self, c, l):
        return self.state[c, l]

    def set_cell(self, c, l, val):
        self.state[c, l] = val
    
    def flip_cell(self, c, l):
        self.state[c, l] = -self.state[c, l]

    def step(self, action):
        # Save state in memory
        self.memory.append((np.copy(self.state), self.player))
        result_signal = self.do(action)
        return self.get_observation(), result_signal
    
    def do(self, action) -> Signal:
        # If player skip turn when he could move
        if action == None:
            if self.can_player_move():
                return Signal.ILLEGAL_MOVE
            else:
                self.changes_player()
                return Signal.VALID_MOVE
        # Player didn't skip turn, get coordinates
        (c, l) = action
        # Check if player didn't move on an occupied cell
        if self.get_cell(c, l) != 0:
            return Signal.ILLEGAL_MOVE
        # Get nb of cells flipped by the action
        flips = []
        for (di, dj) in [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]:
            flips += self.get_flips_in_dir(c, l, di, dj)
        # If flips is empty then the move is useless and qualified illegal
        if len(flips) == 0:
            return Signal.ILLEGAL_MOVE
        # If flips is not empty, apply and flip tokens 
        self.set_cell(c, l, self.player)
        for (c, l) in flips:
            self.flip_cell(c, l)
        # Il none of the players can move, game is over
        self.changes_player()
        if not self.can_player_move():
            self.changes_player()
            if not self.can_player_move():
                return Signal.GAME_OVER
        # If everything is good then send valid signal
        return Signal.VALID_MOVE
    
    def undo(self):
        if len(self.memory) != 0:
            (state, player) = self.memory.pop()
            self.player = player
            self.state = state

    def changes_player(self):
        # Changes the token color (Black 1 or White -1)
        self.player = -self.player
    
    def can_player_move(self):
        actions = self.get_possible_actions()
        return len(actions) != 0

    def get_possible_actions(self):
        # For each empty cell, compute if any move can flip tokens
        # If move is valid, add coordinates (c, l) in [actions]
        actions = []
        for c in range(8):
            for l in range(8):
                if self.get_cell(c, l) == 0:
                    for (di, dj) in [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]:
                        _flips = self.get_flips_in_dir(c, l, di, dj)
                        if len(_flips) != 0:
                            actions.append((c, l))
        return actions
    
    def sample(self):
        # Randomly choose in possible actions
        actions = self.get_possible_actions()
        if len(actions) == 0:
            return None
        # return actions[0]
        return random.choice(actions)

    def get_flips_in_dir(self, c, l, di, dj):
        # Move a step forward in the direction
        c += di
        l += dj
        # Compute the list of (c, l) to flip
        flips = []
        while c >= 0 and c < 8 and l >= 0 and l < 8:
            cell_id = self.get_cell(c, l)
            if cell_id == -self.player:
                flips.append((c, l)) # Possible flip
            elif cell_id == self.player:
                return flips
            else:
                return []
            # Move forward
            c += di
            l += dj
        return []

    def get_winner(self):
        white = self.get_white_score()
        black = self.get_black_score()
        if white > black:
            return 1
        elif white < black:
            return -1
        else:
            return 0
    
    def get_observation(self):
        observation = np.zeros((2, 8, 8), dtype=int)
        observation[0] = (self.state == 1)
        observation[1] = (self.state == -1)
        return observation
    
    def render(self, state=None):
        if not state:
            state = self.state
        print("---" * 8)
        for l in range(8):
            line = ""
            for c in range(8):
                cell_id = state[c, l]
                if cell_id == 1:
                    line += "X\t"
                elif cell_id == -1:
                    line += "O\t"
                else:
                    line += f".\t"
                    # line += f"{c+8*l}\t"
            print(line)
        print("---" * 8)

    def get_white_score(self):
        white = (self.state == 1).sum()
        return white

    def get_black_score(self):
        black = (self.state == -1).sum()
        return black        


if __name__ == "__main__":
    # Instanciate board
    board = Board()

    # Init game
    observation, signal = board.reset()
    board.render()

    while True:
        # Sample action
        action = board.sample()
        observation, signal = board.step(action)
        if signal is Signal.GAME_OVER:
            break
    board.render()

    winner = board.get_winner()
    white_score = board.get_white_score()
    black_score = board.get_black_score()
    if winner == 1:
        print("Winner is WHITE: {}/{}".format(white_score, black_score))
    elif winner == -1:
        print("Winner is BLACK: {}/{}".format(white_score, black_score))
    else:
        print("Game is DRAW: {}/{}".format(white_score, black_score))
        