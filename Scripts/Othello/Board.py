import numpy as np
import random


class Board():
    """
    Othello is a 8x8 grid
    [Black ; empty; White] -> [-1; 0; 1]
    """
    def __init__(self):
        self.player = 1 # White first
        self.state = np.zeros((8, 8), dtype=int) # Grid representation
        self.reset() # Clear & populates central cells

    def reset(self):
        self.state = np.zeros((8, 8), dtype=int)
        self.set_cell(3, 3, 1)
        self.set_cell(4, 4, 1)
        self.set_cell(3, 4, -1)
        self.set_cell(4, 3, -1)
        self.player = 1 # White first
    
    def get_cell(self, i, j):
        return self.state[i, j]

    def set_cell(self, i, j, val):
        self.state[i, j] = val
    
    def flip_cell(self, i, j):
        if self.state[i, j] == 0:
            print("<!> Flipping empty cell <!>")
        self.state[i, j] = -self.state[i, j]

    def step(self, action):
        (i, j) = action

        # Get nb of cells flipped by the action
        flips = []
        for (di, dj) in [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]:
            flips += self.get_flips_in_dir(i, j, di, dj)

        # If none it's illegal
        if len(flips) == 0:
            return self.state, -100, True, "ILLEGAL_MOVE"

        # If legal 
        self.set_cell(i, j, self.player)
        print(flips)
        for (i, j) in flips:
            self.flip_cell(i, j)
        
        # Check end of game
        self.changes_player()
        if not self.can_player_move():
            self.changes_player()
            if not self.can_player_move():
                return self.state, self.player * self.get_winner() * 100, True, "GAME OVER"

        return self.state, 1, False, ""
    
    def changes_player(self):
        # Changes the token color (Black 1 or White -1)
        self.player = -self.player
    
    def can_player_move(self):
        actions = self.get_possible_actions()
        return len(actions) != 0

    def get_possible_actions(self):
        actions = []
        for i in range(8):
            for j in range(8):
                if self.get_cell(i, j) == 0:
                    for (di, dj) in [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]:
                        _flips = self.get_flips_in_dir(i, j, di, dj)
                        if len(_flips) != 0:
                            actions.append((i, j))
        return actions
    
    def sample(self):
        actions = self.get_possible_actions()
        return random.choice(actions)

    def get_flips_in_dir(self, i, j, di, dj):
        # Move a step forward in the direction
        i += di
        j += dj
        # Compute the list of (i, j) to flip
        flips = []
        while i >= 0 and i < 8 and j >= 0 and j < 8:
            cell_id = self.get_cell(i, j)
            if cell_id == -self.player:
                flips.append((i, j)) # Possible flip
            elif cell_id == self.player:
                return flips
            else:
                return []
            # Move forward
            i += di
            j += dj
        return []

    def get_winner(self):
        white = (self.state == 1).sum()
        black = (self.state == -1).sum()
        # empty = (self.state == 0).sum()
        if white > black:
            return 1
        elif white < black:
            return -1
        else:
            return 0

if __name__ == "__main__":
    env = Board()
    env.reset()
    done = False
    i = 0
    while not done:
        i += 1
        action = env.sample()
        print("action:", action)
        print("Move ", i, "of player", env.player)
        observation, reward, done, info = env.step(action)
        print(reward, info)
        print(observation)
        if done:
            break