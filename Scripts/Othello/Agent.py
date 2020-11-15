from Board import Board, Signal

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import random


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        # Input (2, 8, 8)
        self.input_layer = nn.Conv2d(2, 128, 3, padding=1)
        self.hidden_layer_1 = nn.Conv2d(128, 128, 3, padding=1)
        self.hidden_layer_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.hidden_layer_3 = nn.Conv2d(128, 128, 3, padding=1)
        self.hidden_layer_4 = nn.Conv2d(128, 128, 3, padding=1)
        self.output_layer = nn.Conv2d(128, 1, 3, padding=1)
        self.skip_turn = nn.Linear(128*8*8, 1)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer_1(x))
        x = F.relu(self.hidden_layer_2(x))
        x = F.relu(self.hidden_layer_3(x))
        x = F.relu(self.hidden_layer_4(x))
        output = F.softmax(self.output_layer(x))
        skip = F.sigmoid(self.skip_turn(torch.flatten(x)))
        return skip, output


class Memory(object):
    def __init__(self, max_memory=1000):
        self.max_memory = max_memory  # maximum elements stored
        self.memory = list()  # initialize the memory

    def size(self):
        return len(self.memory)

    def remember(self, m):
        if len(self.memory) <= self.max_memory:  # if not full
            self.memory.append(m)  # store element m at the end
        else:
            self.memory.pop(0)  # remove the first element
            self.memory.append(m)  # store element m at the end

    def random_access(self, batch_size):
        return random.sample(self.memory, batch_size)  # random sample, from memory, of size batch_size


class AgentTorch():
    def __init__(self, epsilon=0.1, discount=0.99, batch_size=50):
        self.epsilon = epsilon
        self.memory = Memory()
        # Discount for Q learning (gamma)
        self.discount = discount
        self.batch_size = batch_size
        # self.model = Model(criterion=nn.MSELoss())
        self.model = Model()
        # self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.03)

    def set_epsilon(self, e):
        self.epsilon = e

    def act(self, s, train=True):
        """ This function should return the next action to do:
        an integer between 0 and 4 (not included) with a random exploration of epsilon"""
        if train:
            if np.random.rand() <= self.epsilon:
                action = tuple(torch.randint(0, 8, (2,)).tolist())
            else:
                action = self.learned_act(s)
        else:  # in some cases, this can improve the performance.. remove it if poor performances
            action = self.learned_act(s)

        return action

    def learned_act(self, s):
        """ Act via the policy of the agent, from a given observation s
        it proposes an action a"""
        skip, output = self.model(s)
        # print("############################################")
        # print(skip)
        if skip > 0.5: # Skip turn
            return None
        else: # Play action
            m = output.view(-1).argmax()
            i = (m % 8).view(-1, 1).item()
            j = (m // 8).view(-1, 1).item()
            # print(output)
            # print("i: ", i)
            # print("j: ", j)
            return (i, j)

    def reinforce(self, s, n_s, a, r, game_over_):
        """ This function is the core of the learning algorithm.
        It takes as an input the current observation s_, the next observation n_s_
        the action a_ used to move from s_ to n_s_ and the reward r_.

        Its goal is to learn a policy.
        """
        # Two steps: first memorize the observations, second learn from the pool

        # 1) memorize
        self.memory.remember([s, n_s, a, r, game_over_])
    
        # 2) Learn from the pool
        input_observations = torch.zeros((self.batch_size, 2, 8, 8))
        target_q = torch.zeros((self.batch_size, 8, 8))

        if self.memory.size() < self.batch_size:  # if not enough elements in memory we do nothing
            return 1e5  # unknown (loss)

        samples = self.memory.random_access(self.batch_size)

        for i in range(self.batch_size):
            holder, next_s, a, r, end = samples[i]  # observation, next_observation, action, reward, game_over
            input_observations[i] = torch.tensor(holder)

            # update the target
            if end:
                target_q[i, a] = r
            else:
                # compute max_a Q(nex_observation, a) using the model
                (pred_skip, pred_action) = self.model(torch.tensor(next_s, dtype=torch.float).unsqueeze(0))
                Q_next_observation = torch.max(pred_action)

                # r + gamma * max_a Q(nex_observation, a)
                target_q[i, a] = r + self.discount * Q_next_observation

        # HINT: Clip the target to avoid exploding gradients.. -- clipping is a bit tighter
        target_q = torch.clip(target_q, -3, 3)

        # train the model on the batch
        input_data = torch.tensor(input_observations) #[input_observations[i] for i in range(self.batch_size)])
        loss = self.train_on_batch(input_data, target_q)

        return loss

    def train_on_batch(self, x, y):
        self.model.train()
        (skip_pre, y_pre) = self.model(torch.tensor(x, dtype=torch.float))
        loss = self.criterion(torch.tensor(y, dtype=torch.float), y_pre)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self):
        """ This function returns basic stats if applicable: the
        loss and/or the model"""
        pass

    def load(self):
        """ This function allows to restore a model"""
        pass


def reward_from_signal(signal: Signal):
    reward = 0
    if signal is Signal.ILLEGAL_MOVE:
            reward = -10000
    elif signal is Signal.VALID_MOVE:
        reward = 1
    else: # Game over
        winner = board.get_winner()
        if winner == 1: # White
            reward = 100
        elif winner == -1: # Black
            reward = -100
        else: # Draw
            reward = -50
    return reward

def game_over_from_signal(signal: Signal):
    game_over = signal in [Signal.GAME_OVER, Signal.ILLEGAL_MOVE]
    return game_over

def train(agent: AgentTorch, board: Board, epoch):
    # Number of won games
    score = 0
    loss = 0

    for e in tqdm.tqdm(range(epoch)):
        # At each epoch, we restart to a fresh game and get the initial observation
        observation, signal = board.reset()
        observation = torch.tensor(observation)

        # This assumes that the games will terminate
        game_over = False

        win = 0
        lose = 0

        while not game_over:
            # Keep old observation in memory
            prev_observation = observation

            # Ask the agent an action
            action = agent.act(torch.tensor(observation, dtype=torch.float).unsqueeze(0))
            
            # Perform the action
            observation, signal = board.step(action)
            observation = torch.tensor(observation)
            # print(f"Action: {action}: {signal}")

            # Reward attribution
            reward = reward_from_signal(signal)
            game_over = game_over_from_signal(signal)

            # Update the counters
            if reward > 0:
                win = win + reward
            if reward < 0:
                lose = lose - reward

            # Apply the reinforcement strategy
            loss = agent.reinforce(prev_observation, observation, action, reward, game_over)

            # Random BLACK agent
            action = board.sample()
            observation, signal = board.step(action)
            game_over = game_over_from_signal(signal)

        # Update stats
        score += win - lose

        print(f"Epoch {e}/{epoch}, loss {round(np.float64(loss), 4)}, win/lose count {win}/{lose} ({win - lose})")


if __name__ == '__main__':
    epoch = 1000
    board = Board()
    agent = AgentTorch(batch_size=1)
    train(agent, board, epoch)
