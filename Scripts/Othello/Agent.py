from Board import Board, Signal

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        # Input (2, 8, 8)
        self.input_layer = nn.Conv2d(2, 128, 3, padding=1)
        self.hidden_layer_1 = nn.Conv2d(128, 128, 3, padding=1)
        self.hidden_layer_21 = nn.Conv2d(128, 128, 3, padding=1)
        # self.hidden_layer_22 = nn.Conv2d(128, 128, 3, padding=1)
        # self.hidden_layer_23 = nn.Conv2d(128, 128, 3, padding=1)
        self.hidden_layer_3 = nn.Linear(128*8*8, 128*8)
        self.hidden_layer_4 = nn.Linear(128*8, 128)
        self.output_layer = nn.Linear(128, 65)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer_1(x))
        x = F.relu(self.hidden_layer_21(x))
        # x = F.relu(self.hidden_layer_22(x))
        # x = F.relu(self.hidden_layer_23(x))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.hidden_layer_3(x))
        x = F.relu(self.hidden_layer_4(x))
        output = self.output_layer(x)
        return output
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Memory(object):
    # def __init__(self, max_memory=100):
    def __init__(self, max_memory=512):
        self.max_memory = max_memory  # maximum elements stored
        self.memory = list()  # initialize the memory

    def size(self):
        return len(self.memory)

    def remember(self, m):
        if len(self.memory) <= self.max_memory:  # if not full
            self.memory.appdone(m)  # store element m at the end
        else:
            self.memory.pop(0)  # remove the first element
            self.memory.append(m)  # store element m at the end

    def random_access(self, batch_size):
        return random.sample(self.memory, batch_size)  # random sample, from memory, of size batch_size


class AgentTorch():
    def __init__(self, epsilon=0.3, discount=0.99, batch_size=50, learning_rate=0.3):
        self.epsilon = epsilon
        self.memory = Memory()
        # Discount for Q learning (gamma)
        self.discount = discount
        self.batch_size = batch_size
        self.model = Model()
        self.model.to(device)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

    def set_epsilon(self, e):
        self.epsilon = e

    def act(self, s, train=True):
        """ This function should return the next action to do:
        an integer between 0 and 65 (not included) with a random exploration of epsilon"""
        action = 0
        if train:
            if np.random.rand() <= self.epsilon:
                action = torch.randint(0, 65, (1,))
            else:
                action = self.learned_act(s)
        else:  # in some cases, this can improve the performance.. remove it if poor performances
            action = self.learned_act(s)
        return action

    def learned_act(self, s):
        """ Act via the policy of the agent, from a given observation s
        it proposes an action a"""
        output = self.model(s.to(device))
        a = output.view(-1).argmax()
        return a

    def reinforce(self, s, n_s, a, r, game_over_):
        """ 
        This function is the core of the learning algorithm.
        Its goal is to learn a policy.
        
        It takes as an input the current observation s_, the next observation 
        n_s_, the action a_ used to move from s_ to n_s_, and the reward r_.

        Two steps: first memorize the s, second learn from the pool
        - Memorize current state given previous state (s), action (a) and current state (n_s)
        - Learn from the pool
        """

        self.memory.remember([s, n_s, a, r, game_over_])
    
        # input_s: list of every states used in the learning process here
        input_s = torch.zeros((self.batch_size, 2, 8, 8))

        # target_q: the predicted future reward for each action
        # We expect target_q of the base sate to have positive values for 
        # values (20, 29, 34, 43) and negative for any other action
        target_q = torch.zeros((self.batch_size, 65))

        if self.memory.size() < self.batch_size:  # if not enough elements in memory we do nothing
            return 1e5  # unknown (loss)

        # For each sample of state, update target_q matrix of the given state
        samples = self.memory.random_access(self.batch_size)
        for i_batch in range(self.batch_size):
            s, next_s, a, r, end = samples[i_batch]  # observation, next_observation, action, reward, game_over
            input_s[i_batch] = torch.tensor(s, dtype=torch.float)
            if end:
                target_q[i_batch, a] = r
            else:
                # compute max_a Q(nex_observation, a) using the model
                y_pred = self.model(torch.tensor(next_s, dtype=torch.float).unsqueeze(0).to(device)).to("cpu")
                Q_next_observation = torch.max(y_pred)
                # r + gamma * max_a Q(nex_observation, a)
                target_q[i_batch, a] = r + self.discount * Q_next_observation

        # HINT: Clip the target to avoid exploding gradients.. -- clipping is a bit tighter
        target_q = torch.clip(target_q, -3, 3)

        # Train the model on the batch
        input_data = torch.tensor(input_s) #[input_s[i] for i in range(self.batch_size)])
        loss = self.train_on_batch(input_data, target_q)

        return loss

    def train_on_batch(self, x, y):
        self.model.train()
        y_pre = self.model(torch.tensor(x, dtype=torch.float).to(device))
        loss = self.criterion(torch.tensor(y, dtype=torch.float).to(device), y_pre)
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


def reward_from_signal(signal: Signal, board: Board):
    reward = 0
    if signal is Signal.ILLEGAL_MOVE:
            reward = -1
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

def game_over_from_signal(signal: Signal, board: Board):
    game_over = signal in [Signal.GAME_OVER, Signal.ILLEGAL_MOVE]
    return game_over

def encode_action(a):
    if a == 64:
        return None
    else:
        i = a % 8
        j = a // 8
        return (i, j)

def train(agent: AgentTorch, board: Board, epoch):
    # Number of won games
    score = 0
    loss = 0

    # for e in tqdm.tqdm(range(epoch)):
    for e in range(epoch):
        # At each epoch, we restart to a fresh game and get the initial observation
        observation, signal = board.reset()
        observation = torch.tensor(observation)

        # This assumes that the games will terminate
        game_over = False

        win = 0
        lose = 0

        itour = 0
        while not game_over:
            # Keep old observation in memory
            prev_observation = observation

            # Ask the agent an action
            action = agent.act(torch.tensor(observation, dtype=torch.float).unsqueeze(0))
            
            # Perform the action
            observation, signal = board.step(encode_action(action))
            observation = torch.tensor(observation)

            # Reward attribution
            reward = reward_from_signal(signal) + itour

            game_over = game_over_from_signal(signal)
            if signal is not Signal.ILLEGAL_MOVE:
                print(f"Action: {action}: {signal}; {game_over}")

            # Update the counters
            if reward > 0:
                win = win + reward
            if reward < 0:
                lose = lose - reward

            if game_over:
                # Apply the reinforcement strategy
                loss = agent.reinforce(prev_observation, observation, action, reward, game_over)
                break
            else:
                # Random BLACK agent
                black_action = board.sample()
                observation, signal = board.step(black_action)
                game_over = game_over_from_signal(signal)
                loss = agent.reinforce(prev_observation, observation, action, reward, game_over)
                itour += 1

        # Update stats
        score += win - lose

        print(f"Epoch {e}/{epoch}, loss {round(np.float64(loss), 4)}, win/lose count {win}/{lose} ({win - lose})")
    
    observation, signal = board.reset()
    torch_observation = torch.tensor(observation, dtype=torch.float).unsqueeze(0).to(device)
    output = agent.model(torch_observation)
    print(output.to("cpu"))
    action = agent.act(torch.tensor(observation, dtype=torch.float).unsqueeze(0))
    observation, signal = board.step(encode_action(action))

    black_action = board.sample()
    observation, signal = board.step(black_action)

    torch_observation = torch.tensor(observation, dtype=torch.float).unsqueeze(0).to(device)
    output = agent.model(torch_observation)
    print(output.to("cpu"))



    
    # if done:
    # else:
    #     # compute max_a Q(nex_observation, a) using the model
    #     y_pred = self.model(torch.tensor(next_s, dtype=torch.float).unsqueeze(0).to(device)).to("cpu")
    #     Q_next_observation = torch.max(y_pred)
    #     # r + gamma * max_a Q(nex_observation, a)
    #     target_q[i_batch, a] = r + self.discount * Q_next_observation



if __name__ == '__main__':
    # epoch = 2048# 100000
    # board = Board()
    # agent = AgentTorch(batch_size=64)
    tests()
    # train(agent, board, epoch)
