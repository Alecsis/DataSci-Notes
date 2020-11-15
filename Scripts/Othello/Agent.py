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
        self.hidden_layer_22 = nn.Conv2d(128, 128, 3, padding=1)
        self.hidden_layer_23 = nn.Conv2d(128, 128, 3, padding=1)
        self.hidden_layer_3 = nn.Linear(128*8*8, 128*8)
        self.hidden_layer_4 = nn.Linear(128*8, 128)
        self.output_layer = nn.Linear(128, 65)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer_1(x))
        x = F.relu(self.hidden_layer_21(x))
        x = F.relu(self.hidden_layer_22(x))
        x = F.relu(self.hidden_layer_23(x))
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
            self.memory.append(m)  # store element m at the end
        else:
            self.memory.pop(0)  # remove the first element
            self.memory.append(m)  # store element m at the end

    def random_access(self, batch_size):
        return random.sample(self.memory, batch_size)  # random sample, from memory, of size batch_size


class AgentTorch():
    def __init__(self, epsilon=0.3, discount=0.99, batch_size=50):
        self.epsilon = epsilon
        self.memory = Memory()
        # Discount for Q learning (gamma)
        self.discount = discount
        self.batch_size = batch_size
        self.model = Model()
        self.model.to(device)
        self.criterion = torch.nn.MSELoss()
        # self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.3)

    def set_epsilon(self, e):
        self.epsilon = e

    def act(self, s, train=True):
        """ This function should return the next action to do:
        an integer between 0 and 4 (not included) with a random exploration of epsilon"""
        if train:
            if np.random.rand() <= self.epsilon:
                # self.epsilon *= 0.9999
                action = tuple(torch.randint(0, 8, (2,)).tolist())
            else:
                action = self.learned_act(s)
        else:  # in some cases, this can improve the performance.. remove it if poor performances
            action = self.learned_act(s)

        return action

    def learned_act(self, s):
        """ Act via the policy of the agent, from a given observation s
        it proposes an action a"""
        output = self.model(s.to(device))
        m = output.view(-1).argmax()
        # print("m:")
        # print(m.to("cpu").item())
        if m == 64:
            return None
        else: # Play action
            i = (m % 8).view(-1, 1).item()
            j = (m // 8).view(-1, 1).item()
            # print(i, j)
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
        target_q = torch.zeros((self.batch_size, 65))

        if self.memory.size() < self.batch_size:  # if not enough elements in memory we do nothing
            return 1e5  # unknown (loss)

        samples = self.memory.random_access(self.batch_size)

        for i_batch in range(self.batch_size):
            s, next_s, a, r, end = samples[i_batch]  # observation, next_observation, action, reward, game_over
            input_observations[i_batch] = torch.tensor(s)

            # update the target
            action_id = 0
            if a == None:
                action_id = 64
            else:
                (i, j) = a
                action_id = i + 8 * j

                
            if end:
                target_q[i_batch, action_id] = r
            else:
                # compute max_a Q(nex_observation, a) using the model
                y_pred = self.model(torch.tensor(next_s, dtype=torch.float).unsqueeze(0).to(device)).to("cpu")
                Q_next_observation = torch.max(y_pred)
                
                # r + gamma * max_a Q(nex_observation, a)
                target_q[i_batch, action_id] = r + self.discount * Q_next_observation
                # if action_id in [20, 29, 34, 43]:
            # print(action_id, target_q[i_batch, action_id].item())

        # HINT: Clip the target to avoid exploding gradients.. -- clipping is a bit tighter
        target_q = torch.clip(target_q, -3, 3)


        # train the model on the batch
        input_data = torch.tensor(input_observations) #[input_observations[i] for i in range(self.batch_size)])
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


def reward_from_signal(signal: Signal):
    reward = 0
    if signal is Signal.ILLEGAL_MOVE:
            reward = -1
    elif signal is Signal.VALID_MOVE:
        reward = 5
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
            observation, signal = board.step(action)
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
    observation, signal = board.step(action)

    black_action = board.sample()
    observation, signal = board.step(black_action)

    torch_observation = torch.tensor(observation, dtype=torch.float).unsqueeze(0).to(device)
    output = agent.model(torch_observation)
    print(output.to("cpu"))


if __name__ == '__main__':
    epoch = 2048# 100000
    board = Board()
    agent = AgentTorch(batch_size=32)
    train(agent, board, epoch)
