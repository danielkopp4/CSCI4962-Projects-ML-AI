import torch
from torch import nn
import torch.functinoal as F
from torch import optim
import matlotlib.pyplot as plt


class Actor(nn.Module):
    def __init__(self, state_size, action_size, learning_rate):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        self.fc1 = nn.Linear(self.state_size, 128)
        self.fc2 = nn.Linear(128, self.action_size)

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.softmax(self.fc2(x), dim=1)
        return x

    def save_model(self, path):
        torch.save(self.state_dict(), path + '_actor.pth')

    def load_model(self, path):
        self.load_state_dict(torch.load(path + '_actor.pth'))


class Critic(nn.Module):
    def __init__(self, state_size, learning_rate):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.learning_rate = learning_rate

        self.fc1 = nn.Linear(self.state_size, 128)
        self.fc2 = nn.Linear(128, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        return x

    def save_model(self, path):
        torch.save(self.state_dict(), path + '_critic.pth')

    def load_model(self, path):
        self.load_state_dict(torch.load(path + '_critic.pth'))
        
        
class Categorical:
    def __init__(self, probs):
        self.probs = probs
        self.distribution = Categorical(probs)

    def sample(self):
        return self.distribution.sample()

    def log_prob(self, x):
        return self.distribution.log_prob(x)

    def entropy(self):
        return self.distribution.entropy()

class PPOModel:
    def __init__(self, environment, learning_rate, gamma, lam, epochs, batch_size,
                 epsilon, epsilon_decay, epsilon_min, render, save_path):
        self.env = environment
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lam = lam
        self.epochs = epochs
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.render = render
        self.save_path = save_path

        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

        self.actor = Actor(self.state_size, self.action_size, self.learning_rate)
        self.critic = Critic(self.state_size, self.learning_rate)

        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
        self.entropies = []
        self.advantages = []

        self.episode_rewards = []
        self.episode_lengths = []

    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action_probs = self.actor(state)
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()
        self.log_probs.append(action_distribution.log_prob(action))
        self.entropies.append(action_distribution.entropy())
        return action.item()

    def get_value(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        value = self.critic(state)
        self.values.append(value)
        return value

    def update_policy(self):
        R = 0
        returns = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        for log_prob, entropy, value, r in zip(self.log_probs, self.entropies, self.values, returns):
            self.advantages.append(r - value.item())
            policy_loss = -log_prob * r + 0.2 * entropy
            value_loss = F.smooth_l1_loss(value, r)
            self.actor.optimizer.zero_grad()
            self.critic.optimizer.zero_grad()
            policy_loss.backward()
            value_loss.backward()
            self.actor.optimizer.step()
            self.critic.optimizer.step()

        self.states, self.actions, self.rewards, self.dones, self.values, self.log_probs, self.entropies, self.advantages = [], [], [], [], [], [], [], []

    def train(self):
        for epoch in range(self.epochs):
            state = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            while not done:
                if self.render:
                    self.env.render()
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.states.append(state)
                self.actions.append(action)
                self.rewards.append(reward)
                self.dones.append(done)
                state = next_state
                episode_reward += reward
                episode_length += 1
                if done:
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)
                    self.update_policy()
                    break

            if epoch % 1000 == 0:
                print('Epoch: {}/{}'.format(epoch, self.epochs))
                print('Episode Reward: {:0.03f}'.format(episode_reward))
                print('Episode Length: {:0.03f}'.format(episode_length))
                print('--------------------------------')

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        self.actor.save_model(self.save_path)
        self.critic.save_model(self.save_path)

    def test(self):
        self.actor.load_model(self.save_path)
        self.critic.load_model(self.save_path)
        state = self.env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        while not done:
            if self.render:
                self.env.render()
            action = self.get_action(state)
            next_state, reward, done, _ = self.env.step(action)
            state = next_state
            episode_reward += reward
            episode_length += 1
            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                break

        print('Episode Reward: {}'.format(episode_reward))
        print('Episode Length: {}'.format(episode_length))
        print('--------------------------------')
    