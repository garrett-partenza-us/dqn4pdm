#imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import numpy as np
import random
import copy
import collections
import typing
from tqdm import tqdm
from collections import deque
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

#environment class
class Environment():
    
    #initialize dataset, a random trajectory, and current cycle
    def __init__(self):
        self.dataset = pd.read_csv('train.csv')
        self.episode = self.get_trajectory(np.random.randint(low=1, high=79, size=1))
        self.cycle = 0
        
    #get random trajectory
    def get_trajectory(self, engine_id):
        return self.dataset[self.dataset.engine_id==engine_id.item()].health_indicator.to_numpy()
    
    #reset environment    
    def reset(self):
        self.cycle = 0
        self.episode = self.get_trajectory(np.random.randint(low=0, high=79, size=1))
        
    #return current state
    def get_state(self):
        return torch.tensor([self.episode[self.cycle]], requires_grad=False)
    
    #take action
    def take_action(self, action):
        if action == 0:
            #failure occurs
            if self.cycle+1 == self.episode.size:
                res = (None, -1, True)
            #continued operation, return 1 and continue episode
            else:
                res = (self.episode[self.cycle+1], self.episode[self.cycle], False)
            #move to next state
            self.cycle+=1
        elif action == 1:
            res = (None, -self.episode[self.cycle], True)
        return res
    
class Transition():
    
    def __init__(self, state, action, state_new, reward, term):
        self.state = state
        self.action = action
        self.state_new = state_new
        self.reward = reward
        self.term = term
        
class PrioritizedReplayMemory:
    """Fixed-size buffer to store priority, Experience tuples."""

    def __init__(self,
                 batch_size: int,
                 buffer_size: int,
                 alpha: float = 0.0,
                 random_state: np.random.RandomState = None) -> None:
        """
        Initialize an ExperienceReplayBuffer object.

        Parameters:
        -----------
        buffer_size (int): maximum size of buffer
        batch_size (int): size of each training batch
        alpha (float): Strength of prioritized sampling. Default to 0.0 (i.e., uniform sampling).
        random_state (np.random.RandomState): random number generator.
        
        """
        self._batch_size = batch_size
        self._buffer_size = buffer_size
        self._buffer_length = 0 # current number of prioritized experience tuples in buffer
        self._buffer = np.empty(self._buffer_size, dtype=[("priority", np.float32), ("transition", Transition)])
        self._alpha = alpha
        self._random_state = np.random.RandomState() if random_state is None else random_state
        
    def __len__(self) -> int:
        """Current number of prioritized experience tuple stored in buffer."""
        return self._buffer_length

    @property
    def alpha(self):
        """Strength of prioritized sampling."""
        return self._alpha

    @property
    def batch_size(self) -> int:
        """Number of experience samples per training batch."""
        return self._batch_size
    
    @property
    def buffer_size(self) -> int:
        """Maximum number of prioritized experience tuples stored in buffer."""
        return self._buffer_size

    def add(self, transition: Transition) -> None:
        """Add a new experience to memory."""
        priority = 1.0 if self.is_empty() else self._buffer["priority"].max()
        if self.is_full():
            if priority > self._buffer["priority"].min():
                idx = self._buffer["priority"].argmin()
                self._buffer[idx] = (priority, transition)
            else:
                pass # low priority experiences should not be included in buffer
        else:
            self._buffer[self._buffer_length] = (priority, transition)
            self._buffer_length += 1

    def is_empty(self) -> bool:
        """True if the buffer is empty; False otherwise."""
        return self._buffer_length == 0
    
    def is_full(self) -> bool:
        """True if the buffer is full; False otherwise."""
        return self._buffer_length == self._buffer_size
    
    def sample(self, beta: float) -> typing.Tuple[np.array, np.array, np.array]:
        """Sample a batch of experiences from memory."""
        # use sampling scheme to determine which experiences to use for learning
        ps = self._buffer[:self._buffer_length]["priority"]
        sampling_probs = ps**self._alpha / np.sum(ps**self._alpha)
        idxs = self._random_state.choice(np.arange(ps.size),
                                         size=self._batch_size,
                                         replace=True,
                                         p=sampling_probs)
        
        # select the experiences and compute sampling weights
        transitions = self._buffer["transition"][idxs]        
        weights = (self._buffer_length * sampling_probs[idxs])**-beta
        normalized_weights = weights / weights.max()
        
        return idxs, transitions, normalized_weights

    def update_priorities(self, idxs: np.array, priorities: np.array) -> None:
        """Update the priorities associated with particular experiences."""
        self._buffer["priority"][idxs] = priorities
        
#dqn model clas 
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.lin1 = nn.Linear(1,2)
        self.lin2 = nn.Linear(2,2)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return x
    
#get action given a policy network, state, and epsilon probability
def get_action(net, state, epsilon):
    with torch.no_grad():
        greedy = np.random.choice([True, False], p=[1-epsilon, epsilon])
        if greedy:
            state = torch.tensor([state], dtype=torch.float32)
            q_values = net(state)
            action = torch.argmax(q_values, dim=0)
        else:
            action = random.choice([0,1])
        return action

#agent class with hyper parameters
class Agent():
    
    def __init__(self, episodes=1000, trainsteps=4, updatesteps=10000, batchsize=64):
        
        #hyperparameters
        self.exp_replay_size = 100000
        self.gamma = 0.99
        self.epsilon = 0.2
        self.target_update_steps = updatesteps
        self.num_episodes = episodes
        self.batch_size = batchsize
        self.train_step_count = trainsteps
        self.steps = 0
        self.lr = 0.001
        self.eps_decay = 5e-7
        self.loss_func = nn.HuberLoss()
        self.alpha = 0.2
        self.episode_count = 0
        
        #networks
        self.QNet = DQN()
        self.TNet = DQN()
        self.optimizer = torch.optim.Adam(self.QNet.parameters(), lr=self.lr)
        
        #replay buffer
        self.ER = PrioritizedReplayMemory(
            batch_size = self.batch_size,
            buffer_size = self.exp_replay_size,
            alpha = self.alpha
        )
        
        
#optimization step function
def optimize(agent):
        
    idxs, sample_transitions, sampling_weights = agent.ER.sample(beta=1-np.exp(-agent.lr*agent.episode_count))
            
    #get batch information
    state_batch = [transition.state for transition in sample_transitions]
    action_batch = [transition.state for transition in sample_transitions]
    reward_batch = [transition.reward for transition in sample_transitions]
    state_new = [transition.state_new for transition in sample_transitions]
    term_batch = [transition.term for transition in sample_transitions]

    state_tensor = torch.tensor(state_batch, dtype=torch.float32, requires_grad=True)
    state_tensor = state_tensor.reshape(agent.batch_size, -1)
    
    policy_preds = agent.QNet(state_tensor)
    policy_values = torch.stack([qvalues[idx] for qvalues, idx in zip(policy_preds, map(int, action_batch))])
    
    state_new_tensor = torch.tensor(state_batch, dtype=torch.float32, requires_grad=True)
    state_new_tensor = torch.nan_to_num(state_new_tensor, nan=0.0)
    state_new_tensor = state_new_tensor.reshape(agent.batch_size, -1)
    target_values = agent.TNet(state_new_tensor)
    target_values = torch.max(target_values, dim=1).values
    
    for idx, is_term in enumerate(term_batch):
        target_values[idx] = 0.0 if is_term else target_values[idx]
        
    target_values = agent.gamma * target_values + torch.tensor(reward_batch, dtype=torch.float32, requires_grad=True)
    
    deltas = torch.subtract(policy_values, target_values)
    
    priorities = (deltas.abs()
                            .cpu()
                            .detach()
                            .numpy()
                            .flatten())
    
    agent.ER.update_priorities(idxs, priorities + 1e-5)
    
    _sampling_weights = (torch.Tensor(sampling_weights).view((-1, 1)))
    
    loss = torch.mean((deltas * _sampling_weights)**2)
    
    return loss

#main learning loop
def main(episodes=1000, trainsteps=4, updatesteps=10000, batchsize=64):
    agent = Agent(episodes=episodes, trainsteps=trainsteps, updatesteps=updatesteps, batchsize=batchsize)
    agent.QNet.train()
    losses = []
    cummulative_rewards = []

    for episode in tqdm(range(agent.num_episodes)):

        environment = Environment()
        cummulative_reward = 0

        while True:

            agent.episode_count+=1

            #observation
            state = environment.get_state()
            action = get_action(agent.QNet, state, agent.epsilon)
            state_new, reward, terminated = environment.take_action(action)
            #append to replay buffer
            agent.ER.add(Transition(state, action, state_new, reward, terminated))

            #update variables
            cummulative_reward+=reward

            #increament step count
            agent.steps+=1

            #train after every 'train_step_count' steps
            if agent.steps%agent.train_step_count==0 and agent.ER.__len__()>agent.batch_size:
                agent.optimizer.zero_grad()
                loss = optimize(agent)
                loss.backward()
                agent.optimizer.step()
                agent.epsilon=max(0, agent.epsilon-agent.eps_decay)
                losses.append(loss.item())

            #copy weights to target network after every 'target_update_steps' updates
            if agent.steps%agent.target_update_steps==0:
                print("updating target network...")
                agent.TNet.load_state_dict(agent.QNet.state_dict())

            #break when episode is complete
            if terminated:
                break

        cummulative_rewards.append(cummulative_reward)
        
    torch.save(agent.QNet.state_dict(), 'models/'+datetime.now().strftime("%d-%m-%Y_%H:%M:%S")+'.pt')
    
    return losses
        
#parse arguments
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000, help="number of episodes to learn on")
    parser.add_argument("--trainsteps", type=int, default=4, help="step interval between experience replay")
    parser.add_argument("--updatesteps", type=int, default=10000, help="step interval between updating target network")
    parser.add_argument("--batchsize", type=int, default=64, help="number of experiences to sample from replay memory")
    args = parser.parse_args()
    loss = main(
        episodes=args.episodes,
        trainsteps=args.trainsteps,
        updatesteps=args.updatesteps,
        batchsize=args.batchsize
    )
    
    plt.plot((np.convolve(loss, np.ones(1000), 'valid') / 1000), label='hubor loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.savefig('charts/'+datetime.now().strftime("%d-%m-%Y_%H:%M:%S")+'.png')

