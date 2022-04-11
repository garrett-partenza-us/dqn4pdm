#imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import numpy as np
import random
import copy
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import typing
from numpy.random import default_rng
import argparse

#disable warnings
import warnings
warnings.filterwarnings("ignore")


#argument parameters
parser = argparse.ArgumentParser(description='Allow user to pass agent hyperparameters')
parser.add_argument('--episodes', type=int, help='number of epsisodes to train', default=5000)
parser.add_argument('--trainsteps', type=int, help='training step interval', default=4)
parser.add_argument('--updatesteps', type=int, help='target update step interval', default=10000)
parser.add_argument('--batchsize', type=int, help='number of experiences to sample from memory', default=64)
parser.add_argument('--alpha', type=float, help='alpha to use for prioritized experience replay', default=0.5)
parser.add_argument('--fleetsize', type=int, help='number of engines in fleet', default=5)

#tell user if GPU is available
print("GPU available : {}".format(torch.cuda.is_available()))

class Engine():
    
    #initialize dataset, a random trajectory, and current cycle
    def __init__(self, dataset):
        self.dataset = dataset
        self.service = True
        self.episode = self.get_trajectory(np.random.randint(low=1, high=79, size=1))
        self.cycle = np.random.randint(0,len(self.episode),1).item()

    #get random trajectory
    def get_trajectory(self, engine_id):
        return self.dataset[self.dataset.engine_id==engine_id.item()].health_indicator.to_numpy()
    
    #return current state
    def get_state(self):
        return self.cycle, self.episode[self.cycle]
    
    #take action
    def step(self, action):
        if action == 0:
            #failure occurs
            if self.cycle+1 == self.episode.size:
                res = (None, -78)
                self.service = False
            #continued operation, return 1 and continue episode
            else:
                res = (self.episode[self.cycle+1], self.episode[self.cycle])
                self.cycle+=1
        elif action == 1:
            self.cycle=0+int(np.random.uniform(0,50))
            res = (self.episode[self.cycle], -self.episode[self.cycle])
        return res
    
#environment class
class Environment():
    
    #initialize dataset, a random trajectory, and current cycle
    def __init__(self, fleet_size=5):
        self.dataset = pd.read_csv('train.csv')
        self.fleet_size = fleet_size
        self.fleet = []
        for engine in range(fleet_size):
            self.fleet.append(Engine(self.dataset))
        self.balance = 100
        self.repair_cost = 25
        
    def get_state(self):
        healths = []
        cycles = []
        for engine in self.fleet:
            if engine.service:
                cycle, health = engine.get_state()
                healths.append(health)
                cycles.append(cycle)
            else:
                healths.append(-1)
                cycles.append(-1)
        return np.array([self.balance]+cycles+healths).flatten()
        
    def take_action(self, actions):
        #initialize reward as zero
        reward = []
        #iterate over action-engine pair
        for action, engine in zip(actions, self.fleet):
            #hold
            if action == 0:
                if engine.service:
                    _, r = engine.step(action)
                    reward.append(r)
                    # if engine.service:
                    #     self.balance+=0.1
                    #     pass
                else:
                    reward.append(0)
            #replace
            elif action == 1:
                #perform replacement if enough money
                if self.balance >= self.repair_cost:
                    self.balance-=self.repair_cost
                    if engine.service:
                        _, r = engine.step(action)
                        reward.append(r)
                    else:
                        reward.append(-10)
                #penalize if not enough money
                else:
                    reward.append(-10)
                    if engine.service:
                        engine.step(0)
        #return reward and terminal flag
        if np.all([engine.service==False for engine in self.fleet]):
            return reward, True
        else:
            return reward, False
        
#transition class for replay memory
class Transition():
    
    def __init__(self, state, action, state_new, reward, term ):
        self.state = state
        self.action = action
        self.state_new = state_new
        self.reward = reward
        self.term = term
        

#prioritized replay memory class
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
        
    
#dqn model class
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.lin1 = nn.Linear(input_dim,128)
        self.lin2 = nn.Linear(128,256)
        self.lin3 = nn.Linear(256,128)
        self.lin4 = nn.Linear(128,output_dim)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.dropout1(x)
        x = F.relu(self.lin2(x))
        x = self.dropout2(x)
        x = F.relu(self.lin3(x))
        x = F.relu(self.lin4(x))
        return x
    

#get action given net, state, and probability of random action
def get_action(net, state, epsilon, size):
    with torch.no_grad():
        greedy = np.random.choice([True, False], p=[1-epsilon, epsilon])
        if greedy:
            state = torch.tensor(state, dtype=torch.float32)
            q_values = net(state)
            q_values = q_values.reshape(-1,2)
            actions = torch.argmax(q_values, dim=1)
        else:
            actions = np.random.choice([1,0], size=size)
        return actions
    

#agent class
class Agent():
    
    def __init__(self, episodes=5000, trainsteps=4, updatesteps=10000, batchsize=64, alpha=0.5, fleet_size=5):
        
        #hyperparameters
        self.exp_replay_size = 100000
        self.gamma = 0.99
        self.epsilon = 0.01
        self.target_update_steps = updatesteps
        self.num_episodes = episodes
        self.batch_size = batchsize
        self.train_step_count = trainsteps
        self.steps = 0
        self.lr = 0.01
        self.eps_decay = 5e-8
        self.loss_func = nn.MSELoss()
        self.optimizer_steps = 0
        self.optimizer_events = []
        self.alpha = alpha
        self.fleet_size = fleet_size
        self.episode_count=0
        
        #networks
        self.QNet = DQN(fleet_size*2+1, fleet_size*2)
        self.TNet = DQN(fleet_size*2+1, fleet_size*2)
        self.optimizer = torch.optim.Adam(self.QNet.parameters(), lr=self.lr)
        
        #replay buffer
        self.ER = PrioritizedReplayMemory(
            batch_size = self.batch_size,
            buffer_size = self.exp_replay_size,
            alpha = self.alpha
        )
        
        
def selector(a,b):  
    res = torch.tensor([a[row][idx] for row, idx in enumerate(b)], requires_grad=True)
    return res
#optimize function
def optimize(agent):
        
    agent.optimizer_steps+=1
    idxs, sample_transitions, sampling_weights = agent.ER.sample(beta=1-np.exp(-agent.lr*agent.episode_count))
                        
    #get batch information
    state_batch = [transition.state for transition in sample_transitions]
    action_batch = [transition.action for transition in sample_transitions]
    reward_batch = [transition.reward for transition in sample_transitions]
    state_new = [transition.state_new for transition in sample_transitions]
    term_batch = [transition.term for transition in sample_transitions]

    state_tensor = torch.tensor(state_batch, dtype=torch.float32, requires_grad=True)
    state_tensor = state_tensor.reshape(agent.batch_size, -1)
    
    policy_preds = agent.QNet(state_tensor).reshape(agent.batch_size,-1,2)
    policy_values = torch.stack([selector(qvalues, idx) for qvalues, idx in zip(policy_preds, action_batch)])
    
    state_new_tensor = torch.tensor(state_batch, dtype=torch.float32, requires_grad=True)
    state_new_tensor = torch.nan_to_num(state_new_tensor, nan=0.0)
    state_new_tensor = state_new_tensor.reshape(agent.batch_size, -1)
    target_values = agent.TNet(state_new_tensor).reshape(agent.batch_size,-1,2)
    target_values = torch.max(target_values, dim=2).values
    
    for idx, is_term in enumerate(term_batch):
        if is_term:
            for engine in range(agent.fleet_size):
                target_values[idx][engine]=0.0
        
    reward_batch = torch.tensor(reward_batch, dtype=torch.float32, requires_grad=True)
    target_values = agent.gamma * target_values + reward_batch
    
    deltas = torch.sum(torch.abs(torch.subtract(policy_values, target_values)), dim=1)
    
    priorities = (deltas.abs().cpu().detach().numpy().flatten())
                  
    agent.ER.update_priorities(idxs, priorities + 1e-5) #priorities must be positive
    
    _sampling_weights = (torch.Tensor(sampling_weights).view((-1, 1)))
    
    loss = torch.mean((deltas * _sampling_weights)**2)
    
    return loss


#main
def main(episodes, trainsteps, updatesteps, batchsize, alpha, fleet_size):
    
    agent = Agent(episodes, trainsteps, updatesteps, batchsize, alpha, fleet_size)
    agent.QNet.train()
    losses = []
    cummulative_rewards = []
    epsilons = []

    for episode in tqdm(range(agent.num_episodes)):

        environment = Environment(fleet_size)
        cummulative_reward = 0
        moves = 0

        while True:

            agent.episode_count+=1
            moves+=1
            #observation
            state = environment.get_state()
            action = get_action(agent.QNet, state, agent.epsilon, agent.fleet_size)
            reward, terminated = environment.take_action(action)
            state_new = environment.get_state()
            #append to replay buffer
            agent.ER.add(Transition(state, action, state_new, reward, terminated))

            #update variables
            cummulative_reward+=sum(reward)

            #increament step count
            agent.steps+=1

            #train after every 'train_step_count' steps
            if agent.steps%agent.train_step_count==0 and agent.ER.__len__()>agent.batch_size:
                agent.optimizer.zero_grad()
                loss = optimize(agent)
                loss.backward()
                agent.optimizer.step()
                epsilons.append(agent.epsilon)
                agent.epsilon=max(0, agent.epsilon-agent.eps_decay)
                losses.append(loss.item())

            #copy weights to target network after every 'target_update_steps' updates
            if agent.steps%agent.target_update_steps==0:
                agent.optimizer_events.append(agent.optimizer_steps)
                agent.TNet.load_state_dict(agent.QNet.state_dict())

            #break when episode is complete
            if terminated:
                break

        cummulative_rewards.append(cummulative_reward)
        
    #save relevant information for presentation and paper
    np.save('DDQN-PER-MULTI-wRS/losses.npy', np.array(losses))
    np.save('DDQN-PER-MULTI-wRS/rewards.npy', np.array(cummulative_rewards))
    np.save('DDQN-PER-MULTI-wRS/epsilons.npy', np.array(epsilons))
    torch.save(agent.QNet.state_dict(), 'DDQN-PER-MULTI-wRS/model.pt')

    #plot performance report

    #decalre subplot figure
    fig, axs = plt.subplots(3, 1, constrained_layout=True)
    fig.suptitle('Performance Report', fontsize=20)
    fig.set_figheight(12)
    fig.set_figwidth(16)

    #plot losses
    axs[0].plot(
        (np.convolve(losses, np.ones(100), 'valid') / 100),
        label='rolling (100) mse',
        c='blue'
    )

    #plot target update events
    for event in agent.optimizer_events:
        axs[0].axvline(
            event, 
            alpha=0.5,
            c='red'
        )

    #plot cummulative rewards
    axs[1].plot(
        (np.convolve(cummulative_rewards, np.ones(100), 'valid') / 100),
        label='rolling (100) cummulative reward',
        c='green',
    )

    #plot epsilon decay
    axs[2].plot(
        epsilons,
        label='decaying epsilon',
        c='orange'
    )

    #set titles for subplots
    axs[0].title.set_text('Loss and Target Network Updates')
    axs[1].title.set_text('Cummulative Reward')
    axs[2].title.set_text('Decaying Epsilon')

    #turn on legend
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    #show figure
    plt.legend()
    plt.savefig('DDQN-PER-MULTI-wRS/report.png')


if __name__=='__main__':
    args = parser.parse_args()
    main(
        episodes=args.episodes,
        trainsteps=args.trainsteps,
        updatesteps=args.updatesteps,
        batchsize=args.batchsize,
        alpha=args.alpha,
        fleet_size=args.fleetsize
    )
    