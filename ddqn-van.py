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


#tell user if GPU is available
print("GPU available : {}".format(torch.cuda.is_available()))


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
        return np.array([self.cycle, self.episode[self.cycle]])
    
    #take action
    def take_action(self, action):
        if action == 0:
            #failure occurs
            if self.cycle+1 == self.episode.size:
                res = (None, -78, True)
            #continued operation, return 1 and continue episode
            else:
                res = (np.array([self.cycle, self.episode[self.cycle+1]]), self.episode[self.cycle], False)
            #move to next state
            self.cycle+=1
        elif action == 1:
            res = (None, -self.episode[self.cycle], True)
        return res
    

#transition class for replay memory
class Transition():
    
    def __init__(self, state, action, state_new, reward, term ):
        self.state = state
        self.action = action
        self.state_new = state_new
        self.reward = reward
        self.term = term

        
#vanilla replay memory
class ReplayMemory():

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    
#vanilla double dqn model class
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.lin1 = nn.Linear(2,50)
        self.lin2 = nn.Linear(50,2)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return x
    
    
#get action given model, state, and probability of random choice
def get_action(net, state, epsilon):
    with torch.no_grad():
        greedy = np.random.choice([True, False], p=[1-epsilon, epsilon])
        if greedy:
            state = torch.tensor(state, dtype=torch.float32)
            q_values = net(state)
            action = torch.argmax(q_values, dim=0)
        else:
            action = random.choice([0,1])
        return action
    
    
#agent class to store hyperparamters, nets, and replay memory
class Agent():
    
    def __init__(self, episodes=5000, trainsteps=4, updatesteps=10000, batchsize=64):
        
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
        self.eps_decay = 1e-8
        self.loss_func = nn.MSELoss()
        self.optimizer_steps = 0
        self.optimizer_events = []
        
        #networks
        self.QNet = DQN()
        self.TNet = DQN()
        self.optimizer = torch.optim.Adam(self.QNet.parameters(), lr=self.lr)
        
        #replay buffer
        self.ER = ReplayMemory(self.exp_replay_size)
        
        
#optimizer function
def optimize(agent):
    
    agent.optimizer_steps+=1
    sample_transitions = agent.ER.sample(agent.batch_size)
            
    #get batch information
    state_batch = [transition.state for transition in sample_transitions]
    action_batch = [transition.action for transition in sample_transitions]
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
    
    return agent.loss_func(policy_values, target_values)


def main(episodes, trainsteps, updatesteps, batchsize):
    
    #declare agent
    agent = Agent(episodes, trainsteps, updatesteps, batchsize)

    #set QNet to train mode
    agent.QNet.train()

    #good metrics to track
    losses, cummulative_rewards, epsilons = [], [], []

    #begin training loop
    for episode in tqdm(range(agent.num_episodes)):

        #reinitialize an environment
        environment = Environment()
        cummulative_reward = 0

        while True:

            #observation
            state = environment.get_state()
            action = get_action(agent.QNet, state, agent.epsilon)
            state_new, reward, terminated = environment.take_action(action)

            #append to replay buffer
            agent.ER.push(state, action, state_new, reward, terminated)

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
                epsilons.append(agent.epsilon)
                losses.append(loss.item())

            #copy weights to target network after every 'target_update_steps' updates
            if agent.steps%agent.target_update_steps==0:
                agent.optimizer_events.append(agent.optimizer_steps)
                agent.TNet.load_state_dict(agent.QNet.state_dict())

            #break when episode is complete
            if terminated:
                break

        #track episodes cummulative reward
        cummulative_rewards.append(cummulative_reward)


    #save relevant information for presentation and paper
    np.save('DDQN-VAN/losses.npy', np.array(losses))
    np.save('DDQN-VAN/rewards.npy', np.array(cummulative_rewards))
    np.save('DDQN-VAN/epsilons.npy', np.array(epsilons))
    torch.save(agent.QNet.state_dict(), 'DDQN-VAN/model.pt')

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
    plt.savefig('DDQN-VAN/report.png')

    #plot suggested replacement times for training trajectories
    df = pd.read_csv('train.csv')

    #get 25 random trajectories from the train set
    rng = default_rng()
    numbers = rng.choice(79, size=25, replace=False)
    engine_ids=[num+1 for num in sorted(numbers)]

    #set to eval model
    agent.QNet.eval()

    #create subplot figure
    fig, axs = plt.subplots(5, 5, constrained_layout=True)
    fig.suptitle('Suggested Replacement Cycles (Train)', fontsize=20)
    fig.set_figheight(8)
    fig.set_figwidth(16)

    #iterate over sampled trajectories
    for row in range(5):
        for col in range(5):

            replacement_events = []
            engine_id = int(engine_ids[row+col])
            trajectory = df[df.engine_id==engine_id].health_indicator.to_numpy()

            #for each state, make prediction
            for idx, health in enumerate(trajectory):
                action = torch.argmax(agent.QNet(torch.tensor([idx, health], dtype=torch.float32)))
                if action == 1:
                    replacement_events.append(idx)

            #plot predictions and first suggested replacement time
            axs[row,col].plot(trajectory)
            if replacement_events:
                suggested_replacement = replacement_events[0]
                axs[row,col].axvline(suggested_replacement, c='red')
            else:
                pass
            axs[row,col].title.set_text('Engine {}'.format(engine_id))

    #show figure
    plt.savefig('DDQN-VAN/train_replacements.png')

    #plot suggested replacement times for test trajectories
    df = pd.read_csv('test.csv')

    #get all test engine_ids
    engine_ids=list(range(81,101))

    #set model to eval model
    agent.QNet.eval()

    #declare subplot figure
    fig, axs = plt.subplots(5, 4, constrained_layout=True)
    fig.suptitle('Suggested Replacement Cycles (Test)', fontsize=20)
    fig.set_figheight(8)
    fig.set_figwidth(16)

    #counter to keep track of engine_id being plotted
    engine_counter = 0

    #iterate over all tracjectories
    for row in range(5):
        for col in range(4):

            replacement_events = []
            engine_id = engine_ids[engine_counter]
            engine_counter+=1
            trajectory = df[df.engine_id==engine_id].health_indicator.to_numpy()

            #iterate over all states and make predictions
            for idx, health in enumerate(trajectory):
                action = torch.argmax(agent.QNet(torch.tensor([idx, health], dtype=torch.float32)))
                if action == 1:
                    replacement_events.append(idx)

            #plot health indicator and first replacement event
            axs[row,col].plot(trajectory)
            if replacement_events:
                suggested_replacement = replacement_events[0]
                axs[row,col].axvline(suggested_replacement, c='red')
            else:
                pass
            axs[row,col].title.set_text('Engine {}'.format(engine_id))

    #show figure
    plt.savefig('DDQN-VAN/test_replacements.png')
    

if __name__=='__main__':
    #parse arguments
    args = parser.parse_args()
    main(episodes=args.episodes, trainsteps=args.trainsteps, updatesteps=args.updatesteps, batchsize=args.batchsize)