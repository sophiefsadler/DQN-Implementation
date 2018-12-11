import torch
from torch import optim
from dqnagent import *
from board_log import Tensorboard

class Trainer(object):

    def __init__(self, env, frame_skip=4, capacity=1000, target_update_freq=500, n_episodes=5000, batch_size=30, gamma=0.99, gpu=False):
        if gpu:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.memory = ReplayMemory(capacity)
        self.net = QNetwork(env.action_space.n).to(self.device)
        self.target = QNetwork(env.action_space.n).to(self.device)
        self.target.load_state_dict(self.net.state_dict())
        self.optimizer = optim.RMSprop(self.net.parameters())
        self.loss = nn.SmoothL1Loss()
        self.env = env
        self.target_update_freq = target_update_freq
        self.frame_skip = frame_skip
        self.policy = AgentPolicy(env)
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = 1
        self.global_iter = 0
        self.tensorboard = Tensorboard('Train_log')

    def update_target(self):
        self.target.load_state_dict(self.net.state_dict())

    def new_episode(self):
        self.env.reset()
        frames_to_stack = [] # There will be 4 frames in this list as preprocessing function phi expects to stack 4 frames
        for _ in range(4):
            action = self.policy.choose_action(self.net, self.epsilon)
            for _ in range(self.frame_skip):
                observation, _, _, _ = self.env.step(action)
            frames_to_stack.append(observation) # Only the final frame is going to be stacked as the rest are skipped
        return frames_to_stack

    def y_value(self, reward, new_state, done):
        if done:
            y = reward # Occurs if new_state is a terminal state
        else:
            y = reward + self.gamma*self.target(new_state).squeeze(0).max().item()
        return torch.Tensor([y], device=self.device)

    def optimize(self, sample):
        states = torch.stack([exp[0] for exp in sample]).to(self.device)
        new_states = torch.stack([exp[3] for exp in sample]).to(self.device)
        # These now have shape [batch_size, *] where * is the state dimensions
        actions = [exp[1] for exp in sample]
        rewards = [exp[2] for exp in sample]
        dones = [exp[4] for exp in sample]
        # These are lists of length batch_size
        q_values = torch.stack([self.net(states[i]).squeeze(0)[actions[i]] for i in range(self.batch_size)]).to(self.device)
        # q_values is now a tensor of shape [batch_size]
        y_values = torch.stack([self.y_value(rewards[i], new_states[i], dones[i]) for i in range(self.batch_size)]).squeeze(1).to(self.device)
        # y_values is now a tensor of shape [batch_size]
        loss = self.loss(q_values, y_values)
        self.tensorboard.log('Loss', loss, self.global_iter)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        for ep_number in range(self.n_episodes):
            print('----------NEW EPISODE----------')
            ep_iter = 0
            ep_reward = 0
            frames_to_stack = self.new_episode()
            previous_state = phi(frames_to_stack, self.device)
            current_state = None
            is_done = False
            while not is_done:
                reward_sum = 0 # Rewards will be summed over the skipped frames
                action = self.policy.choose_action(self.net, self.epsilon, current_state)
                for _ in range(self.frame_skip):
                    observation, reward, done, _ = self.env.step(action)
                    reward_sum += reward
                    if done:
                        break
                del frames_to_stack[0]
                frames_to_stack.append(observation) # Add final frame after skipping to the frames to stack
                current_state = phi(frames_to_stack, self.device)
                experience = (previous_state, action, np.clip(reward_sum, -1, 1), current_state, done)
                previous_state = current_state
                self.memory.update(experience)
                ep_iter += 1
                self.global_iter += 1
                if ep_iter % 100 == 0:
                    print('ITERATION NUMBER ', ep_iter)
                ep_reward += np.clip(reward_sum, -1, 1)
                if self.global_iter > self.batch_size: # Check that the replay memory is full enough to take a sample 
                    sample = self.memory.sample(self.batch_size)
                    self.optimize(sample)
                if self.global_iter % self.target_update_freq == 0:
                    self.update_target()
                is_done = done
            self.tensorboard.log('Episode Length', ep_iter, ep_number)
            self.tensorboard.log('Episode Reward', ep_reward, ep_number)
            self.tensorboard.log('Epsilon Value', self.epsilon, ep_number)
            if self.epsilon > 0.1:
                self.epsilon = -(0.9/1000000)*self.global_iter + 1
