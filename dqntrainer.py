from torch import optim
from dqnagent import *

class Trainer(object):

    def __init__(self, env, frame_skip=4, capacity=10000, n_episodes=5000, batch_size=300):
        self.memory = ReplayMemory(capacity)
        self.net = QNetwork(env.action_space.n)
        self.optimizer = optim.RMSProp(self.net.parameters())
        self.loss = nn.SmoothL1Loss()
        self.env = env
        self.frame_skip = frame_skip
        self.policy = AgentPolicy(env)
        self.n_episodes = n_episodes
        self.batch_size = batch_size

    def new_episode(self):
        self.env.reset()
        frames_to_stack = [] # There will be 4 frames in this list as preprocessing function phi expects to stack 4 frames
        for _ in range(4):
            action = self.policy.choose_action(self.net)
            for _ in range(self.frame_skip):
                observation, reward, _, _ = self.env.step(action)
            frames_to_stack.append(observation) # Only the final frame is going to be stacked as the rest are skipped
        return frames_to_stack

    def optimize(self, sample):
        q_values = 
        y_values =
        loss = self.loss(q_values, y_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        for _ in range(self.n_episodes):
            self.memory.clear()
            previous_state = self.new_episode()
            current_state = None
            is_done = False
            t = 0
            while not is_done:
                reward_sum = 0 # Rewards will be summed over the skipped frames
                action = self.policy.choose_action(self.net, current_state)
                for _ in range(self.frame_skip):
                    observation, reward, done, _ = self.env.step(action)
                    reward_sum += reward
                if not done:
                    current_state = previous_state.copy()
                    del current_state[0]
                    current_state.append(observation) # Add final frame after skipping to the frames to stack
                    current_state = phi(current_state)
                else:
                    current_state = None
                previous_state = phi(previous_state)
                experience = (previous_state, action, np.clip(reward_sum, -1, 1), current_state)
                self.memory.update(experience)
                t += 1
                if t > self.batch_size: # Check that the replay memory is full enough to take a sample
                    sample = self.memory.sample(self.batch_size)
                    self.optimize(sample)
                is_done = done
