from dqnagent import *

class Trainer(object):

    def __init__(self, replay_memory, q_network, env, frame_skip):
        self.memory = replay_memory
        self.q_values = q_network
        self.env = env
        self.frame_skip = frame_skip

    def new_episode(self):
        self.env.reset()
        frames_to_stack = [] # There will be 4 frames in this list as preprocessing function phi expects to stack 4 frames
        for _ in range(4):
            action = self.policy.choose_action()
            for _ in range(self.frame_skip):
                observation, reward, _, _ = env.step(action)
            frames_to_stack.append(observation) # Only the final frame is going to be stacked as the rest are skipped
        initial_state = phi

    def train(self, n_episodes):
        for _ in range(n_episodes):
            self.env.reset()
            action = self.env.action_space.sample()
            frames_to_stack = [] # This will have a constant length of 4 as preprocessing function phi expects to stack 4 frames
            rewards_to_sum = [] # This will have a constant length of the frame skip as rewards will be summed over skipped frames
            for _ in range(self.frame_skip):
                observation, reward, _, _ = env.step(action)
                frames_to_stack.append(observation)
                rewards_to_sum.append(reward)

        return