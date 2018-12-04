import gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

def preprocessor(image_array, device):
    # This function expects an observation directly from an Atari game in the gym (a numpy array representing a frame)
    image = Image.fromarray(image_array)
    # The transformations performed here are exactly those performed on Atari frames in the original DQN paper
    image_transform = transforms.Compose([transforms.Grayscale(), transforms.Resize((110, 84)), transforms.CenterCrop(84), 
                                            transforms.ToTensor()])
    transformed_image = image_transform(image).to(device)
    return transformed_image

def phi(image_list, device):
    # image_list is expected to be a list of observations directly from an Atari game in the gym (numpy arrays representing frames)
    processed_list = []
    for image in image_list:
        processed_image = preprocessor(image, device)
        processed_image = processed_image.view(84, 84)
        processed_list.append(processed_image)
    phi_stacked = torch.stack(processed_list)
    phi_stacked = phi_stacked.unsqueeze(0) # Add batch dimension
    return phi_stacked # Output is 4 84x84 frames stacked; this is the desired input for the Q Network

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.current_experience = 0 # This tracks which position in the memory we are updating with a new experience     

    def update(self, new_experience):
        if self.current_experience >= len(self.memory):
            self.memory.append(None)
        self.memory[self.current_experience] = new_experience
        self.current_experience = (self.current_experience + 1) % self.capacity

    def sample(self, batch_size):
        # Used to obtain the random samples for training
        return random.sample(self.memory, batch_size)

    def clear(self):
        self.memory = []

class QNetwork(nn.Module):

    def __init__(self, n_actions):
        super(QNetwork, self).__init__()
        # Architecture taken directly from the DQN paper
        # 4 input channels as we have 4 stacked grey-scale images (each has 1 channel)
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        # Image size = 16 x 20 x 20
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        # Image size = 32 x 9 x 9
        self.fc = nn.Linear(32*9*9, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view((x.shape[0], -1))
        return self.fc(x) # Ouput has shape [1, n_actions]; the extra dimension is currently removed in the agent policy

class AgentPolicy(object):

    def __init__(self, env):
        self.env = env

    def choose_action(self, net, epsilon, observation=None):
        try:
            q_values = net(observation).squeeze(0) # Output from Q-Network has dimension [1, n_actions]; remove the extra dimension
            action = self.epsilon_greedy(epsilon, q_values)
        except TypeError:
            action = self.env.action_space.sample()
        return action

    def epsilon_greedy(self, epsilon, q_values): # DQN is trained with epsilon greedy policy
        value = random.random()
        if value < epsilon:
            action_value = random.randint(0, len(q_values) - 1)
            return action_value
        else:
            _, action_value = q_values.max(0)
            return action_value.item()
