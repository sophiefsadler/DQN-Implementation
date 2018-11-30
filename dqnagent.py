import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

def preprocessor(image_array):
    # This function expects an observation directly from an Atari game in the gym (a numpy array representing a frame)
    image = Image.fromarray(image_array)
    # The transformations performed here are exactly those performed on Atari frames in the original DQN paper
    image_transform = transforms.Compose([transforms.Grayscale(), transforms.Resize((110, 84)), transforms.CenterCrop(84), 
                                            transforms.ToTensor()])
    transformed_image = image_transform(image)
    return transformed_image

def phi(image_list):
    # image_list is expected to be a list of observations directly from an Atari game in the gym (numpy arrays representing frames)
    processed_list = []
    for image in image_list:
        processed_image = preprocessor(image)
        processed_image = processed_image.view(84, 84)
        processed_list.append(processed_image)
    phi_stacked = torch.stack(processed_list)
    phi_stacked = phi_stacked.unsqueeze(0) # Add batch dimension
    return phi_stacked # Output is 4 84x84 frames stacked; this is the desired input for the Q Network

class ReplayMemory(object):

    def __init__(self, capacity, agent):
        self.capacity = capacity

class QNetwork(nn.Module):

    def __init__(self, n_actions):
        super(QNetwork, self).__init__()
        # Architecture taken directly from the DQN paper
        # 4 input channels as we have 4 stacked grey-scale images (each has 1 channel)
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        # Image size = 16 x 20 x 20
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        # Image size = 32 x 9 x 9
        self.fc1 = nn.Linear(32*9*9, 256)
        self.fc2 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view((x.shape[0], -1))
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class DQNAgent():

    def __init__(self):
        pass

    def 


