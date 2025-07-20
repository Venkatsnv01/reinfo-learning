import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Qneuralnet(nn.Module):
    def __init__(self, dimension_action, dimension_states):
        super(Qneuralnet, self).__init__()
        
        self.fc1 = nn.Linear(dimension_states, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, dimension_action)
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
    def sample_action(self, observation, epsilon):
        a = self.forward(observation)

        if random.random() < epsilon: # explore
            return random.randint(0, 3) # explore
        else:
            return a.argmax().item() # exploit