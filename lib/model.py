import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class RNN_v4(nn.Module):

    def __init__(self, num_layers, feature_size, embedding_size, hidden_size, taget_size):
        super(RNN_v4, self).__init__()
        self.num_layers = num_layers
        self.feature_size = feature_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.taget_size = taget_size

        self.fc1 = nn.Linear(feature_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, embedding_size)
        
        self.rnn = nn.GRU(embedding_size, hidden_size, num_layers)
        self.fc_out = nn.Linear(hidden_size, taget_size)

    def forward(self, inputs):
        output = F.relu(self.fc1(inputs))
        output = F.relu(self.fc2(output))

        output, _ = self.rnn(output)
        output = self.fc_out(output) # map hidden dim to output dim
        # output[:,:,0:4] = torch.cumsum(output[:,:,0:4], dim=0) # cdf sum of probability
        output[:,:,0:4] = torch.sigmoid(output[:,:,0:4])

        return output
    
    
    