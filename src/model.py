import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class OutlineGenerationModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(OutlineGenerationModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        

    def forward(self, inputs):
        outputs, _ = self.lstm(inputs)
        outputs = self.linear(outputs)
        return outputs
class StoryExpansionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(StoryExpansionModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, inputs):
        outputs, _ = self.lstm(inputs)
        outputs = self.linear(outputs)
        return outputs

input_dim =  640
hidden_dim =  300
output_dim = 640
num_layers = 2
learning_rate = 0.001
num_epochs = 10


outline_generation_model = OutlineGenerationModel(input_dim, hidden_dim, output_dim, num_layers)
story_expansion_model = StoryExpansionModel(input_dim, hidden_dim, output_dim, num_layers)


criterion = nn.CrossEntropyLoss()
optimizer_outline = optim.Adam(outline_generation_model.parameters(), lr=learning_rate)
optimizer_story = optim.Adam(story_expansion_model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    outline_generation_model.train()
    for inputs, targets in train_loader_outline:
        optimizer_outline.zero_grad()
        outputs = outline_generation_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer_outline.step()


    outline_generation_model.eval()
    with torch.no_grad():
        for inputs, targets in val_loader_outline:



for epoch in range(num_epochs):
    story_expansion_model.train()
    for inputs, targets in train_loader_story:
        optimizer_story.zero_grad()
        outputs = story_expansion_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer_story.step()


    story_expansion_model.eval()
    with torch.no_grad():
        for inputs, targets in val_loader_story:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = story_expansion_model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            val_total += targets.size(0)
            val_correct += (outputs.argmax(1) == targets).sum().item()
    print('Epoch: {}, Train Loss: {}, Val Loss: {}, Val Acc: {}'.format(epoch, train_loss, val_loss, val_correct/val_total))
    



outline_generation_model.eval()
story_expansion_model.eval()
with torch.no_grad():
    for inputs, targets in test_loader:

