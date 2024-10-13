import torch 
import torch.nn as nn
import torch.nn.utils as utils

class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()
    self.encoder = nn.Sequential(
      utils.parametrizations.weight_norm(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5,1), stride=(3,1), padding = 0)), # Convolutional layer 1
      nn.BatchNorm2d(16),
      nn.ReLU(),
      utils.parametrizations.weight_norm(nn.Conv2d(in_channels=16, out_channels=4, kernel_size=(5,1), stride=(3,1), padding = 0)), # Convolutional layer 1
      nn.BatchNorm2d(4),
      nn.ReLU()
      )

  def forward(self, x):
    return self.encoder(x)

class Encoder2(nn.Module):
  def __init__(self):
    super(Encoder2, self).__init__()
    self.encoder = nn.Sequential(
      utils.parametrizations.weight_norm(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5,1), stride=(3,1), padding = 0)), # Convolutional layer 1
      nn.BatchNorm2d(16),
      nn.ReLU(),
      utils.parametrizations.weight_norm(nn.Conv2d(in_channels=16, out_channels=4, kernel_size=(5,1), stride=(3,1), padding = 0)), # Convolutional layer 1
      nn.BatchNorm2d(4),
      nn.ReLU()
      )

  def forward(self, x):
    encoded_x = self.encoder(x)
    trimmed_to_one_sec_x = encoded_x[:, :, 1:-1, :]
    return trimmed_to_one_sec_x

class Decoder(nn.Module):
  def __init__(self):
    super(Decoder, self).__init__()
    self.decoder = nn.Sequential(
      utils.parametrizations.weight_norm(nn.ConvTranspose2d(in_channels=4, out_channels=16, kernel_size=(6,1), stride=(3,1), padding = 0)), # Convolutional layer 1
      nn.BatchNorm2d(16),
      nn.ReLU(),
      utils.parametrizations.weight_norm(nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(6,1), stride=(3,1), padding = (1,0))), # Convolutional layer 1
      nn.BatchNorm2d(1),
      nn.ReLU()
      )

  def forward(self, x):
    return self.decoder(x)

class Classifier2(nn.Module):
  def __init__(self):
    super(Classifier2, self).__init__()
    self.classifier2 = nn.Sequential(
      utils.parametrizations.weight_norm(nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(100,4), stride=(50,1), padding = 0)), # Convolutional layer 1
      nn.BatchNorm2d(16),
      nn.ReLU(),
      nn.Flatten(),
      nn.Linear(16*3, 1),
      nn.Sigmoid()
      )

  def forward(self, x):
    return self.classifier2(x)

class Localizator(nn.Module):
  def __init__(self):
    super(Localizator, self).__init__()
    self.localizator = nn.Sequential(
      utils.parametrizations.weight_norm(nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(100,4), stride=(50,1), padding = 0)), # Convolutional layer 1
      nn.BatchNorm2d(16),
      nn.ReLU(),
      nn.Flatten(),
      nn.Linear(16*3, 1),
      nn.Sigmoid()
      )

  def forward(self, x):
    return self.localizator(x)