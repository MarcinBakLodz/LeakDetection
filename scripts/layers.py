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
  
  ########### GAN #############
  
class Generator(nn.Module):
  "Class to generate 20 secconds data"
  def __init__(self, latent_data_lenght: int = 100, latent_data_channels:int = 1, result_sampling_frequency: int =100, result_lenght_in_sec: int = 22, result_channels: int = 4, debug :bool = False):
    super().__init__()
    self.result_lenght_in_sec: int = result_lenght_in_sec
    self.latent_data_lenght: int = latent_data_lenght
    self.latent_data_channels: int = latent_data_channels
    self.result_sampling_frequency: int = result_sampling_frequency
    self.result_channels: int = result_channels
    self.debug = debug
    
    self.ch1 = 4
    self.ch2 = 12
    self.ch3 = 16
    
    
    self.tc1 = torch.nn.ConvTranspose1d(self.latent_data_channels, self.ch1*self.latent_data_channels, kernel_size=(9), stride=1)
    self.batchNorm1 = torch.nn.BatchNorm1d(self.ch1*self.latent_data_channels)
    self.tc2 = torch.nn.ConvTranspose1d(self.ch1*self.latent_data_channels, self.ch2*self.latent_data_channels, kernel_size=(31), stride=2)
    self.batchNorm2 = torch.nn.BatchNorm1d(self.ch2*self.latent_data_channels)
    self.tc3 = torch.nn.ConvTranspose1d(self.ch2*self.latent_data_channels, self.ch3*self.latent_data_channels, kernel_size=(31), stride=3)
    self.batchNorm3 = torch.nn.BatchNorm1d(self.ch3*self.latent_data_channels)
    self.tc4 = torch.nn.ConvTranspose2d(self.ch3*self.latent_data_channels, self.ch3*self.latent_data_channels, kernel_size=(49,4), stride=(3, 1))
    self.batchNorm4 = torch.nn.BatchNorm2d(num_features=self.ch3*self.latent_data_channels)
    self.c1 = torch.nn.Conv2d(self.ch3*latent_data_channels, result_channels, kernel_size=(121,4), stride=(1,1), padding= 0)
    self.batchNormc1 = torch.nn.BatchNorm2d(result_channels)
    # self.c2 = torch.nn.Conv1d(8*latent_data_channels, result_channels, kernel_size=121, stride=1) #na wszelki wielki
    self.leakyRealu = torch.nn.LeakyReLU()
    self.tahn = torch.nn.Tanh()
    self.dropout02 = torch.nn.Dropout2d(p=0.2)
    self.dropout05 = torch.nn.Dropout2d(p=0.5)
    
  def generate_random_sample_from_gaussian_distribution(self, batch_size:int)-> torch.Tensor:
    for param in self.parameters():
      if param.dtype == torch.float32:
        return torch.normal(mean= 1, std=1, size=(batch_size, self.latent_data_channels, self.latent_data_lenght))
      elif param.dtype == torch.float64:
        return torch.normal(mean= 1, std=1, size=(batch_size, self.latent_data_channels, self.latent_data_lenght)).double()
      else:
        raise ValueError("Incorrect dicriminator noise format")

    
  def forward(self, batch_size:int)->torch.Tensor:
    x0 = self.generate_random_sample_from_gaussian_distribution(batch_size)
    if self.debug: print("x0: ", x0.shape)
    x1 = self.dropout02(self.leakyRealu(self.batchNorm1(self.tc1(x0))))
    if self.debug: print("x1: ", x1.shape)
    x2 = self.dropout05(self.leakyRealu(self.batchNorm2(self.tc2(x1))))
    if self.debug: print("x2: ", x2.shape)
    x3 = self.dropout05(self.leakyRealu(self.batchNorm3(self.tc3(x2))))
    if self.debug: print("x3: ", x3.shape)
    x31 = x3.unsqueeze(-1)
    if self.debug: print("x31: ", x31.shape)
    x4 = self.dropout05(self.leakyRealu(self.batchNorm4(self.tc4(x31))))
    if self.debug: print("x4: ", x4.shape)
    x5 = self.leakyRealu(self.batchNormc1(self.c1(x4)))
    if self.debug: print("x5: ", x5.shape)
    x51 = x5.squeeze(-1)
    if self.debug: print("x51: ", x51.shape)
    # x6 = self.LeakyRealu(self.c2(x51))
    # print("x6: ", x6.shape)
    start_index = (x51.shape[2] - self.result_lenght_in_sec*self.result_sampling_frequency) // 2
    end_index = start_index + self.result_lenght_in_sec*self.result_sampling_frequency
    x52 = x51[:, :, start_index:end_index]
    if self.debug: print("x52: ", x52.shape)
    return x52
  
class Discriminator(nn.Module):
  def __init__(self, input_channels:int = 4, debug:bool = False):
    super().__init__()
    self.input_channels = input_channels
    self.debug = debug
    
    self.ch1 = 8
    self.ch2 = 16
    self.ch3 = 32
    self.ch4 = 32
    
    self.c1 = torch.nn.Conv1d(in_channels=self.input_channels, out_channels=self.ch1, kernel_size=21, stride= 2)
    self.batchNorm1 = torch.nn.BatchNorm1d(self.ch1)
    self.c2 = torch.nn.Conv1d(self.ch1, out_channels=self.ch2, kernel_size=21, stride= 2)
    self.batchNorm2 = torch.nn.BatchNorm1d(self.ch2)
    self.c3 = torch.nn.Conv2d(in_channels=self.ch2, out_channels=self.ch3, kernel_size=(21,1), stride= (2,1))
    self.batchNorm3 = torch.nn.BatchNorm2d(self.ch3)
    self.c4 = torch.nn.Conv2d(in_channels=self.ch3, out_channels=self.ch4, kernel_size=(21,1), stride= (2,1))
    self.batchNorm4 = torch.nn.BatchNorm2d(self.ch4)
    self.l1 = torch.nn.Linear(in_features=self.ch4*119, out_features=1)
    
    
    self.leakyReLU = torch.nn.LeakyReLU()
    self.sigmoid = torch.nn.Sigmoid()
    self.dropout = torch.nn.Dropout(0.2)
    
    
  def forward(self, x:torch.Tensor)->torch.Tensor:
    assert self.check_if_input_has_good_shape(x), "bad input shape, it should be batch_size, 4, 2200"
    x1 = self.dropout(self.leakyReLU(self.batchNorm1(self.c1(x))))
    if self.debug: print("x1: ", x1.shape)
    x2 = self.dropout(self.leakyReLU(self.batchNorm2(self.c2(x1))))
    if self.debug: print("x2: ", x2.shape)
    x21 = x2.unsqueeze(-1)
    if self.debug: print("x21: ", x21.shape)
    x3 = self.dropout(self.leakyReLU(self.batchNorm3(self.c3(x21))))
    if self.debug: print("x3: ", x3.shape)
    x4 = self.dropout(self.leakyReLU(self.batchNorm4(self.c4(x3))))
    if self.debug: print("x4: ", x4.shape)
    x41 = torch.flatten(x4.squeeze(-1), start_dim= 1)
    if self.debug: print("x41: ", x41.shape)
    x5 = self.sigmoid(self.l1(x41))
    if self.debug: print("x5: ", x5.shape)
    return x5
    
  def check_if_input_has_good_shape(self, x:torch.Tensor)->bool:
    return x.shape[1:] == torch.Size([4, 2200])
    
    
    
    
    
    
  
if __name__ == "__main__":
  "place for all tests"
  generator = Generator(latent_data_lenght=100, latent_data_channels= 1)
  generated_data = generator(16)
  
  print("----------------")
  discriminator = Discriminator()
  discriminator(generated_data)