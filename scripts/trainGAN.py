from comet_ml import Experiment
from layers import Discriminator, Generator
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from datasets import LocalizationDataFormat
import random
import math


class SynteticDataGenerator(nn.Module):
    def __init__(self, generator, discriminator, experiment:Experiment, batch_size:int = 8, learning_rate:float = 3e-3, ):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.learning_rate = learning_rate
        self.patience = 0
        self.last_loss = float("inf")
        
        #hyperparameters
        self.batch_size = batch_size
        self.experiment = experiment
        self.number_of_epochs = 0
        
        
        self.criterion = nn.BCELoss()
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.learning_rate)
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.learning_rate)
        
    def get_dset(self, train_share=0.8):
        dset = LocalizationDataFormat(root_dir='/home/mbak/LeakDetection/data/localization/v2_samples126_lenght22_typeLocalisation.npz')
        train_size = int(train_share * len(dset))
        validation_size = len(dset) - 2*train_size//3
        test_size = len(dset) - train_size//3
        return random_split(dataset=dset, lengths=[train_size, validation_size, test_size], generator=torch.Generator().manual_seed(42))  # fix the generator for reproducible results
    
    def train(self, train_dataloader:DataLoader, validation_dataloader:DataLoader,  num_of_epochs:int =10):
        self.number_of_epochs = num_of_epochs
        self.validation_dataloader = validation_dataloader
        if self.experiment: self.log_hyperparameters()
        for epoch in range(self.number_of_epochs):
            for batch, (real_data, real_leak_label, real_localization) in enumerate(train_dataloader):
                real_data = real_data.reshape(real_data.size(0), -1, real_data.size(3))
                real_data = real_data.permute(0, 2, 1)
                real_d_loss = self.real_phase_train(real_data)
                generated_data, fake_d_loss, g_loss = self.fake_phase_train(real_data.size(0))
                
                self.log_losses(epoch, batch, real_d_loss, fake_d_loss, g_loss)     
                self.log_data_as_plot(real_data, epoch, batch, "real_data")        
                self.log_data_as_plot(generated_data, epoch, batch, "generated_data") 
            
            self.phase_validation()
            self.soft_stop()
            
              
                         
    def real_phase_train(self, real_data):
        self.optimizer_D.zero_grad()
        real_labels = torch.ones(real_data.size(0), 1).double()
        
        outputs = self.discriminator(real_data)
        d_loss = self.criterion(outputs, real_labels)
        d_loss.backward()
        self.optimizer_D.step()
        return d_loss.item()
        
    def fake_phase_train(self, batch_size):
        # Train discriminator on fake data
        self.optimizer_D.zero_grad()
        fake_data = self.generator(batch_size).detach()  # Detach to avoid generator gradients
        fake_labels = torch.zeros(fake_data.size(0), 1).double()
        outputs = self.discriminator(fake_data)
        d_loss = self.criterion(outputs, fake_labels)
        d_loss.backward()
        self.optimizer_D.step()
            
        # Train generator
        self.optimizer_G.zero_grad()
        fake_data = self.generator(batch_size)  # Recompute to create a new graph
        outputs = self.discriminator(fake_data)
        g_loss = self.criterion(outputs, torch.ones(fake_data.size(0), 1).double())  # Generator wants to fool the discriminator
        g_loss.backward()
        self.optimizer_G.step()
        
        return fake_data, d_loss.item(), g_loss.item()
    
    def phase_validation(self, batch_size):

        self.optimizer_G.zero_grad()
        fake_data = self.generator(batch_size)  # Recompute to create a new graph
        outputs = self.discriminator(fake_data)
        g_loss = self.criterion(outputs, torch.ones(fake_data.size(0), 1).double())  # Generator wants to fool the discriminator

        
        return fake_data, g_loss.item()
    
        
        
    def soft_stop(self, loss):
        pass
    
    def log_hyperparameters(self):
        self.experiment.log_parameters({
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_epochs": self.number_of_epochs
        })
    
    def log_losses(self, epoch:int, batch:int, real_d_loss:float, fake_d_loss:float, g_loss:float):
        print(f"Epoch/Batch:  {epoch}/{batch}")
        print(f"\treal dicriminator loss:\t{real_d_loss}")
        print(f"\tfake dicriminator loss:\t{fake_d_loss}")
        print("f\tgenerator loss:\t{g_loss}")
        
        if self.experiment:  # Ensure the experiment is not None
            self.experiment.log_metric("real_d_loss", real_d_loss, step=epoch*batch)
            self.experiment.log_metric("fake_d_loss", fake_d_loss, step=epoch*batch)
            self.experiment.log_metric("g_loss", g_loss, step=epoch*batch)
            
    def log_data_as_plot(self, data: torch.Tensor, epoch:int, batch:int, name:str = "data"):
        name = f"{epoch}/{batch}_{name}"
        index = random.randint(0,data.shape[0]-1)
        random_element = data[index]
        plt.figure(figsize=(80, 8))
        for i in range(1):
            plt.plot(random_element[i].detach().numpy(), label=f'manometr{i+1}', linestyle='-')
        self.experiment.log_figure(figure_name= name, figure= plt)
        
        
        

        
        
        
        
        
        

if __name__ == "__main__":
    generator = Generator()
    discriminator = Discriminator()
    
    experiment = Experiment(
        api_key="V6xW30HU42MtnnpSl6bsGODZ1",    # Replace 'your-api-key' with your actual Comet API key
        project_name="LeakDetection"
    )
    
    dataGenerator = SynteticDataGenerator(generator.double(), discriminator.double(), experiment)
    train_dataset, validation_dataset, test_dataset = dataGenerator.get_dset()
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=8, shuffle=True, pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=8, drop_last=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=8, drop_last=True, pin_memory=True)
    

    dataGenerator.train(train_loader, validation_loader, 50)
    
    