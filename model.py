import torch
import numpy as np

class Encoder(torch.nn.Module):
    def __init__(self, flag_size: int, latent_size: int):
        super().__init__()
        self.input_size = flag_size
        self.conv1 = torch.nn.Conv2d(4, 64, 3, 1, padding=1)
        self.maxpool1 = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.maxpool2 = torch.nn.MaxPool2d(2, 2)
        self.conv3 = torch.nn.Conv2d(64, 32, 5, stride=2, padding=2)
        self.maxpool3 = torch.nn.MaxPool2d(2, 2)
        self.conv4 = torch.nn.Conv2d(32, 32, 5, stride=2, padding=2)
        self.maxpool4 = torch.nn.MaxPool2d(2, 2)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.pre_flatten_size = self.pre_flatten(torch.zeros(1, 4, self.input_size, self.input_size)).shape[1:]
        self.linear = torch.nn.Linear(np.prod(self.pre_flatten_size), latent_size)
        
    def pre_flatten(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool3(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxpool4(x)
        return x
        
    def forward(self, x: torch.Tensor):
        x = self.pre_flatten(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
    
class Decoder(torch.nn.Module):
    def __init__(self, latent_size: int, pre_flatten_size: list[int]):
        super().__init__()
        self.input_size = latent_size
        self.pre_flatten_size = pre_flatten_size
        self.linear = torch.nn.Linear(latent_size, np.prod(pre_flatten_size))
        self.unpool1 = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.deconv1 = torch.nn.ConvTranspose2d(32, 32, 5, stride=2, padding=2, output_padding=1)
        self.unpool2 = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.deconv2 = torch.nn.ConvTranspose2d(32, 64, 5, stride=2, padding=2, output_padding=1)
        self.unpool3 = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.deconv3 = torch.nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1)
        self.unpool4 = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.deconv4 = torch.nn.ConvTranspose2d(64, 4, 3, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = x.view(x.size(0), *self.pre_flatten_size)
        x = self.unpool1(x)
        x = self.deconv1(x)
        x = self.relu(x)
        x = self.unpool2(x)
        x = self.deconv2(x)
        x = self.relu(x)
        x = self.unpool3(x)
        x = self.deconv3(x)
        x = self.relu(x)
        x = self.unpool4(x)
        x = self.deconv4(x)
        x = self.sigmoid(x)
        return x