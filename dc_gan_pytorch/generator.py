# importing important stuff

import torch 
import torch.nn as nn

# generator for the GAN

class Generator_block(nn.Module):
    def __init__(self, noise,channels,features):
        super(Generator_block, self).__init__()
        
        self.conv2 = nn.Sequential(
                    self.batchnorm(noise, features*16, 4, 1, 0), #4*4
                    self.batchnorm(features*16, features*8, 4, 2, 1), #8*8
                    self.batchnorm(features*8, features*4, 4, 2, 1), #16*16
                    self.batchnorm(features*4, features*2, 4, 2,1), #32*32
                    nn.ConvTranspose2d(features*2, channels, kernel_size=4, stride=2, padding=1),
                    nn.Tanh(),

                )

    # function for batch nomralization
    def batchnorm(self, input, output, kernel_size, stride, padding):
        return nn.Sequential(
                    nn.ConvTranspose2d(input, output, kernel_size, stride, padding, bias=False),
                    nn.BatchNorm2d(output),
                    nn.ReLU(),
                )
    
    # forward function 
    def forward(self, x):
        x = self.conv2(x)
        return x

if __name__ == "__main__":
    print("Generator is here !!")
