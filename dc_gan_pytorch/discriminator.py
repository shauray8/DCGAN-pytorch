# importing important stuff

import torch
import torch.nn as nn
import numpy

# defining the discriminator

class Discriminator_block(nn.Module):
    def __init__(self, input_size, features):
        super(Discriminator_block, self).__init__()
        self.conv1 = nn.Sequential(
            # 64*64
                
            nn.Conv2d(input_size, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(.2),
            # 32*32
            
            self.batchnorm(features, features*2, 4, 2, 1),
            # 16*16
            
            self.batchnorm(featuresi*2, features*4, 4, 2, 1),
            # 8*8
            
            self.batchnorm(features*4, features*8, 4, 2, 1),
            # 4*4

            nn.Conv2d(features*8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid(),
        )

    # Layers with batch normalization 
    def batchnorm(self, input, out, kernel_size, stride, padding):
        return nn.Sequential(
                    nn.Conv2d(input,out,kernel_size,stride,padding,bias=False),
                    nn.BatchNorm2d(out),
                    nn.LeakyReLU(.2),
                )

    # forward function cause we have to have one 
    def forward(self, x):
        x = self.conv1(x)
        return x

# i dont know why !
if __name__ == "__main__":
    print("Discriminator is here!!")

