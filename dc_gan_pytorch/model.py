import torch
import torch.nn as nn
import torchvision
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import trange
import matplotlib.pyplot as plt

from discriminator import Discriminator_block
from generator import Generator_block

def initialize_weight(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data,0.0,0.02)

def test():
    N, in_channel, H, W = 8,3,64,64
    z_dim =100
    x = torch.randn((N, in_channel, H, W))
    disc = Discriminator_block(in_channel, 8)
    initialize_weight(disc)
    assert disc(x).shape == (N, 1, 1, 1)
    gen = Generator_block(z_dim, in_channel, 8)
    z = torch.randn((N, z_dim, 1, 1))
    initialize_weight(gen)
    assert gen(z).shape == (N, in_channel, H, W)
    print("Working")

transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
])

batch_size=128

dataset = datasets.ImageFolder(root='../simpsons/',transform=transform)
dataset_loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size, shuffle=True,)

def image_show():
    for _,(a,_) in enumerate(dataset_loader):
        plt.imshow(a[0][0])
        plt.show()
        break

max_epoch = 30
device = torch.device('cuda' if torch.cuda.is_availavle() else 'cpu')
LEARNING_RATE = 2e-3
Z = 100



G = Generator_block(Z, NOISE_DIM, CHANNELS_IMG, FEATURE_GEN).to(device)
D = Discriminator_block(CHANNELS_IMG, FEATURES_DISC).to(device)

initialize_weight(G)
initialize_weight(D)

optim_gen = optim.ADAM(G.parameters, lr = LEARNING_RATE, betas=(.5, 0.999))
optim_dis = optim.ADAM(D.parameters, lr = LEARNING_RATE, betas=(.5, 0.999))
loss_function = nn.BCELoss()



G.train()
D.train()

for epoch in range(max_epoch):
    for _,(images,_) in enumerate(dataset_loader):
        x = images.to(device)
        x_output = D(x)
