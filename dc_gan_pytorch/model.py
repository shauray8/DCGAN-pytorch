import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
 
import pickle
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

def image_show():
    for _,(a,_) in enumerate(dataset_loader):
        plt.imshow(a[0][0])
        plt.show()
        break


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


max_epoch = 30
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 2e-3
NOISE = 100
CHANNELS = 3
FEATURES = 64


G = Generator_block(NOISE, CHANNELS, FEATURES).to(device)
D = Discriminator_block(CHANNELS, FEATURES).to(device)

initialize_weight(G)
initialize_weight(D)

optim_G = optim.Adam(G.parameters, lr = LEARNING_RATE, betas=(.5, 0.999))
optim_D = optim.Adam(D.parameters, lr = LEARNING_RATE, betas=(.5, 0.999))
loss_function = nn.BCELoss()

G.train()
D.train()

D_labels = torch.ones([batch_size,1]).to('cuda')
D_fakes = torch.zeros([batch_size,1]).to('cuda')
D_losses = []
G_losses = []
img_list = []

for epoch in (wal := trange(max_epoch)):
    for id,(images,_) in enumerate(dataset_loader):
        x = images.to(device)
        x_output = D(x)
        D_x_loss = loss_function(x_output, D_labels)

        z = torch.randn(batch_size, NOISE).to(device)
        z_output = D(G(z))
        D_z_loss = loss_function(z_output, D_fakes)
        D_loss = D_x_loss + D_z_loss

        optim_D.zero_grad()
        D_loss.backward()
        optim_D.step()

        z = torch.randn(batch_size, NOISE).to(device)
        z_output = D(G(z))
        G_loss = loss_function(z_output, D_labels)

        optim_G.zero_grad()
        G_loss.backward()
        optim_G.step()

        wal.set_description(f'Step: {id}, D Loss: {D_loss.item()}, G Loss: {G_loss.item()}')
        D_losses.append(D_loss.item())
        G_losses.append(G_loss.item())

        if step % 200 == 0:
            with torch.no_grad():
                fake = G(z).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

with open('images.pkl','wb') as f:
    pickle.dump(img_list, f)

plt.plot(D_losses)
plt.plot(G_losses)
plt.show()

torch.save(Generator_block, './pretrained_models')