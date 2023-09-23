# prerequisites
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
import pytorch_lightning as pl

from model import VAE

bs = 100
# MNIST Dataset
train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=False)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)


# build model
vae = VAE(x_dim=784, h_dim1= 512, h_dim2=256, z_dim=2)

if __name__ == "__main__":
    trainer = pl.Trainer(
        max_epochs=20,
    )
    trainer.fit(
        model=vae,
        train_dataloaders=train_loader,
    )
    with torch.no_grad():
        z = torch.randn(64, 2)
        sample = vae.decoder(z)
        
        save_image(
            sample.view(64, 1, 28, 28),
            './samples/sample_' + '.png')
