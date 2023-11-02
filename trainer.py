import pytorch_lightning as pl

from model import VAE
from dataset import get_mnist_dataset

z_dim = 49 * 4

# encoder
# schema=list('CCPcCCPc'),
# channels=[8, 8, 8, 4, 8, 8, 8, 4,], 

# decoder
# schema=list('CCUcCCUc'),
# channels=[8, 8, 8, 4, 8, 8, 8], 

if __name__ == "__main__":
    batch_size = 32
    train_loader, test_loader = get_mnist_dataset()
    # vae = VAE.load_from_checkpoint(
    #     './lightning_logs/version_47/checkpoints/epoch=9-step=9380.ckpt')
    vae = VAE(dataset="mnist", x_dim=28, 
              class_names= map(str, range(1, 10))
              )

    trainer = pl.Trainer(
        max_epochs=20,
    )
    trainer.fit(
        model=vae,
        train_dataloaders=train_loader,
        val_dataloaders=test_loader
    )
