import pytorch_lightning as pl

from model import VAE
from dataset import get_cifar_dataset

z_dim = 49 * 4
class_names = [
    "Airplane", 
    "Automobile", 
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
]

if __name__ == "__main__":
    batch_size = 32
    train_loader, test_loader = get_cifar_dataset()
    vae = VAE(x_dim=32, x_channel=3, reconst_lambda=1e-5,
              clf_lambda=200000,
              p_using_true_label=0.95, class_names=class_names)

    trainer = pl.Trainer(
        max_epochs=20,
        fast_dev_run=False,
        num_sanity_val_steps=0
    )
    trainer.fit(
        model=vae,
        train_dataloaders=train_loader,
        val_dataloaders=test_loader
    )
