import pytorch_lightning as pl

from model import VAE
from dataset import get_mnist_dataset

z_dim = 49 * 4

if __name__ == "__main__":
    batch_size = 64
    train_loader, test_loader = get_mnist_dataset()
    # vae = VAE.load_from_checkpoint(
    #     './lightning_logs/version_47/checkpoints/epoch=9-step=9380.ckpt')
    vae = VAE(x_dim=784, h_dim1=196, z_dim=z_dim)
    trainer = pl.Trainer(
        max_epochs=20,
    )
    trainer.fit(
        model=vae,
        train_dataloaders=train_loader,
        val_dataloaders=test_loader
    )
    # with torch.no_grad():
    #     batch_size = 25
    #     sample = next(iter(train_loader))[:1]
    #     print('actual_digit:', sample[1])
    #     test_batch = sample[0][0].expand(batch_size, 1, 28, 28), \
    #         torch.randint(0, 9, [batch_size, ])
    #     print(test_batch[1])
    #     reconstr, pred_prob, mu, log_var = vae(*test_batch)
    #     
    #     save_image(
    #         reconstr.view(25, 1, 28, 28),
    #         './samples/sample_' + '.png')
