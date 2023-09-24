import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import pytorch_lightning as pl

def get_layer(layer_type, in_channel, out_channel,
              dropout_value=0.10):
    if layer_type == 'C':
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channel,
                      out_channels=out_channel,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(dropout_value)
        )
    elif layer_type == 'c':
        return nn.Conv2d(in_channels=in_channel,
                        out_channels=out_channel,
                        kernel_size=(1, 1), padding=0, bias=False)
    elif layer_type == 'P':
        return nn.MaxPool2d(2, 2)
    elif layer_type == 'G':
        return nn.AdaptiveAvgPool2d(output_size=1)
    elif layer_type == "U":
        return nn.Upsample(scale_factor=2, mode='nearest')
    else:
        raise ValueError("wrong `layer_type`")

def build_encoder(
        schema=list('CCPcCCPc'),
        channels=[8, 8, 8, 4, 8, 8, 8, 4,], 
        dropout_value=0.10):
    layers = []
    for layer_type, channel_in, channel_out in zip(
        schema, [1, *channels], channels):
        layers.append(
            get_layer(
                layer_type,
                channel_in,
                channel_out,
                dropout_value)
        )
    return nn.Sequential(*layers)

def build_decoder(
        schema=list('CCUcCCUc'),
        channels=[8, 8, 8, 4, 8, 8, 8, 1], 
        dropout_value=0.10):
    """
    input channels be 4
    output channels be 1
    """
    layers = []
    for layer_type, channel_in, channel_out in zip(
        schema, [4, *channels], channels):
        layers.append(
            get_layer(
                layer_type,
                channel_in,
                channel_out,
                dropout_value)
        )
    return nn.Sequential(*layers)

def build_classifier(
        schema=list('CCCPcCCCPcCCCPcCCcGc'),
        channels=[8, 8, 16, 16, 4, 16, 8, 16, 16, 4, 16, 8, 16, 16,
                 4, 16, 8, 8, 8, 10], 
        dropout_value=0.10):
    layers = []
    for layer_type, channel_in, channel_out in zip(
        schema, [1, *channels], channels):
        layers.append(
            get_layer(
                layer_type,
                channel_in,
                channel_out,
                dropout_value)
        )
    return nn.Sequential(*layers)


class VAE(pl.LightningModule):
    def __init__(
            self,
            x_dim, h_dim1, z_dim, 
            n_classes=10,
            label_emb = 20
        ):
        super(VAE, self).__init__()
        self.save_hyperparameters()
        self.x_dim = x_dim
        # encoder part
        self.encoder_ = build_encoder()
        self.fc1 = nn.Linear(h_dim1, z_dim)
        self.fc2 = nn.Linear(h_dim1, z_dim)
        # decoder part
        self.decoder_ = build_decoder()
        self.classifier = build_classifier()
        self.label_encoder = nn.Embedding(
            n_classes, label_emb)
        self.label_mixer = nn.Linear(z_dim + label_emb, z_dim)
        
    def encoder(self, x):
        h = self.encoder_(x).view(-1, 4 * 49)
        return self.fc1(h), self.fc2(h) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z, label):
        z = torch.concat((z, self.label_encoder(label)),
                         dim=1)
        z = self.label_mixer(z)
        z = z.view(-1, 4, 7, 7)
        reconst = self.decoder_(z)
        return F.sigmoid(reconst) 
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), 1e-3)

    def forward(self, x, label):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        reconstr = self.decoder(z, label)
        pred_prob = F.log_softmax(
            self.classifier(reconstr),
            dim=1).view(-1, 10)
        return  reconstr, pred_prob, mu, log_var

    @staticmethod
    def loss_function(recon_x, x, y, mu, log_var, pred_prob):
        BCE = F.binary_cross_entropy(
            recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        pred_loss = F.nll_loss(pred_prob, y)
        return {"BCE": BCE / 100,
                "KLD": KLD / 100,
                "clf": pred_loss }

    def training_step(self, batch):
        x, y = batch
        if random.random() < 0.1:
            y = torch.randint(0, 9, y.shape).to(torch.long).to(self.device)
        recon_batch, pred_prob, mu, log_var = self(x, y)
        loss = self.loss_function(
            recon_batch, x, y, mu, log_var, pred_prob)
        self.log_dict(loss, prog_bar=True)
        return sum(list(loss.values()))

    
    def validation_step(self, batch, batch_idx):
        if batch_idx == 1:
            with torch.no_grad():
                batch_size = 25
                sample, y = batch[0][:1], batch[1][:1]
                print('actual_digit:', y)
                test_batch = sample[0].expand(batch_size, 1, 28, 28).to(self.device), \
                    torch.randint(0, 9, [batch_size, ]).to(self.device)
                reconstr, pred_prob, mu, log_var = self(*test_batch)
                
                save_image(
                    reconstr.view(batch_size, 1, 28, 28),
                    f'./samples/sample_{self.current_epoch}' + '.png')
        x, y = batch
        y = torch.randint(0, 9, y.shape).to(torch.long).to(self.device)
        recon_batch, pred_prob, mu, log_var = self(x, y)
        loss = self.loss_function(
            recon_batch, x, y, mu, log_var, pred_prob)
        self.log_dict(loss, prog_bar=True)
        return sum(list(loss.values()))
