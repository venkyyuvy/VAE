"""
right and wrong labels in the same batch

"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from classifier_model import ResNet18

from plot_utils import plot_resconstr

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
        # return nn.ConvTranspose2d(in_channel, out_channel, 
        #                           4, stride=2,
        #                           padding=1)
    else:
        raise ValueError("wrong `layer_type`")

def build_encoder(
        schema=list('CCPcCCPc'),
        channels=[1, 8, 8, 8, 4, 8, 8, 8, 4], 
        dropout_value=0.10):
    """
    len(schema) == len(channels) + 1 
    """
    layers = []
    for layer_type, channel_in, channel_out in zip(
        schema, channels, channels[1:]):
        layers.append(
            get_layer(
                layer_type,
                channel_in,
                channel_out,
                dropout_value)
        )
    return nn.Sequential(*layers)

def build_decoder(
        schema=list('CUcCCUc'),
        channels=[8, 8, 8, 4, 8, 8, 8, 1], 
        dropout_value=0.10):
    """
    input channels be 4
    output channels be 1
    """
    return nn.Sequential(
        *[ 
            get_layer(
                layer_type,
                channel_in,
                channel_out,
                dropout_value)
            for layer_type, channel_in, channel_out in zip(
                schema, channels, channels[1:])
        ]
    )

def build_classifier(
        in_channel=1,
        schema=list('CCCPcCCCPcCCCPcCCcGc'),
        channels=[8, 8, 16, 16, 4, 16, 8, 16, 16, 4, 16, 8, 16, 16,
                 4, 16, 8, 8, 8, 10], 
        dropout_value=0.10):
    layers = []
    for layer_type, channel_in, channel_out in zip(
        schema, [in_channel, *channels], channels):
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
        dataset,
        x_dim, 
        class_names,
        x_channel: int=1,
        z_step_down: int=4, 
        z_n_channel: int=4,
        n_classes: int=10,
        label_emb: int=20,
        kld_lambda: float=1e-2,
        reconst_lambda: float=2e-3,
        clf_lambda: float=1e-4,
        p_using_true_label: float=0.1
        ):
        """
        Encoder: 
        reduces x_dim to h_dim1
        projects to mu and sigma (x_dim / z_stepdown each)

        label encoding with label_emb

        Decoder:
        concat sample~N(mu, sigma) and label_emb
        projects to z_dim (z_n_channel * z_step_down ** 2)
        expands back to x_dim
        """
        super(VAE, self).__init__()
        self.save_hyperparameters()
        self.dataset = dataset
        self.x_dim = x_dim
        self.class_names = class_names
        self.kld_lambda = kld_lambda
        self.reconst_lambda = reconst_lambda
        self.clf_lambda = clf_lambda
        self.p_using_true_label = p_using_true_label
        self.x_channel = x_channel
        self.z_channel_size = x_dim // z_step_down
        self.z_n_channel = z_n_channel
        self.n_classes = n_classes
        self.z_dim = z_n_channel * self.z_channel_size ** 2
        # encoder part
        self.encoder_ = build_encoder(
            schema=list('CCCPcCCCPcCCc'),
            channels=[3, 32, 128, 128, 128, 32,
                      32, 128, 256, 256, 
                      32, 256, 256,
                      4], 
        )
        self.fc1 = nn.Linear(self.z_dim, self.z_dim)
        self.fc2 = nn.Linear(self.z_dim, self.z_dim)
        # decoder part
        self.decoder_ = build_decoder(
            schema=list('CCCUcCCCUcCCc'),
            channels=[4, 64, 128, 128, 128, 32, 
                      128, 256, 256, 256, 
                      32, 256, 256,
                      3], 
        )
        if self.dataset == "cifar":
            self.classifier = ResNet18()
            # load the pretrained model
            self.classifier.load_state_dict(
                torch.load("./resnet18.pth")
            )
            for param in self.classifier.parameters():
                param.requires_grad = False
        else:
            self.classifier = build_classifier(
                in_channel=3
            )
        self.label_encoder = nn.Embedding(
            n_classes, label_emb)
        self.label_mixer = nn.Linear(
            self.z_dim + label_emb, self.z_dim)
        
    def encoder(self, x):
        h = self.encoder_(x).view(-1, self.z_dim)
        return self.fc1(h), self.fc2(h) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z, label):
        z = torch.concat(
            (z, self.label_encoder(label)),
            dim=1
        )
        z = self.label_mixer(z)
        z = z.view(
            -1, 
            self.z_n_channel,
            self.z_channel_size, 
            self.z_channel_size
        )
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
            dim=1).view(-1, self.n_classes)
        return  reconstr, pred_prob, mu, log_var

    def loss_function(self, recon_x, x, y, mu, log_var, pred_prob):
        if self.x_channel == 1:
            RECONST = F.binary_cross_entropy(
                recon_x, x, reduction='sum')
        else:
            RECONST = F.mse_loss(recon_x, x)

        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        pred_loss = F.nll_loss(pred_prob, y)
        return {"RECONST": RECONST * self.reconst_lambda,
                "KLD": KLD * self.kld_lambda,
                "clf": pred_loss * self.clf_lambda }

    def training_step(self, batch):
        x, y = batch
        y_new = []
        for label in y:
            if random.random() < self.p_using_true_label:
                y_new.append(label)
            else:
                y_new.append(random.randint(0, self.n_classes - 1))
            # y = self.get_random_labels(y.shape)
        y = torch.tensor(y_new).to(torch.long).to(self.device)
        recon_batch, pred_prob, mu, log_var = self(x, y)
        loss = self.loss_function(
            recon_batch.view(-1, self.x_dim ** 2),
            x.view(-1, self.x_dim ** 2),
            y, mu, log_var, pred_prob)
        self.log_dict(loss, prog_bar=True)
        return sum(list(loss.values()))

    def get_random_labels(self, batch_size):
        return torch.randint(
            0, self.n_classes - 1, batch_size,
        ).to(torch.long).to(self.device)

    def validation_step(self, batch, batch_idx):
        if batch_idx == 1:
            plot_resconstr(self, batch)
        x, y = batch
        y = self.get_random_labels(y.shape)
        recon_batch, pred_prob, mu, log_var = self(x, y)
        loss = self.loss_function(
            recon_batch, x, y, mu, log_var, pred_prob)
        self.log_dict(loss, prog_bar=True)
        return sum(list(loss.values()))
