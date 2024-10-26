import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from pydantic import BaseModel, ConfigDict
from typing import List, Tuple
from utils import prepare_data


# Define Pydantic model
class ClassificationScore(BaseModel):
    score: float

    model_config = ConfigDict(arbitrary_types_allowed=True)  # Allow arbitrary types


def label_classifier(latents: np.ndarray, labels: np.ndarray, num: int = 200) -> ClassificationScore:
    np.random.seed(1)
    x_train, x_test, y_train, y_test = train_test_split(latents, labels, test_size=0.5, random_state=1)
    clf = make_pipeline(StandardScaler(), SVC())
    clf.fit(x_train[:num], y_train[:num])
    score = clf.score(x_test, y_test)
    return ClassificationScore(score=score)


class DecodingHistory:
    def __init__(self, dataset: str):
        _, self.test_data, _, _, _, self.test_labels = prepare_data(dataset, labels=True)
        self.decoding_history = []

    def on_epoch_begin(self, model, epoch: int):
        with torch.no_grad():
            latents = model.encoder(torch.tensor(self.test_data).float().to(model.device))
            score = label_classifier(latents[0].cpu().numpy(), self.test_labels)
            self.decoding_history.append(score.score)


class Sampling(nn.Module):
    def forward(self, z_mean, z_log_var):
        epsilon = torch.randn_like(z_mean)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon


class EncoderNetworkLarge(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int], latent_dim: int = 100):
        super(EncoderNetworkLarge, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv2d(3, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.mean = nn.Linear(256, latent_dim)
        self.log_var = nn.Linear(256, latent_dim)
        self.sampling = Sampling()

    def forward(self, x):
        x = self.encoder(x).view(x.size(0), -1)
        z_mean = self.mean(x)
        z_log_var = self.log_var(x)
        z = self.sampling(z_mean, z_log_var)
        return z_mean, z_log_var, z


class DecoderNetworkLarge(nn.Module):
    def __init__(self, latent_dim: int = 100):
        super(DecoderNetworkLarge, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 4096),
            nn.ReLU(),
            nn.Unflatten(1, (256, 4, 4)),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.decoder(z)


class VAE(nn.Module):
    def __init__(self, encoder, decoder, kl_weighting: float = 1):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.kl_weighting = kl_weighting

    def forward(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction, z_mean, z_log_var, z

    def compute_loss(self, x, reconstruction, z_mean, z_log_var):
        # Reconstruction Loss
        reconstruction_loss = F.mse_loss(reconstruction, x, reduction='mean')

        # KL Divergence Loss
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - torch.exp(z_log_var))
        kl_loss /= x.size(0)  # Normalize by batch size
        return reconstruction_loss + self.kl_weighting * kl_loss


def build_encoder_decoder_large(latent_dim: int = 5) -> Tuple[EncoderNetworkLarge, DecoderNetworkLarge]:
    input_shape = (64, 64, 3)
    encoder = EncoderNetworkLarge(input_shape, latent_dim)
    decoder = DecoderNetworkLarge(latent_dim)
    return encoder, decoder


models_dict = {"shapes3d": build_encoder_decoder_large}
