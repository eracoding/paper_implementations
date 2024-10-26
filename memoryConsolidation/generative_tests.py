import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from sklearn.manifold import TSNE
import torch

from utils import display, noise, prepare_data


def check_generative_recall(vae, test_data, noise_level=0.1):
    # Add noise to test data
    test_data_noisy = noise(test_data, noise_factor=noise_level).noisy_data
    test_data_noisy = torch.tensor(test_data_noisy).float().to(vae.device)
    
    with torch.no_grad():
        latents = vae.encoder(test_data_noisy)
        predictions = vae.decoder(latents[2]).cpu().numpy()
    
    fig = display(test_data, predictions, title='Inputs and outputs for VAE')
    return fig


def plot_history(history, decoding_history, titles=False):
    recon_loss_values = history['reconstruction_loss']
    decoding_acc_values = decoding_history.decoding_history
    epochs = range(1, len(recon_loss_values) + 1)

    fig, ax = plt.subplots(figsize=(3, 3))
    ax2 = ax.twinx()
    ax2.set_ylim(0, 1)
    ax.set_ylabel("Reconstruction error")
    ax2.set_ylabel("Decoding accuracy")

    ax.plot(epochs, recon_loss_values, label='Reconstruction Error', color='red')
    ax2.plot(epochs, decoding_acc_values, label='Decoding Accuracy', color='blue')

    if titles:
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=0)
        plt.title('Reconstruction error and decoding accuracy over time')

    ax.set_xlabel('Epoch')
    plt.show()
    return fig


def interpolate_ims(latents, vae, first, second):
    encoded_imgs = latents[2].cpu().numpy()
    enc1 = encoded_imgs[first:first + 1]
    enc2 = encoded_imgs[second:second + 1]

    linfit = interp1d([1, 10], np.vstack([enc1, enc2]), axis=0)

    fig = plt.figure(figsize=(20, 5))

    for j in range(10):
        ax = plt.subplot(1, 10, j + 1)
        with torch.no_grad():
            decoded_imgs = vae.decoder(torch.tensor(linfit(j + 1)).float().to(vae.device)).cpu().numpy()
        ax.imshow(decoded_imgs[0].transpose(1, 2, 0))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig.suptitle('Interpolation between items')
    plt.show()
    return fig


def vector_arithmetic(imgs, latents, vae, first, second, third):
    img1 = imgs[first]
    img2 = imgs[second]
    img3 = imgs[third]

    encoded_imgs = latents[2].cpu().numpy()
    enc1 = encoded_imgs[first:first + 1]
    enc2 = encoded_imgs[second:second + 1]
    enc3 = encoded_imgs[third:third + 1]

    fig, axs = plt.subplots(1, 4, figsize=(10, 2))
    axs[0].imshow(img1)
    axs[0].axis('off')
    axs[1].imshow(img2)
    axs[1].axis('off')
    axs[2].imshow(img3)
    axs[2].axis('off')
    # enc1-enc2=enc3-enc4 -> enc4=enc3+enc2-enc1
    res = -enc1 + enc2 + enc3

    with torch.no_grad():
        result_img = vae.decoder(torch.tensor(res).float().to(vae.device)).cpu().numpy()

    axs[3].imshow(result_img[0].transpose(1, 2, 0))
    axs[3].axis('off')
    fig.suptitle('Vector arithmetic')
    plt.show()
    return fig


def plot_latent_space_with_labels(latents, labels, titles=False):
    np.random.seed(1)
    fig = plt.figure(figsize=(4, 4))

    embedded = TSNE(n_components=2, init='pca').fit_transform(latents[2].cpu().numpy()[0:800])
    x = [x[0] for x in embedded]
    y = [x[1] for x in embedded]

    plt.scatter(x, y, c=labels[0:800], alpha=0.5, cmap=plt.cm.plasma)
    if titles:
        plt.title('Latent space in 2D, colour-coded by label')
    plt.show()
    return fig
