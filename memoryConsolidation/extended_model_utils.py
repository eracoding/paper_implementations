import hashlib
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image, ImageDraw
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import torch
from pydantic import BaseModel, ConfigDict
from typing import List, Tuple

from utils import noise


# Pydantic Models
class RecalledImagesAndLatents(BaseModel):
    predictions: np.ndarray
    latents: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    model_config = ConfigDict(arbitrary_types_allowed=True)  # Allow arbitrary types


class ClassifierModel(BaseModel):
    clf: SVC

    model_config = ConfigDict(arbitrary_types_allowed=True)  # Allow arbitrary types


class PredictionsAndLabels(BaseModel):
    predictions: np.ndarray
    labels: List[int]

    model_config = ConfigDict(arbitrary_types_allowed=True)  # Allow arbitrary types


class DiffResult(BaseModel):
    diff: np.ndarray

    model_config = ConfigDict(arbitrary_types_allowed=True)  # Allow arbitrary types

def get_recalled_ims_and_latents(vae, test_data, noise_level=0) -> RecalledImagesAndLatents:
    test_data_noisy = noise(test_data, noise_factor=noise_level).noisy_data
    test_data_tensor = torch.tensor(test_data_noisy).float().to(vae.device)
    with torch.no_grad():
        latents = vae.encoder(test_data_tensor)
        predictions = vae.decoder(latents[2]).cpu().numpy()
    return RecalledImagesAndLatents(predictions=predictions, latents=latents)


def latent_variable_to_label(latents, labels) -> ClassifierModel:
    clf = make_pipeline(StandardScaler(), SVC())
    latents_np = [latents[2][i].cpu().numpy() for i in range(len(latents[2]))]
    clf.fit(latents_np, labels)
    return ClassifierModel(clf=clf)


def deterministic_seed(image: np.ndarray) -> int:
    hash_object = hashlib.sha1(image.tobytes())
    return int(hash_object.hexdigest(), 16) % (10 ** 8)


def add_white_square(d: np.ndarray, dims: Tuple[int, int, int], seed=0) -> np.ndarray:
    random.seed(deterministic_seed(d))
    square_size = int(dims[0] / 8)
    im1 = Image.fromarray((d * 255).astype("uint8"))
    im2 = Image.fromarray((np.ones((square_size, square_size, 3)) * 255).astype("uint8"))
    Image.Image.paste(im1, im2, (random.randrange(0, dims[0]), random.randrange(0, dims[0])))
    return (np.array(im1) / 255).reshape((dims[0], dims[0], 3))


def blend_images(src: np.ndarray, dst: np.ndarray, alpha: float) -> np.ndarray:
    return (src * alpha) + (dst * (1 - alpha))


def add_multiple_white_squares(d: np.ndarray, dims: Tuple[int, int, int], n: int, seed=4321) -> np.ndarray:
    random.seed(seed)
    square_size = int(dims[0] / 8)
    im1 = (d * 255).astype("uint8")

    for _ in range(n):
        transparency = random.randint(0, 255) / 255
        im2 = np.ones((square_size, square_size, 3)) * 255

        x = random.randrange(0, dims[0] - square_size)
        y = random.randrange(0, dims[0] - square_size)

        im1[y:y + square_size, x:x + square_size] = blend_images(
            im2, im1[y:y + square_size, x:x + square_size], transparency
        ).astype("uint8")

    return (im1 / 255).reshape((dims[0], dims[0], 3))


def get_predictions_and_labels(input_data: np.ndarray, vae, clf: SVC) -> PredictionsAndLabels:
    input_tensor = torch.tensor(input_data).float().to(vae.device)
    with torch.no_grad():
        latents = vae.encoder(input_tensor)
        predictions = vae.decoder(latents[2]).cpu().numpy()
    latents_np = [latents[2][i].cpu().numpy() for i in range(len(latents[2]))]
    labels = clf.predict(latents_np)
    return PredictionsAndLabels(predictions=predictions, labels=labels)


def get_true_pred_diff(input_data: np.ndarray, predictions: np.ndarray) -> DiffResult:
    diff = input_data - predictions
    return DiffResult(diff=diff)


def display_with_labels(array1: np.ndarray, array1_labels: np.ndarray, array2: np.ndarray, array2_labels: np.ndarray, 
                        seed=None, title='Inputs and outputs of the model', random_seed=0, n=10, n_labels=10) -> plt.Figure:
    dim = array1[0].shape[0]
    if seed is not None:
        np.random.seed(seed)

    np.random.seed(random_seed)
    indices = range(0, n)
    images1 = array1[indices, :]
    images2 = array2[indices, :]
    labels1 = array1_labels[indices, :]
    labels2 = array2_labels[indices, :]

    fig = plt.figure(figsize=(20, 4))
    for i, (image1, image2, label1, label2) in enumerate(zip(images1, images2, labels1, labels2)):
        ax = plt.subplot(4, n, i + 1)
        plt.imshow((image1 + 1) / 2)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(4, n, i + 1 + n)
        plt.imshow(label1.reshape(1, n_labels), cmap='binary', vmin=-1, vmax=1)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(4, n, i + 1 + 2 * n)
        plt.imshow((image2 + 1) / 2)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(4, n, i + 1 + 3 * n)
        plt.imshow(label2.reshape(1, n_labels), cmap='binary', vmin=-1, vmax=1)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig.suptitle(title)
    plt.show()
    return fig
