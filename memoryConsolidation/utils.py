import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from pydantic import BaseModel, conlist
from typing import List, Tuple, Union
from config import dims_dict

DEFAULT_KEY_DICT = {'shapes3d': 'label_shape'}


# Define Pydantic models
class OutputPaths(BaseModel):
    pdf_path: str
    history_path: str
    decoding_path: str

class PreprocessedData(BaseModel):
    data: np.ndarray

class NoisyData(BaseModel):
    noisy_data: np.ndarray

class DatasetSplit(BaseModel):
    train_data: np.ndarray
    test_data: np.ndarray
    train_labels: Union[np.ndarray, None] = None
    test_labels: Union[np.ndarray, None] = None

class PreparedData(BaseModel):
    train_data: np.ndarray
    test_data: np.ndarray
    noisy_train_data: np.ndarray
    noisy_test_data: np.ndarray
    train_labels: Union[np.ndarray, None] = None
    test_labels: Union[np.ndarray, None] = None

def get_output_paths(dataset: str, num: int, generative_epochs: int, latent_dim: int, lr: float, kl_weighting: float) -> OutputPaths:
    base_format = "{}_{}items_{}eps_{}lv_{}lr_{}kl"
    base = base_format.format(dataset, num, generative_epochs, latent_dim, lr, kl_weighting)

    pdf_path = "./outputs/output_" + base + ".pdf"
    history_path = "./outputs/history_" + base + ".pkl"
    decoding_path = "./outputs/decoding_" + base + ".pkl"

    return OutputPaths(pdf_path=pdf_path, history_path=history_path, decoding_path=decoding_path)

def preprocess(array: np.ndarray) -> PreprocessedData:
    # Normalizes the supplied array and reshapes it into the appropriate format.
    array = array.astype("float64") / 255.0
    return PreprocessedData(data=array)

def noise(array: np.ndarray, noise_factor: float = 0.4, seed: Union[int, None] = None, gaussian: bool = False, replacement_val: int = 0) -> NoisyData:
    # Replace a fraction noise_factor of pixels with replacement_val or gaussian noise
    if seed is not None:
        np.random.seed(seed)
    shape = array.shape
    array = array.flatten()
    indices = np.random.choice(np.arange(array.size), replace=False, size=int(array.size * noise_factor))
    if gaussian is True:
        array[indices] = np.random.normal(loc=0.5, scale=1.0, size=array[indices].shape)
    else:
        array[indices] = replacement_val
    array = array.reshape(shape)
    return NoisyData(noisy_data=np.clip(array, 0.0, 1.0))

def display(array1: np.ndarray, array2: np.ndarray, seed: Union[int, None] = None, title: str = 'Inputs and outputs of the model', n: int = 10) -> plt.Figure:
    hopfield = False

    dim = array1[0].shape[0]
    # Displays ten random images from each one of the supplied arrays.
    if seed is not None:
        np.random.seed(seed)

    indices = np.random.randint(len(array1), size=n)
    images1 = array1[indices, :]
    images2 = array2[indices, :]

    fig = plt.figure(figsize=(20, 4))
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        ax = plt.subplot(2, n, i + 1)
        if hopfield is True:
            plt.imshow(image1.reshape(dim, dim), cmap='binary', vmin=-1, vmax=1)
        else:
            plt.imshow(image1.reshape(dim, dim, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        if hopfield is True:
            plt.imshow(image2.reshape(dim, dim), cmap='binary', vmin=-1, vmax=1)
        else:
            plt.imshow(image2.reshape(dim, dim, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig.suptitle(title)
    plt.show()
    return fig

def load_dataset(dataset: str, num: int = 15000, labels: bool = False, key_dict: dict = DEFAULT_KEY_DICT) -> DatasetSplit:
    dim = dims_dict[dataset]

    transform = transforms.Compose([
        transforms.Resize((dim[0], dim[1])),
        transforms.ToTensor()
    ])

    if dataset == "mnist":
        ds = datasets.MNIST(root='./data/', train=True, download=True, transform=transform)
    elif dataset == "cifar10":
        ds = datasets.CIFAR10(root='./data/', train=True, download=True, transform=transform)
    else:
        raise ValueError("Dataset not supported in this implementation. Please add custom dataset loading.")

    data = [np.array(img.permute(1, 2, 0)) for img, _ in ds]
    labels_arr = [label for _, label in ds]

    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels_arr, test_size=0.1, random_state=42)

    train_data = np.array(train_data).reshape(len(train_data), dim[0], dim[1], 3)
    test_data = np.array(test_data).reshape(len(test_data), dim[0], dim[1], 3)

    if labels:
        return DatasetSplit(train_data=train_data, test_data=test_data, train_labels=np.array(train_labels), test_labels=np.array(test_labels))
    else:
        return DatasetSplit(train_data=train_data, test_data=test_data)

def prepare_data(dataset: str, display: bool = False, noise_factor: float = 0.6, labels: bool = False) -> PreparedData:
    if labels:
        split = load_dataset(dataset, labels=True)
        train_data, test_data, train_labels, test_labels = split.train_data, split.test_data, split.train_labels, split.test_labels
    else:
        split = load_dataset(dataset, labels=False)
        train_data, test_data = split.train_data, split.test_data

    # Normalize and reshape the data
    train_data = preprocess(train_data).data
    test_data = preprocess(test_data).data

    # Create a copy of the data with added noise
    noisy_train_data = noise(train_data, noise_factor=noise_factor).noisy_data
    noisy_test_data = noise(test_data, noise_factor=noise_factor).noisy_data

    # Display the train data and a version of it with added noise
    if display:
        display(train_data, noisy_train_data)

    if labels:
        return PreparedData(train_data=train_data, test_data=test_data, noisy_train_data=noisy_train_data, noisy_test_data=noisy_test_data, train_labels=train_labels, test_labels=test_labels)
    else:
        return PreparedData(train_data=train_data, test_data=test_data, noisy_train_data=noisy_train_data, noisy_test_data=noisy_test_data)
