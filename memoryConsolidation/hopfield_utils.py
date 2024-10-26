import torch
import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict
from typing import List, Union
from config import dims_dict
from hopfield_models import ContinuousHopfield, DenseHopfield, ClassicalHopfield
from utils import load_dataset


# Pydantic models
class HopfieldNetworkConfig(BaseModel):
    num: int
    hopfield_type: str = 'continuous'
    dataset: str = 'mnist'
    beta: float = 10.0

    model_config = ConfigDict(arbitrary_types_allowed=True)  # Allow arbitrary types


class HopfieldNetworkResult(BaseModel):
    network: Union[ContinuousHopfield, DenseHopfield, ClassicalHopfield]

    model_config = ConfigDict(arbitrary_types_allowed=True)  # Allow arbitrary types


class ImagesData(BaseModel):
    images: List[Image.Image]

    model_config = ConfigDict(arbitrary_types_allowed=True)  # Allow arbitrary types


class ConvertedImages(BaseModel):
    images_np: List[np.ndarray]

    model_config = ConfigDict(arbitrary_types_allowed=True)  # Allow arbitrary types


class RandomMaskedImages(BaseModel):
    random_arrays: List[np.ndarray]
    
    model_config = ConfigDict(arbitrary_types_allowed=True)  # Allow arbitrary types


def create_hopfield(config: HopfieldNetworkConfig) -> HopfieldNetworkResult:
    images = load_images_dataset(config.num, dataset=config.dataset).images
    images_np = convert_images(images).images_np
    images_np = [im_np.reshape(-1, 1) for im_np in images_np]

    n_pixel = dims_dict[config.dataset][0]
    n_channels = 3
    orig_shape = (n_pixel, n_pixel, n_channels)
    n = np.prod(orig_shape)
    train_patterns = images_np

    if config.hopfield_type == 'continuous':
        net = ContinuousHopfield(n, beta=config.beta)
    elif config.hopfield_type == 'dense':
        net = DenseHopfield(n, beta=config.beta)
    elif config.hopfield_type == 'classical':
        net = ClassicalHopfield(n)
    else:
        raise ValueError(f"Unknown hopfield_type: {config.hopfield_type}")

    net.learn(train_patterns)
    return HopfieldNetworkResult(network=net)


def load_images_dataset(num: int, dataset: str = 'mnist') -> ImagesData:
    train_data, _ = load_dataset(dataset)
    np.random.shuffle(train_data)
    images = []
    for i in range(num):
        im_arr = train_data[i]
        im = Image.fromarray(im_arr)
        images.append(im)
    return ImagesData(images=images)


def convert_images(images: List[Image.Image]) -> ConvertedImages:
    # Converts images with values 0 to 255 to ones with values -1 to 1
    images_np = []
    for im in images:
        im_np = ((np.asarray(im) / 255) * 2) - 1
        images_np.append(im_np)
    return ConvertedImages(images_np=images_np)


def mask_image_random(n: int) -> RandomMaskedImages:
    random_arrays = []
    for i in range(n):
        random_array = np.random.uniform(-1, 1, size=(64, 64, 3))
        random_arrays.append(random_array)
    return RandomMaskedImages(random_arrays=random_arrays)
