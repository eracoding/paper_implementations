import matplotlib.backends.backend_pdf
import numpy as np
import os
import pickle
import torch
from torch.optim import Adam
from random import randrange
from tqdm import tqdm
from pydantic import BaseModel, ConfigDict
from typing import Union

import hopfield_utils
from hopfield_models import ContinuousHopfield, DenseHopfield, ClassicalHopfield
from config import dims_dict
from generative_model import *
from generative_tests import interpolate_ims, check_generative_recall, plot_history, vector_arithmetic, \
    plot_latent_space_with_labels
from utils import prepare_data, display, get_output_paths


# Pydantic Models
class EndToEndConfig(BaseModel):
    dataset: str = 'shapes3d'
    generative_epochs: int = 10
    num: int = 100
    latent_dim: int = 5
    kl_weighting: float = 1.0
    hopfield_type: str = 'continuous'
    lr: float = 0.001
    do_vector_arithmetic: bool = False
    interpolate: bool = False
    plot_space: bool = True
    hopfield_beta: int = 100
    use_weights: bool = True

    model_config = ConfigDict(arbitrary_types_allowed=True)  # Allow arbitrary types


class EndToEndResult(BaseModel):
    net: Union[None, ClassicalHopfield, DenseHopfield, ContinuousHopfield]
    vae: VAE

    model_config = ConfigDict(arbitrary_types_allowed=True)  # Allow arbitrary types


def run_end_to_end(config: EndToEndConfig) -> EndToEndResult:
    """
    Runs an end-to-end simulation of consolidation as teacher-student training of a generative network.
    """
    # Get paths to write results
    pdf_path, history_path, decoding_path = get_output_paths(
        config.dataset, config.num, config.generative_epochs, config.latent_dim, config.lr, config.kl_weighting
    )

    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_path)

    print("Preparing datasets")
    train_data, test_data, _, _, train_labels, test_labels = prepare_data(config.dataset, labels=True)
    dims = dims_dict[config.dataset]

    if config.use_weights:
        print("Using existing weights - set use_weights to False to train a new model with the specified parameters")
        encoder, decoder = models_dict[config.dataset](latent_dim=config.latent_dim)
        encoder.load_state_dict(torch.load(f"model_weights/{config.dataset}_encoder.pth"))
        decoder.load_state_dict(torch.load(f"model_weights/{config.dataset}_decoder.pth"))
        vae = VAE(encoder, decoder, kl_weighting=config.kl_weighting)
        vae.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        net = None

    else:
        np_f = f'./mhn_memories/predictions_{config.dataset}_{config.num}.npy'
        if os.path.exists(np_f):
            print("Using saved MHN predictions from previous run.")
            with open(np_f, 'rb') as fh:
                predictions = np.load(fh)
            net = None

        else:
            print("Creating Hopfield network.")
            net = hopfield_utils.create_hopfield(
                config.num, hopfield_type=config.hopfield_type, dataset=config.dataset, beta=config.hopfield_beta
            ).network
            predictions = []
            tests = []

            images_masked_np = hopfield_utils.mask_image_random(config.num).random_arrays
            images_masked_np = [im_np.reshape(-1, 1) for im_np in images_masked_np]

            print("Sampling from modern Hopfield network.")
            for test_ind in tqdm(range(config.num)):
                test = images_masked_np[test_ind].reshape(-1, 1)
                if config.hopfield_type == 'classical':
                    reconstructed = net.retrieve(test)
                else:
                    reconstructed = net.retrieve(test, max_iter=10)
                predictions.append(reconstructed.reshape((1, dims[0], dims[1], dims[2])))
                tests.append(test.reshape((1, dims[0], dims[1], dims[2])))

            predictions = np.concatenate(predictions, axis=0)
            tests = np.concatenate(tests, axis=0)

            fig = display(tests, predictions, title='Inputs and outputs for modern Hopfield network')
            pdf.savefig(fig)
            predictions = (predictions + 1) / 2

            with open(np_f, 'wb') as fh:
                np.save(fh, predictions)

        print("Starting to train VAE.")
        encoder, decoder = build_encoder_decoder_large(latent_dim=config.latent_dim)
        vae = VAE(encoder, decoder, kl_weighting=config.kl_weighting)
        vae.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # Training settings
        optimizer = Adam(vae.parameters(), lr=config.lr, amsgrad=True)
        decoding_history = DecodingHistory(config.dataset)
        history = {'reconstruction_loss': []}
        early_stopping_patience = 3
        no_improvement_epochs = 0
        best_loss = float('inf')

        print("Training VAE:")
        vae.train()
        predictions_tensor = torch.tensor(predictions).float().to(vae.device)
        for epoch in range(config.generative_epochs):
            optimizer.zero_grad()
            reconstruction, z_mean, z_log_var, _ = vae(predictions_tensor)
            loss = vae.compute_loss(predictions_tensor, reconstruction, z_mean, z_log_var)
            loss.backward()
            optimizer.step()

            # Track loss history
            history['reconstruction_loss'].append(loss.item())
            print(f'Epoch {epoch+1}/{config.generative_epochs}, Loss: {loss.item()}')

            # Early stopping condition
            if loss.item() < best_loss:
                best_loss = loss.item()
                no_improvement_epochs = 0
            else:
                no_improvement_epochs += 1
                if no_improvement_epochs >= early_stopping_patience:
                    print("Early stopping triggered.")
                    break

        # Save model weights
        torch.save(vae.encoder.state_dict(), f"model_weights/{config.dataset}_encoder.pth")
        torch.save(vae.decoder.state_dict(), f"model_weights/{config.dataset}_decoder.pth")

        fig = plot_history(history, decoding_history)
        pdf.savefig(fig, bbox_inches='tight')

        with open(history_path, "wb") as hist_file, open(decoding_path, "wb") as dec_file:
            pickle.dump(history['reconstruction_loss'], hist_file)
            pickle.dump(decoding_history.decoding_history, dec_file)

    print("Recalling noisy images with the generative model:")
    fig = check_generative_recall(vae, train_data[0:100])
    pdf.savefig(fig)

    latents = vae.encoder(torch.tensor(test_data).float().to(vae.device)).detach().cpu().numpy()

    if config.interpolate:
        print("Interpolating between image pairs:")
        for i in range(10):
            fig = interpolate_ims(latents, vae, randrange(50), randrange(50))
            pdf.savefig(fig)

    if config.do_vector_arithmetic:
        print("Doing vector arithmetic:")
        for i in range(10):
            random_class = np.random.choice(range(len(set(train_labels))))
            class_indices = np.where(test_labels == random_class)[0]
            first, third = np.random.choice(class_indices, size=2, replace=False)
            second = randrange(100)
            fig = vector_arithmetic(test_data, latents, vae, first, second, third)
            pdf.savefig(fig)

    if config.plot_space:
        print("Plotting latent space with labels:")
        fig = plot_latent_space_with_labels(latents, test_labels)
        pdf.savefig(fig)

    pdf.close()

    return EndToEndResult(net=net, vae=vae)
