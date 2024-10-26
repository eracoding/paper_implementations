import logging
import matplotlib.gridspec as gridspec
import torch
from sklearn.utils import shuffle
import numpy as np
from pydantic import BaseModel, ConfigDict
from typing import Union

from end_to_end import *
from extended_model_utils import *
from hopfield_models import ContinuousHopfield

plt.rcParams["figure.figsize"] = (15, 3)
torch.manual_seed(1)


# Pydantic Models
class ExtendedModelConfig(BaseModel):
    dataset: str = 'shapes3d'
    vae: Union[None, VAE] = None
    beta: float = 100.0
    generative_epochs: int = 100
    num_ims: int = 1000
    latent_dim: int = 10
    threshold: float = 0.01
    return_errors_and_counts: bool = False
    n_squares: int = 3
    n: int = 100

    model_config = ConfigDict(arbitrary_types_allowed=True)  # Allow arbitrary types


def get_predictions(input_data: np.ndarray, vae: VAE) -> np.ndarray:
    input_tensor = torch.tensor(input_data).float().to(vae.device)
    with torch.no_grad():
        latents = vae.encoder(input_tensor)
        predictions = vae.decoder(latents[2]).cpu().numpy()
    return predictions


def find_min_max(arr: np.ndarray) -> Tuple[float, float]:
    min_val = np.amin(arr)
    max_val = np.amax(arr)
    return min_val, max_val


def ims_to_hpc_format(input_ims: np.ndarray, vae: VAE, threshold: float, latent_dim: int, 
                      dims=(64, 64, 3), fixed_latents=None):
    predictions = get_predictions(input_ims, vae)
    diff = get_true_pred_diff(input_ims, predictions).diff

    avg_diff = np.mean(abs(diff), axis=-1, keepdims=True)
    mask = avg_diff > threshold
    version_to_visualise = np.where(mask, (input_ims * 2) - 1, 0)

    if fixed_latents is None:
        _, latents = get_recalled_ims_and_latents(vae, input_ims, noise_level=0)
        latents = np.array([latents[2][i].cpu().numpy() for i in range(len(latents[2]))])
        latents = latents.reshape((input_ims.shape[0], latent_dim, 1))
    else:
        latents = fixed_latents

    sparse_to_encode = version_to_visualise.reshape((input_ims.shape[0], np.prod(dims), 1))
    hpc_traces = np.concatenate((sparse_to_encode, latents), axis=1)
    return hpc_traces, latents, version_to_visualise


def recall_memories(test_data: np.ndarray, net: ContinuousHopfield, vae: VAE, dims, latent_dim: int, 
                    test_ind=0, noise_factor=0.2, threshold=0.005, return_final_im=False, 
                    fixed_latents=None, n=10):
    noisy_data = noise(test_data[0:n], noise_factor=noise_factor).noisy_data
    hpc_traces, latents, to_visualise = ims_to_hpc_format(noisy_data, vae, threshold, latent_dim, dims=dims)
    test = hpc_traces[test_ind].reshape(-1, 1)
    recalled = net.retrieve(test, max_iter=5)

    if np.isnan(recalled).all():
        logging.warning("NaN values detected; skipping.")
        return None, None

    high_error = recalled[0:np.prod(dims)].reshape((dims[0], dims[1], dims[2]))
    reconstructed_latent = recalled[np.prod(dims):]

    fig = plt.figure()
    gs = gridspec.GridSpec(2, 6, figure=fig, height_ratios=[3, 1], wspace=0.05, hspace=0.05)

    plt.subplot(gs[0])
    plt.imshow(test_data[test_ind])
    plt.axis('off')

    if fixed_latents is not None:
        plt.subplot(gs[6])
        plt.imshow(fixed_latents.reshape(1, latent_dim), cmap='binary')
        plt.axis('off')

    plt.subplot(gs[1])
    plt.imshow(noisy_data[test_ind])
    plt.axis('off')

    plt.subplot(gs[2])
    plt.imshow((to_visualise[test_ind] + 1) / 2)
    plt.axis('off')

    plt.subplot(gs[8])
    plt.imshow(latents[test_ind].reshape(1, latent_dim), cmap='binary')
    plt.axis('off')

    plt.subplot(gs[3])
    plt.imshow((high_error + 1) / 2)
    plt.axis('off')

    plt.subplot(gs[9])
    plt.imshow(reconstructed_latent.reshape(1, latent_dim), cmap='binary')
    plt.axis('off')

    with torch.no_grad():
        vae_pred = vae.decoder(torch.tensor(reconstructed_latent).float().to(vae.device))
        vae_pred = vae_pred.cpu().numpy().reshape((dims[0], dims[1], dims[2]))

    mask = np.logical_not(np.isclose(high_error, 0, atol=1e-2))
    combined = np.where(mask, high_error, (vae_pred * 2) - 1)

    plt.subplot(gs[4])
    plt.imshow(vae_pred, vmin=0, vmax=1)
    plt.axis('off')

    plt.subplot(gs[5])
    plt.imshow((combined + 1) / 2)
    plt.axis('off')

    plt.show()

    if return_final_im:
        return fig, combined
    else:
        return fig, None


def test_extended_model(config: ExtendedModelConfig):
    dims = dims_dict[config.dataset]

    pdf = matplotlib.backends.backend_pdf.PdfPages(f"./hybrid_model/extended_version_{config.dataset}_{config.threshold}.pdf")

    if config.vae is None:
        _, vae = run_end_to_end(
            dataset=config.dataset, generative_epochs=config.generative_epochs,
            num=config.num_ims, latent_dim=config.latent_dim, kl_weighting=1
        )
        logging.info("Trained VAE.")
    else:
        vae = config.vae

    train_data, test_data, _, _, train_labels, test_labels = prepare_data(config.dataset, labels=True)
    test_data, test_labels = shuffle(test_data, test_labels, random_state=0)

    test_data_squares = [add_multiple_white_squares(d, dims, config.n_squares) for d in test_data][0:config.n]
    test_data_squares = np.stack(test_data_squares, axis=0).astype("float32")[0:config.n]

    predictions = get_predictions(test_data_squares, vae)
    diff = get_true_pred_diff(test_data_squares, predictions).diff

    net = ContinuousHopfield(np.prod(dims) + config.latent_dim, beta=config.beta)

    avg_diff = np.mean(abs(diff), axis=-1, keepdims=True)
    mask = avg_diff > config.threshold
    sparse_to_encode = np.where(mask, (test_data_squares * 2) - 1, 0)

    _, latents = get_recalled_ims_and_latents(vae, test_data_squares, noise_level=0)
    pred_labels = np.array([latents[2][i].cpu().numpy() for i in range(len(latents[2]))])
    pred_labels = pred_labels.reshape((config.n, config.latent_dim, 1))

    a = sparse_to_encode.reshape((config.n, np.prod(dims), 1))
    hpc_traces = np.concatenate((a, pred_labels), axis=1)
    net.learn(hpc_traces[0:config.n])

    len_noise_vec = (64 * 64 * 3) + config.latent_dim
    images_masked_np = np.random.uniform(-1, 1, size=(config.n, len_noise_vec, 1))

    tests, label_inputs, predictions, pred_labels = [], [], [], []

    for test_ind in range(10):
        test_in = images_masked_np[test_ind].reshape(-1, 1)
        test_out = net.retrieve(test_in, max_iter=5)
        reconstructed = test_out[0:np.prod(dims)]
        input_im = test_in[0:np.prod(dims)]

        predictions.append(reconstructed.reshape((1, dims[0], dims[1], dims[2])))
        tests.append(input_im.reshape((1, dims[0], dims[1], dims[2])))
        pred_labels.append(test_out[np.prod(dims):])
        label_inputs.append(test_in[np.prod(dims):])

    predictions = np.concatenate(predictions, axis=0)
    tests = np.concatenate(tests, axis=0)
    pred_labels = np.array(pred_labels).reshape(10, config.latent_dim)
    label_inputs = np.array(label_inputs).reshape(10, config.latent_dim)

    fig = display_with_labels(tests, label_inputs, predictions, pred_labels, n_labels=config.latent_dim)
    pdf.savefig(fig)

    final_outputs = []
    for test_ind in range(config.n):
        fig, final_im = recall_memories(
            test_data_squares, net, vae, dims, config.latent_dim, test_ind=test_ind,
            noise_factor=0.1, threshold=config.threshold, return_final_im=True, n=config.n
        )
        if fig is not None:
            pdf.savefig(fig)
            final_outputs.append(final_im)

    pdf.close()

    if config.return_errors_and_counts:
        final_outputs = (np.array(final_outputs) + 1) / 2
        errors = np.abs(test_data_squares[0:config.n] - final_outputs.reshape((config.n, dims[0], dims[1], dims[2])))
        squared_errors = errors ** 2
        sum_squared_errors = np.sum(squared_errors, axis=(1, 2, 3))
        mean_error = np.mean(sum_squared_errors)

        pixel_counts = np.where(abs(diff) > config.threshold, 1, 0)[0:config.n]
        non_zero_counts = np.count_nonzero(pixel_counts, axis=(1, 2, 3))
        mean_pixel_count = np.mean(non_zero_counts)

        return mean_error, mean_pixel_count, sum_squared_errors, non_zero_counts
