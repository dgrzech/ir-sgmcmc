import json
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import torch

from utils import calc_norm, im_flip
from .writer import SummaryWriter


class TensorboardWriter:
    def __init__(self, log_dir, enabled):
        self.selected_module = ''
        self.timer = datetime.now()
        self.writer = None

        if enabled:
            log_dir = str(log_dir)
            self.writer = SummaryWriter(log_dir)

        self.step = 0
        self.tag_mode_exceptions = {'add_histogram', 'add_embedding'}
        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio', 'add_figure',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding', 'add_histogram'
        }

        self.hist_axlims = None

    def set_step(self, step):
        self.step = step

    def write_hparams(self, config):
        text = json.dumps(config, indent=4, sort_keys=False)
        self.writer.add_text('hparams', text)

    def __getattr__(self, name):
        """
        if visualization is configured, return add_data() methods of tensorboard with additional information (step, tag) added; else return a blank function handle that does nothing
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    add_data(tag, data, self.step, *args, **kwargs)

            return wrapper
        else:  # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object '{}' has no attribute '{}'".format(self.selected_module, name))
            return attr


"""
histogram of residuals
"""


@torch.no_grad()
def log_hist_res(writer, residuals, data_loss, model='VI', chain_no=None):
    device = residuals.device
    residuals = residuals.cpu().numpy()

    fig, ax = plt.subplots()
    g = sns.histplot(data=residuals, stat="density", ax=ax)

    if writer.hist_axlims is None:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        writer.hist_axlims = {'x': (-1.0 * xmax, xmax), 'y': (ymin, ymax + 1.0)}

    g.set(xlim=writer.hist_axlims['x'])
    g.set(ylim=writer.hist_axlims['y'])
    g.set(ylabel='')

    x = torch.linspace(*writer.hist_axlims['x'], steps=10000).unsqueeze(0).unsqueeze(-1).to(device)
    model_fit = data_loss.log_pdf(x).exp().squeeze().cpu().numpy()
    x = x.detach().squeeze().cpu().numpy()

    sns.lineplot(x=x, y=model_fit, color='green', ax=ax)
    figure_name = 'VI/hist_residuals' if model == 'VI' else f'MCMC/chain_{chain_no}/hist_residuals'
    writer.add_figure(figure_name, fig)


"""
images
"""


def im_grid(im_fixed_slices, im_moving_slices, im_moving_warped_slices, im_diff_slices):
    """
    plot of input and output images to log in tensorboard
    """

    fig, axs = plt.subplots(nrows=4, ncols=3, sharex=True, sharey=True, figsize=(8, 8))

    cols = ['axial', 'coronal', 'sagittal']
    rows = ['im_fixed', 'im_moving', 'im_moving_warped', 'im_diff']

    for ax, col in zip(axs[0], cols):
        ax.set_title(col)

    for ax, row in zip(axs[:, 0], rows):
        ax.set_xticks([], minor=False)
        ax.set_xticks([], minor=True)

        ax.set_yticks([], minor=False)
        ax.set_yticks([], minor=True)

        ax.set_ylabel(row, rotation=90, size='large')

    for i in range(3):
        axs[0, i].imshow(im_flip(im_fixed_slices[i]), cmap='gray')
        axs[1, i].imshow(im_flip(im_moving_slices[i]), cmap='gray')
        axs[2, i].imshow(im_flip(im_moving_warped_slices[i]), cmap='gray')
        axs[3, i].imshow(im_flip(im_diff_slices[i]), cmap='gray')

    return fig


def get_im_or_field_mid_slices_idxs(im_or_field):
    return int(im_or_field.shape[4] / 2), int(im_or_field.shape[3] / 2), int(im_or_field.shape[2] / 2)


def get_slices(field, mid_idxs):
    return [field[:, :, mid_idxs[0]], field[:, mid_idxs[1], :], field[mid_idxs[2], :, :]]


def log_images(writer, im_fixed_batch, im_moving_batch, im_moving_warped_batch):
    mid_idxs = get_im_or_field_mid_slices_idxs(im_fixed_batch)

    im_fixed = im_fixed_batch[0, 0].cpu().numpy()
    im_moving = im_moving_batch[0, 0].cpu().numpy()
    im_moving_warped = im_moving_warped_batch[0, 0].cpu().numpy()
    im_diff = torch.abs(im_fixed_batch - im_moving_warped_batch)[0, 0].cpu().numpy()

    im_fixed_slices = get_slices(im_fixed, mid_idxs)
    im_moving_slices = get_slices(im_moving, mid_idxs)
    im_moving_warped_slices = get_slices(im_moving_warped, mid_idxs)
    im_diff_slices = get_slices(im_diff, mid_idxs)

    writer.add_figure('VI/images', im_grid(im_fixed_slices, im_moving_slices, im_moving_warped_slices, im_diff_slices))


"""
vector fields
"""


def fields_grid(mu_v_norm_slices, displacement_norm_slices, sigma_v_norm_slices, u_v_norm_slices, log_det_J_slices):
    """
    plot of the norms of output vector fields to log in tensorboard
    """

    fig, axs = plt.subplots(nrows=5, ncols=3, sharex=True, sharey=True, figsize=(10, 10))

    rows = ['mu_v_norm', 'displacement_norm', 'sigma_v_norm', 'u_v_norm', 'log_det_J']
    cols = ['axial', 'coronal', 'sagittal']

    for ax, col in zip(axs[0], cols):
        ax.set_title(col)

    for ax, row in zip(axs[:, 0], rows):
        ax.set_xticks([], minor=False)
        ax.set_xticks([], minor=True)

        ax.set_yticks([], minor=False)
        ax.set_yticks([], minor=True)

        ax.set_ylabel(row, rotation=90, size='large')

    for i in range(3):
        axs[0, i].imshow(im_flip(mu_v_norm_slices[i]), cmap='hot')
        axs[1, i].imshow(im_flip(displacement_norm_slices[i]), cmap='hot')
        axs[2, i].imshow(im_flip(sigma_v_norm_slices[i]), cmap='hot')
        axs[3, i].imshow(im_flip(u_v_norm_slices[i]), cmap='hot')
        axs[4, i].imshow(im_flip(log_det_J_slices[i]))

    return fig


def log_fields(writer, var_params_batch, displacement_batch, log_det_J_batch):
    mid_idxs_ims = get_im_or_field_mid_slices_idxs(displacement_batch)
    mid_idxs_params = get_im_or_field_mid_slices_idxs(var_params_batch['mu'])

    mu_v_norm = calc_norm(var_params_batch['mu'])[0, 0].cpu().numpy()
    sigma_v_norm = calc_norm(torch.exp(0.5 * var_params_batch['log_var']))[0, 0].cpu().numpy()
    u_v_norm = calc_norm(var_params_batch['u'])[0, 0].cpu().numpy()

    displacement_norm = calc_norm(displacement_batch)[0, 0].cpu().numpy()
    log_det_J = log_det_J_batch[0].cpu().numpy()

    mu_v_norm_slices = get_slices(mu_v_norm, mid_idxs_params)
    sigma_v_norm_slices = get_slices(sigma_v_norm, mid_idxs_params)
    u_v_norm_slices = get_slices(u_v_norm, mid_idxs_params)

    displacement_norm_slices = get_slices(displacement_norm, mid_idxs_ims)
    log_det_J_slices = get_slices(log_det_J, mid_idxs_ims)

    writer.add_figure('VI/q_v', fields_grid(mu_v_norm_slices, displacement_norm_slices, sigma_v_norm_slices, u_v_norm_slices, log_det_J_slices))


"""
samples
"""


def sample_grid(im_moving_warped_slices, v_norm_slices, displacement_norm_slices, log_det_J_slices):
    """
    plot of output images and vector fields related to a sample from MCMC to log in tensorboard
    """

    fig, axs = plt.subplots(nrows=4, ncols=3, sharex=True, sharey=True, figsize=(8, 8))

    cols = ['axial', 'coronal', 'sagittal']
    rows = ['im_moving_warped', 'v_curr_state_norm', 'displacement_norm', 'log_det_J']

    for ax, col in zip(axs[0], cols):
        ax.set_title(col)

    for ax, row in zip(axs[:, 0], rows):
        ax.set_xticks([], minor=False)
        ax.set_xticks([], minor=True)

        ax.set_yticks([], minor=False)
        ax.set_yticks([], minor=True)

        ax.set_ylabel(row, rotation=90, size='large')

    for i in range(3):
        axs[0, i].imshow(im_flip(im_moving_warped_slices[i]), cmap='gray')
        axs[1, i].imshow(im_flip(v_norm_slices[i]), cmap='hot')
        axs[2, i].imshow(im_flip(displacement_norm_slices[i]), cmap='hot')
        axs[3, i].imshow(im_flip(log_det_J_slices[i]))

    return fig


def log_sample(writer, chain_idx, im_moving_warped, v_norm, displacement_norm, log_det_J):
    mid_idxs_params = get_im_or_field_mid_slices_idxs(v_norm)
    mid_idxs_ims = get_im_or_field_mid_slices_idxs(im_moving_warped)

    im_moving_warped = im_moving_warped[chain_idx, 0].cpu().numpy()
    v_norm = v_norm[chain_idx, 0].cpu().numpy()
    displacement_norm = displacement_norm[chain_idx, 0].cpu().numpy()
    log_det_J = log_det_J[chain_idx].cpu().numpy()

    im_moving_warped_slices = get_slices(im_moving_warped, mid_idxs_ims)
    v_norm_slices = get_slices(v_norm, mid_idxs_params)
    displacement_norm_slices = get_slices(displacement_norm, mid_idxs_ims)
    log_det_J_slices = get_slices(log_det_J, mid_idxs_ims)

    writer.add_figure(f'MCMC/chain_{chain_idx}/samples',
                      sample_grid(im_moving_warped_slices, v_norm_slices, displacement_norm_slices, log_det_J_slices))


def summary_stats_grid(displacement_mean_slices, displacement_std_dev_slices):
    fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(8, 8))

    cols = ['axial', 'coronal', 'sagittal']
    rows = ['displacement_mean', 'displacement_std_dev']

    for ax, col in zip(axs[0], cols):
        ax.set_title(col)

    for ax, row in zip(axs[:, 0], rows):
        ax.set_xticks([], minor=False)
        ax.set_xticks([], minor=True)

        ax.set_yticks([], minor=False)
        ax.set_yticks([], minor=True)

        ax.set_ylabel(row, rotation=90, size='large')

    for i in range(3):
        axs[0, i].imshow(im_flip(displacement_mean_slices[i]), cmap='hot')
        axs[1, i].imshow(im_flip(displacement_std_dev_slices[i]), cmap='hot')

    return fig


def log_displacement_mean_and_std_dev(writer, displacement_mean, displacement_std_dev, model):
    displacement_mean_norm = calc_norm(displacement_mean.unsqueeze(0)).cpu().numpy()
    displacement_std_dev_norm = calc_norm(displacement_std_dev.unsqueeze(0)).cpu().numpy()

    mid_idxs = get_im_or_field_mid_slices_idxs(displacement_mean_norm)

    displacement_mean_norm_slices = get_slices(displacement_mean_norm[0, 0], mid_idxs)
    displacement_std_dev_norm_slices = get_slices(displacement_std_dev_norm[0, 0], mid_idxs)

    writer.add_figure(f'{model}/summary_stats',
                      summary_stats_grid(displacement_mean_norm_slices, displacement_std_dev_norm_slices))
