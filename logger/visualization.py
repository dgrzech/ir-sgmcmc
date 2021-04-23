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


def log_hist_res(writer, im_pair_idxs, residuals_batch, data_loss):
    device = residuals_batch.device
    residuals_batch = residuals_batch.view(1, -1).cpu().numpy()

    for loop_idx, im_pair_idx in enumerate(im_pair_idxs):
        residuals = residuals_batch[loop_idx]

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
        model_fit = torch.exp(data_loss.log_pdf(x)).detach().squeeze().cpu().numpy()
        x = x.detach().squeeze().cpu().numpy()

        sns.lineplot(x=x, y=model_fit, color='green', ax=ax)
        writer.add_figure('hist_residuals/' + str(im_pair_idx), fig)


"""
images
"""


def im_grid(im_fixed_slices, im_moving_slices, im_moving_warped_slices):
    """
    plot of input and output images to log in tensorboard
    """

    fig, axs = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(8, 8))

    cols = ['axial', 'coronal', 'sagittal']
    rows = ['im_fixed', 'im_moving', 'im_moving_warped']

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

    return fig


def get_im_or_field_mid_slices_idxs(im_or_field):
    return int(im_or_field.shape[4] / 2), int(im_or_field.shape[3] / 2), int(im_or_field.shape[2] / 2)


def get_slices(field, mid_idxs):
    return [field[:, :, mid_idxs[0]], field[:, mid_idxs[1], :], field[mid_idxs[2], :, :]]


def log_images(writer, im_pair_idxs, im_fixed_batch, im_moving_batch, im_moving_warped_batch):
    im_fixed_batch = im_fixed_batch.cpu().numpy()
    im_moving_batch = im_moving_batch.cpu().numpy()
    im_moving_warped_batch = im_moving_warped_batch.cpu().numpy()

    mid_idxs = get_im_or_field_mid_slices_idxs(im_fixed_batch)

    for loop_idx, im_pair_idx in enumerate(im_pair_idxs):
        im_fixed = im_fixed_batch[loop_idx, 0]
        im_moving = im_moving_batch[loop_idx, 0]
        im_moving_warped = im_moving_warped_batch[loop_idx, 0]

        im_fixed_slices = get_slices(im_fixed, mid_idxs)
        im_moving_slices = get_slices(im_moving, mid_idxs)
        im_moving_warped_slices = get_slices(im_moving_warped, mid_idxs)

        writer.add_figure('im_pairs/' + str(im_pair_idx),
                          im_grid(im_fixed_slices, im_moving_slices, im_moving_warped_slices))


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


def log_fields(writer, im_pair_idxs, var_params_batch, displacement_batch, log_det_J_batch):
    mu_v_norm_batch = calc_norm(var_params_batch['mu']).cpu().numpy()
    sigma_v_norm_batch = calc_norm(torch.exp(0.5 * var_params_batch['log_var'])).cpu().numpy()
    u_v_norm_batch = calc_norm(var_params_batch['u']).cpu().numpy()

    displacement_norm_batch = calc_norm(displacement_batch).cpu().numpy()
    log_det_J_batch = log_det_J_batch.cpu().numpy()

    mid_idxs = get_im_or_field_mid_slices_idxs(mu_v_norm_batch)

    for loop_idx, im_pair_idx in enumerate(im_pair_idxs):
        mu_v_norm = mu_v_norm_batch[loop_idx, 0]
        sigma_v_norm = sigma_v_norm_batch[loop_idx, 0]
        u_v_norm = u_v_norm_batch[loop_idx, 0]

        displacement_norm = displacement_norm_batch[loop_idx, 0]
        log_det_J = log_det_J_batch[loop_idx]

        mu_v_norm_slices = get_slices(mu_v_norm, mid_idxs)
        sigma_v_norm_slices = get_slices(sigma_v_norm, mid_idxs)
        u_v_norm_slices = get_slices(u_v_norm, mid_idxs)

        displacement_norm_slices = get_slices(displacement_norm, mid_idxs)
        log_det_J_slices = get_slices(log_det_J, mid_idxs)

        writer.add_figure('q_v/' + str(im_pair_idx),
                          fields_grid(mu_v_norm_slices, displacement_norm_slices,
                                      sigma_v_norm_slices, u_v_norm_slices, log_det_J_slices))


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


def log_sample(writer, im_pair_idxs, data_loss, res_batch, im_moving_warped_batch, v_batch, displacement_batch,
               log_det_J_batch):
    log_hist_res(writer, im_pair_idxs, res_batch, data_loss)

    im_moving_warped_batch = im_moving_warped_batch.cpu().numpy()
    v_norm_batch = calc_norm(v_batch).cpu().numpy()
    displacement_norm_batch = calc_norm(displacement_batch).cpu().numpy()
    log_det_J_batch = log_det_J_batch.cpu().numpy()

    mid_idxs = get_im_or_field_mid_slices_idxs(v_batch)

    for loop_idx, im_pair_idx in enumerate(im_pair_idxs):
        im_moving_warped = im_moving_warped_batch[loop_idx, 0]
        v_norm = v_norm_batch[loop_idx, 0]
        displacement_norm = displacement_norm_batch[loop_idx, 0]
        log_det_J = log_det_J_batch[loop_idx]

        im_moving_warped_slices = get_slices(im_moving_warped, mid_idxs)
        v_norm_slices = get_slices(v_norm, mid_idxs)
        displacement_norm_slices = get_slices(displacement_norm, mid_idxs)
        log_det_J_slices = get_slices(log_det_J, mid_idxs)

        writer.add_figure('samples/' + str(im_pair_idx),
                          sample_grid(im_moving_warped_slices, v_norm_slices, displacement_norm_slices,
                                      log_det_J_slices))
