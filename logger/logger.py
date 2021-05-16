import logging
import logging.config
from os import path
from pathlib import Path

import nibabel as nib
import numpy as np
from tvtk.api import tvtk, write_data

from utils import read_json


def setup_logging(save_dir, log_config='logger/logger_config.json', default_level=logging.INFO):
    """
    setup logging configuration
    """

    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)

        # modify logging paths based on run config
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(save_dir / handler['filename'])

        logging.config.dictConfig(config)
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)

    print()


def save_field_to_disk(field, file_path, spacing=(1, 1, 1)):
    """
    save a vector field to a .vtk file

    :param field: field to save
    :param file_path: path to use
    :param spacing: voxel spacing
    """

    try:
        spacing = spacing.numpy()
    except:
        pass

    field_x, field_y, field_z = field[0], field[1], field[2]

    vectors = np.empty(field_x.shape + (3, ), dtype=float)
    vectors[..., 0], vectors[..., 1], vectors[..., 2] = field_x, field_y, field_z
    vectors = vectors.transpose(2, 1, 0, 3).copy()  # NOTE (DG): reorder the vectors per VTK requirement of x first, y next and z last
    vectors.shape = vectors.size // 3, 3

    im_vtk = tvtk.ImageData(spacing=spacing, origin=(0, 0, 0), dimensions=field_x.shape)
    im_vtk.point_data.vectors = vectors
    im_vtk.point_data.vectors.name = 'field'

    write_data(im_vtk, file_path)


def save_grid_to_disk(grid, file_path):
    """
    save a VTK structured grid to a .vtk file

    :param grid: grid to save
    :param file_path: path to use
    """

    grid = grid.cpu().numpy()

    x, y, z = grid[0, ...], grid[1, ...], grid[2, ...]

    pts = np.empty(x.shape + (3, ), dtype=float)
    pts[..., 0], pts[..., 1], pts[..., 2] = x, y, z
    pts = pts.transpose(2, 1, 0, 3).copy()
    pts.shape = pts.size // 3, 3

    sg = tvtk.StructuredGrid(dimensions=x.shape, points=pts)
    write_data(sg, file_path)


def save_im_to_disk(im, file_path, spacing=(1, 1, 1)):
    """
    save an image stored in a numpy array to a .nii.gz file

    :param im: 3D image
    :param file_path: path to use
    :param spacing: voxel spacing
    """

    im = nib.Nifti1Image(im, np.eye(4))
    im.header.set_xyzt_units(2)

    try:
        spacing = spacing.numpy()
        im.header.set_zooms(spacing)
    except:
        im.header.set_zooms(spacing)

    im.to_filename(file_path)


"""
(vector) fields
"""


def save_displacement_mean_and_std_dev(logger, save_dirs, spacing, displacement_mean, displacement_std_dev, mask, model):
    folder = save_dirs['samples']
    
    mean_path = path.join(folder, f'{model}_sample_mean.vtk')
    mean_masked_path = path.join(folder, f'{model}_sample_mean_masked.vtk')

    std_dev_path = path.join(folder, f'{model}_sample_std_dev.vtk')
    std_dev_masked_path = path.join(folder, f'{model}_sample_std_dev_masked.vtk')

    displacement_mean = displacement_mean * spacing[0]
    logger.info(f'{model} displacement mean min.: {displacement_mean.min().item():.2f}, displacement mean max.: {displacement_mean.max().item():.2f}')
    
    save_field_to_disk(displacement_mean.cpu().numpy(), mean_path, spacing)
    displacement_mean_masked = displacement_mean * mask[0]
    save_field_to_disk(displacement_mean_masked.cpu().numpy(), mean_masked_path, spacing)

    displacement_std_dev = displacement_std_dev * spacing[0]
    logger.info(f'{model} displacement std. dev. min.: {displacement_std_dev.min().item():.2f}, displacement std. dev. max.: {displacement_std_dev.max().item():.2f}')

    save_field_to_disk(displacement_std_dev.cpu().numpy(), std_dev_path, spacing)
    displacement_std_dev_masked = displacement_std_dev * mask[0]
    save_field_to_disk(displacement_std_dev_masked.cpu().numpy(), std_dev_masked_path, spacing)


def save_field(save_dirs, spacing, field, field_name, model=None):
    folder = save_dirs['samples'] / model if model is not None else save_dirs['fields']
    field_path = path.join(folder, f'{field_name}.vtk')
    save_field_to_disk(field, field_path, spacing)


"""
grids
"""


def save_grids(im_pair_idxs, save_dirs, grids):
    """
    save output structured grids to .vtk
    """

    for loop_idx, im_pair_idx in enumerate(im_pair_idxs):
        grid_path = path.join(save_dirs['grids'], f'grid_{im_pair_idx}.vtk')
        grid = grids[loop_idx]
        save_grid_to_disk(grid, grid_path)


"""
images
"""


def save_im(save_dirs, spacing, im, name, model=None):
    folder = save_dirs['samples'] / model if model is not None else save_dirs['images']
    im_path = path.join(folder, f'{name}.nii.gz')
    save_im_to_disk(im, im_path, spacing)


def save_fixed_im(save_dirs, spacing, im_fixed):
    """
    save the input fixed image to .nii.gz
    """

    im_fixed = im_fixed[0, 0].cpu().numpy()
    im_path = path.join(save_dirs['images'], 'im_fixed.nii.gz')
    save_im_to_disk(im_fixed, im_path, spacing)


def save_fixed_mask(save_dirs, spacing, mask_fixed):
    mask_fixed = mask_fixed[0, 0].float().cpu().numpy()
    im_path = path.join(save_dirs['images'], 'mask_fixed.nii.gz')
    save_im_to_disk(mask_fixed, im_path, spacing)


def save_moving_im(save_dirs, spacing, im_moving_batch):
    """
    save input moving images to .nii.gz
    """

    im_moving = im_moving_batch[0, 0].cpu().numpy()
    save_im(save_dirs, spacing, im_moving, 'im_moving')


def save_moving_mask(save_dirs, spacing, mask_moving):
    mask_moving = mask_moving[0, 0].float().cpu().numpy()
    im_path = path.join(save_dirs['images'], 'mask_moving.nii.gz')
    save_im_to_disk(mask_moving, im_path, spacing)


def save_variational_posterior_mean(save_dirs, spacing, im_moving_warped, displacement):
    im_moving_warped = im_moving_warped[0, 0].cpu().numpy()
    displacement = displacement * spacing[0]
    displacement = displacement[0].cpu().numpy()

    name = f'im_moving_warped_mu'
    save_im(save_dirs, spacing, im_moving_warped, name)

    name = 'displacement_mu'
    save_field(save_dirs, spacing, displacement, name)


"""
samples
"""


def save_sample(save_dirs, spacing, sample_no, im_moving_warped_batch, displacement_batch, log_det_J_batch,
                model, chain_no=None):
    """
    save output images and vector fields related to a sample from VI or MCMC
    """

    im_moving_warped_batch = im_moving_warped_batch.cpu().numpy()

    displacement_batch = displacement_batch * spacing[0]
    displacement_batch = displacement_batch.cpu().numpy()

    log_det_J_batch = log_det_J_batch.cpu().numpy()

    im_moving_warped = im_moving_warped_batch[0, 0]
    name = f'chain_{chain_no}_sample_{sample_no:07}_im_moving_warped' if model == 'MCMC' else f'sample_{sample_no:07}_im_moving_warped'
    save_im(save_dirs, spacing, im_moving_warped, name, model)

    displacement = displacement_batch[0]
    name = f'chain_{chain_no}_sample_{sample_no:07}_displacement' if model == 'MCMC' else f'sample_{sample_no:07}_displacement'
    save_field(save_dirs, spacing, displacement, name, model)

    log_det_J = log_det_J_batch[0]
    name = f'chain_{chain_no}_sample_{sample_no:07}_log_det_J' if model == 'MCMC' else f'sample_{sample_no:07}_log_det_J'
    save_im(save_dirs, spacing, log_det_J, name, model)
