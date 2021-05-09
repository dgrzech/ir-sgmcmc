import json
from collections import OrderedDict
from itertools import repeat
from pathlib import Path

import SimpleITK as sitk
import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from vtk import vtkStructuredPointsReader
from vtk.util.numpy_support import vtk_to_numpy


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    """
    wrapper function for endless data loader
    """

    for loader in repeat(data_loader):
        yield from loader


def add_noise_uniform_field(field, alpha):
    return field + transform_coordinates(get_noise_uniform(field.shape, field.device, alpha))


def add_noise_Langevin(field, sigma, tau):
    return field + get_noise_Langevin(sigma, tau)


def get_noise_uniform(shape, device, alpha):
    return -2.0 * alpha * torch.rand(shape, device=device) + alpha


def get_noise_Langevin(sigma, tau):
    eps = torch.randn_like(sigma)
    return math.sqrt(2.0 * tau) * sigma * eps


def calc_det_J(nabla):
    """
    calculate the Jacobian determinant of a vector field

    :param nabla: field gradients
    :return: Jacobian determinant
    """

    nabla_x = nabla[..., 0]
    nabla_y = nabla[..., 1]
    nabla_z = nabla[..., 2]

    det_J = nabla_x[:, 0] * nabla_y[:, 1] * nabla_z[:, 2] + \
            nabla_y[:, 0] * nabla_z[:, 1] * nabla_x[:, 2] + \
            nabla_z[:, 0] * nabla_x[:, 1] * nabla_y[:, 2] - \
            nabla_x[:, 2] * nabla_y[:, 1] * nabla_z[:, 0] - \
            nabla_y[:, 2] * nabla_z[:, 1] * nabla_x[:, 0] - \
            nabla_z[:, 2] * nabla_x[:, 1] * nabla_y[:, 0]

    return det_J


def load_field(file_path, dims):
    reader = vtkStructuredPointsReader()
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()

    reader.SetFileName(file_path)
    reader.Update()

    data = reader.GetOutput()
    vectors = data.GetPointData().GetArray('field')
    vectors = vtk_to_numpy(vectors).reshape(1, *dims[2:], 3).transpose(0, 3, 2, 1, 4)

    field = torch.zeros(dims)
    field[:, 0] = torch.from_numpy(vectors[..., 0])
    field[:, 1] = torch.from_numpy(vectors[..., 1])
    field[:, 2] = torch.from_numpy(vectors[..., 2])

    return field


@torch.no_grad()
def calc_posterior_statistics(samples, device='cuda:0'):
    samples = samples.to(device)
    mean, std_dev = torch.mean(samples, dim=0), torch.std(samples, dim=0)
    del samples

    return mean, std_dev


@torch.no_grad()
def calc_DSC_GPU(no_samples, seg_fixed, seg_moving, structures_dict):
    """
    calculate the Dice scores
    """

    DSC = torch.zeros(no_samples, len(structures_dict))

    for idx in range(no_samples):
        seg_fixed_sample = seg_fixed[idx]
        seg_moving_sample = seg_moving[idx]

        for structure_idx, structure in enumerate(structures_dict):
            label = structures_dict[structure]

            numerator = 2.0 * ((seg_fixed_sample == label) * (seg_moving_sample == label)).sum()
            denominator = (seg_fixed_sample == label).sum() + (seg_moving_sample == label).sum()

            try:
                score = numerator / denominator
            except:
                score = 0.0

            DSC[idx, structure_idx] = score

    return DSC.numpy()


@torch.no_grad()
def calc_metrics(seg_fixed, seg_moving, structures_dict, spacing, GPU=True, no_samples=1):
    """
    calculate average surface distances and Dice scores
    """

    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
    ASD = np.zeros([no_samples, len(structures_dict)])

    if GPU:
        DSC = calc_DSC_GPU(no_samples, seg_fixed, seg_moving, structures_dict)
    else:
        DSC = np.zeros([no_samples, len(structures_dict)])
        overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()

    seg_fixed = seg_fixed.cpu().numpy()
    seg_moving = seg_moving.cpu().numpy()

    spacing = spacing.numpy().tolist()

    def calc_ASD(seg_fixed_im, seg_moving_im):
        seg_fixed_contour = sitk.LabelContour(seg_fixed_im)
        seg_moving_contour = sitk.LabelContour(seg_moving_im)

        hausdorff_distance_filter.Execute(seg_fixed_contour, seg_moving_contour)
        return hausdorff_distance_filter.GetAverageHausdorffDistance()

    def calc_DSC(seg_fixed_im, seg_moving_im):
        overlap_measures_filter.Execute(seg_fixed_im, seg_moving_im)
        return overlap_measures_filter.GetDiceCoefficient()

    for idx in range(no_samples):
        seg_fixed_arr = seg_fixed[idx].squeeze()
        seg_moving_arr = seg_moving[idx].squeeze()

        for structure_idx, structure_name in enumerate(structures_dict):
            label = structures_dict[structure_name]

            seg_fixed_structure = np.where(seg_fixed_arr == label, 1, 0)
            seg_moving_structure = np.where(seg_moving_arr == label, 1, 0)

            seg_fixed_im = sitk.GetImageFromArray(seg_fixed_structure)
            seg_moving_im = sitk.GetImageFromArray(seg_moving_structure)

            seg_fixed_im.SetSpacing(spacing)
            seg_moving_im.SetSpacing(spacing)

            try:
                ASD[idx, structure_idx] = calc_ASD(seg_fixed_im, seg_moving_im)
            except:
                ASD[idx, structure_idx] = np.inf

            if not GPU:
                DSC[idx, structure_idx] = calc_DSC(seg_fixed_im, seg_moving_im)

    return ASD, DSC


def calc_no_non_diffeomorphic_voxels(transformation, diff_op):
    nabla = diff_op(transformation, transformation=True)
    log_det_J_transformation = calc_det_J(nabla).log()
    return torch.isnan(log_det_J_transformation).sum(dim=(1, 2, 3)).cpu().numpy(), log_det_J_transformation


def calc_norm(field):
    """
    calculate the voxel-wise norm of vectors in a batch of 3D fields
    """

    norms = torch.empty(size=(field.shape[0], 1, field.shape[2], field.shape[3], field.shape[4]), device=field.device)

    for batch_idx in range(field.shape[0]):
        norms[batch_idx, ...] = torch.norm(field[batch_idx], p=2, dim=0)

    return norms


def get_log_path_from_run_ID(save_path, run_ID):
    return save_path + '/' + run_ID + '/log'


def get_module_attr(module, name):
    if isinstance(module, nn.DataParallel):
        return getattr(module.module, name)

    return getattr(module, name)


def get_samples_path_from_run_ID(save_path, run_ID):
    return save_path + '/' + run_ID + '/samples'


def im_flip(array):
    return np.fliplr(np.flipud(np.transpose(array, (1, 0))))


def init_identity_grid_2D(dims):
    """
    initialise a 2D identity grid
    """

    nx, ny = dims[0], dims[1]

    x = torch.linspace(-1, 1, steps=nx)
    y = torch.linspace(-1, 1, steps=ny)

    x = x.expand(ny, -1).unsqueeze(0).unsqueeze(3)
    y = y.expand(nx, -1).transpose(0, 1).unsqueeze(0).unsqueeze(3)

    return torch.cat((x, y), 3)


def init_identity_grid_3D(dims):
    """
    initialise a 3D identity grid
    """

    nx, ny, nz = dims[0], dims[1], dims[2]

    x = torch.linspace(-1, 1, steps=nx)
    y = torch.linspace(-1, 1, steps=ny)
    z = torch.linspace(-1, 1, steps=nz)

    x = x.expand(ny, -1).expand(nz, -1, -1).unsqueeze(0).unsqueeze(4)
    y = y.expand(nx, -1).expand(nz, -1, -1).transpose(1, 2).unsqueeze(0).unsqueeze(4)
    z = z.expand(nx, -1).transpose(0, 1).expand(ny, -1, -1).transpose(0, 1).unsqueeze(0).unsqueeze(4)

    return torch.cat((x, y, z), 4)


def max_field_update(field_old, field_new):
    """
    calculate the largest voxel-wise update to a vector field in terms of the L2 norm

    :param field_old: vector field before the update
    :param field_new: vector field after the update

    :return: voxel index and value of the largest update
    """

    norm_old = calc_norm(field_old)
    norm_new = calc_norm(field_new)

    diff = torch.abs(norm_new - norm_old)
    return torch.max(diff), torch.argmax(diff)


def pixel_to_normalised_2D(px_idx_x, px_idx_y, dim_x, dim_y):
    """
    transform the coordinates of a pixel to range (-1, 1)
    """

    x = -1.0 + 2.0 * px_idx_x / (dim_x - 1.0)
    y = -1.0 + 2.0 * px_idx_y / (dim_y - 1.0)

    return x, y


def pixel_to_normalised_3D(px_idx_x, px_idx_y, px_idx_z, dim_x, dim_y, dim_z):
    """
    transform the coordinates of a voxel to range (-1, 1)
    """

    x = -1.0 + 2.0 * px_idx_x / (dim_x - 1.0)
    y = -1.0 + 2.0 * px_idx_y / (dim_y - 1.0)
    z = -1.0 + 2.0 * px_idx_z / (dim_z - 1.0)

    return x, y, z


def rescale_im(im, range_min=0.0, range_max=1.0):
    """
    rescale the intensity of image pixels/voxels to a given range
    """

    im_min, im_max = torch.min(im), torch.max(im)
    return (range_max - range_min) * (im - im_min) / (im_max - im_min) + range_min


def rescale_residuals(res, mask, data_loss):
    """
    rescale residuals by the estimated voxel-wise standard deviation
    """

    res_masked = torch.where(mask, res, torch.zeros_like(res))
    res_masked_flattened = res_masked.view(1, -1, 1)

    log_std = data_loss.log_std
    scaled_res = res_masked_flattened * torch.exp(-1.0 * log_std)

    scaled_res.requires_grad_(True)
    scaled_res.retain_grad()

    loss_VD = -1.0 * data_loss.log_pdf_VD(scaled_res).sum()
    loss_VD.backward()

    return torch.sum(scaled_res * scaled_res.grad, dim=-1).view(res.shape)


def separable_conv_3D(field, *args):
    """
    implements separable convolution over a three-dimensional vector field either as three 1D convolutions
    with a 1D kernel, or as three 3D convolutions with three 3D kernels of sizes kx1x1, 1xkx1, and 1x1xk

    :param field: input vector field
    :param args: input kernel(s) and the size of padding to use
    :return: input vector field convolved with the kernel
    """

    field_out = field.clone()

    if len(args) == 2:
        kernel = args[0]
        padding_sz = args[1]

        N, C, D, H, W = field_out.shape

        padding_3D = (padding_sz, padding_sz, 0, 0, 0, 0)

        field_out = F.pad(field_out, padding_3D, mode='replicate')
        field_out = field_out.view(N, C, -1)
        field_out = F.conv1d(field_out, kernel, padding=padding_sz, groups=3)  # depth
        field_out = field_out.reshape(N, C, D, H, -1)
        field_out = field_out[:, :, :, :, padding_sz:-padding_sz]

        field_out = field_out.permute((0, 1, 3, 4, 2))  # permute depth, height, and width

        field_out = F.pad(field_out, padding_3D, mode='replicate')
        field_out = field_out.view(N, C, -1)
        field_out = F.conv1d(field_out, kernel, padding=padding_sz, groups=3)  # height
        field_out = field_out.reshape(N, C, D, H, -1)
        field_out = field_out[:, :, :, :, padding_sz:-padding_sz]

        field_out = field_out.permute((0, 1, 3, 4, 2))

        field_out = F.pad(field_out, padding_3D, mode='replicate')
        field_out = field_out.view(N, C, -1)
        field_out = F.conv1d(field_out, kernel, padding=padding_sz, groups=3)  # width
        field_out = field_out.reshape(N, C, D, H, -1)
        field_out = field_out[:, :, :, :, padding_sz:-padding_sz]

        field_out = field_out.permute((0, 1, 3, 4, 2))  # back to the orig. dimensions

    elif len(args) == 4:
        kernel_x = args[0]
        kernel_y = args[1]
        kernel_z = args[2]
        padding = args[3]

        field_out = F.pad(field_out, padding, mode='replicate')

        field_out = F.conv3d(field_out, kernel_z, groups=3)
        field_out = F.conv3d(field_out, kernel_y, groups=3)
        field_out = F.conv3d(field_out, kernel_x, groups=3)

    return field_out


def standardise_im(im):
    """
    standardise image to zero mean and unit variance
    """

    im_mean, im_std = torch.mean(im), torch.std(im)
    return (im - im_mean) / im_std


def transform_coordinates(field):
    """
    coordinate transformation from absolute coordinates (0, 1, ..., n) to normalised (-1, ..., 1)
    """

    field_out = field.clone()
    no_dims, dims = field.shape[1], field.shape[2:]

    for idx in range(no_dims):
        field_out[:, idx] = field_out[:, idx] * 2.0 / float(dims[idx] - 1)

    return field_out


def transform_coordinates_inv(field):
    """
    coordinate transformation from normalised coordinates (-1, ..., 1) to absolute (0, 1, ..., n)
    """

    field_out = field.clone()
    no_dims, dims = field.shape[1], field.shape[2:]

    for idx in range(no_dims):
        field_out[:, idx] = field_out[:, idx] * float(dims[idx] - 1) / 2.0

    return field_out


@torch.no_grad()
def calc_VD_factor(residual, mask):
    """
    virtual decimation

    input x = residual (Gaussian-ish) field with stationary covariance, e.g. residual map (I-J) / sigma,
    where sigma is the noise sigma if you use SSD/Gaussian model or else the EM voxel-wise estimate if you use a GMM.

    EM voxel-wise estimate of precision = var^(-1) is sum_k rho_k precision_k,
    where rho_k is the component responsible for the voxel.

    The general idea is that each voxel-wise observation now only counts for "VD < 1 of an observation";
    imagine sampling a z ~ bernoulli(VD) at each voxel and you only add the voxel's loss if z == 1.

    In practice you do that in expectation. In the simplest case it looks like VD * data_loss,
    and goes well in a VB framework, as if you added a q(z) = Bernoulli(VD) to a VB approximation
    and took the expectation wrt q(z).
    """

    # variance
    residual_masked = residual[mask]
    var_res = torch.mean(residual_masked ** 2)

    # covariance..
    no_unmasked_voxels = mask.sum()
    residual_masked = torch.where(mask, residual, torch.zeros_like(residual))

    cov_x = torch.sum(residual_masked[:, :, :-1] * residual_masked[:, :, 1:]) / no_unmasked_voxels
    cov_y = torch.sum(residual_masked[:, :, :, :-1] * residual_masked[:, :, :, 1:]) / no_unmasked_voxels
    cov_z = torch.sum(residual_masked[:, :, :, :, :-1] * residual_masked[:, :, :, :, 1:]) / no_unmasked_voxels

    corr_x = cov_x / var_res
    corr_y = cov_y / var_res
    corr_z = cov_z / var_res

    sq_VD_x = torch.clamp(-2.0 / math.pi * torch.log(corr_x), max=1.0)
    sq_VD_y = torch.clamp(-2.0 / math.pi * torch.log(corr_y), max=1.0)
    sq_VD_z = torch.clamp(-2.0 / math.pi * torch.log(corr_z), max=1.0)

    return torch.sqrt(sq_VD_x * sq_VD_y * sq_VD_z)


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)

        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
