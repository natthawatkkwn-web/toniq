"""Simplified helpers for running the TONIQ analyses step by step in a notebook.

Messages are printed in Thai for the accompanying notebook; plot labels are in English since
matplotlib's default font lacks Thai glyphs.
"""
import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp

from toniq import ia, gd, snr, sr
from toniq.dicom import load_series_from_path, sort_series
from toniq.masks import get_implant_mask, get_signal_mask, get_artifact_mask
from toniq.util import equalize, masked_copy, normalize

AXIS_NAMES = ('axis 0 (rows)', 'axis 1 (columns)', 'axis 2 (slices)')


def sort_export(export_folder, sorted_folder):
    """ Sort a scanner export into Series<N> folders and print a table of series. """
    series = sort_series(export_folder, sorted_folder)
    if len(series) == 0:
        print('ไม่พบไฟล์ภาพ MR ใน {} กรุณาตรวจชื่อโฟลเดอร์'.format(export_folder))
        return series
    print('จัดไฟล์เสร็จแล้ว ได้ {} series ไว้ที่ {}\n'.format(len(series), sorted_folder))
    print('{:<14}{:>7}{:>8}  {}'.format('โฟลเดอร์', 'ไฟล์', 'ภาพ', 'ชื่อ protocol'))
    for info in series:
        print('{:<14}{:>7}{:>8}  {}'.format(info['folder'], info['files'], info['frames'], info['description']))
    return series


def load_volumes(sorted_folder, series):
    """ Load each named series, skipping names set to None, and check that their shapes agree. """
    volumes = {}
    for name, folder in series.items():
        if folder is None:
            continue
        volumes[name] = load_series_from_path(os.path.join(sorted_folder, folder))
        v = volumes[name]
        print('{:<30}{:<12}ขนาด {}  ความละเอียด {} mm  ({})'.format(
            name, folder, v.shape, tuple(round(r, 2) for r in v.meta.resolution_mm), v.meta.seriesName))
    shapes = {name: v.shape for name, v in volumes.items() if name != 'structured-plastic-reference'}
    if len(set(shapes.values())) > 1:
        raise ValueError('ภาพแต่ละ series มีขนาดไม่เท่ากัน ต้องสแกนด้วย FOV และ matrix เดียวกันทุก series: {}'.format(shapes))
    print('\nโหลดครบ {} series ขนาดภาพตรงกัน'.format(len(volumes)))
    return volumes


def to_slices(crop, shape):
    """ Convert crop given as [[start, stop], [start, stop], [start, stop]] (or None for no crop) to a tuple of slices. """
    if crop is None:
        return tuple(slice(0, n) for n in shape)
    return tuple(slice(start, stop) for start, stop in crop)


def show_overview(volumes, crop=None, name='uniform-metal', save_folder=None):
    """ Show three orthogonal views through the center of the crop region, with the crop region outlined. """
    image = normalize(np.abs(volumes[name].data))
    slc = to_slices(crop, image.shape)
    center = [(s.start + s.stop) // 2 for s in slc]
    res = volumes[name].meta.resolution_mm
    views = (2, 1, 0)
    ratios = []  # physical width / height of each view, so that the views fill the figure
    for axis in views:
        rows, cols = [a for a in range(3) if a != axis]
        ratios.append(image.shape[cols] * res[cols] / (image.shape[rows] * res[rows]))
    fig, axes = plt.subplots(1, 3, figsize=(4.5 * sum(ratios) + 2, 5), gridspec_kw={'width_ratios': ratios})
    for ax, axis in zip(axes, views):
        view = np.take(image, center[axis], axis=axis)
        rows, cols = [a for a in range(3) if a != axis]
        ax.imshow(view, cmap='gray', vmin=0, vmax=1, aspect=res[rows] / res[cols])
        ax.add_patch(plt.Rectangle((slc[cols].start, slc[rows].start), slc[cols].stop - slc[cols].start, slc[rows].stop - slc[rows].start,
                                   fill=False, edgecolor='yellow', linewidth=2))
        ax.set_title('{} = {}'.format(AXIS_NAMES[axis], center[axis]))
        ax.set_ylabel(AXIS_NAMES[rows])
        ax.set_xlabel(AXIS_NAMES[cols])
    fig.suptitle('{}: yellow box = crop region'.format(name))
    fig.tight_layout()
    _save(fig, save_folder, 'overview.png')
    plt.show()


def prepare(volumes, names, crop):
    """ Equalize intensities of the named volumes (on the full volume), then crop. """
    images = equalize([volumes[name].data for name in names])
    slc = to_slices(crop, images[0].shape)
    return [image[slc] for image in images]


def run_ia(volumes, crop):
    """ Intensity artifact map from the uniform plastic & metal scans. """
    plastic, metal = prepare(volumes, ('uniform-plastic', 'uniform-metal'), crop)
    implant_mask = get_implant_mask(plastic)
    ia_map = ia.get_map(plastic, metal, implant_mask)
    print('คำนวณ IA map เสร็จแล้ว')
    return {'plastic': plastic, 'metal': metal, 'implant_mask': implant_mask, 'ia_map': ia_map}


def run_gd(volumes, crop, ia_result, ia_threshold):
    """ Geometric distortion map from the structured plastic & metal scans. """
    plastic, metal = prepare(volumes, ('structured-plastic', 'structured-metal'), crop)
    plastic_mask, metal_mask = gd.get_masks(ia_result['implant_mask'], ia_result['ia_map'], ia_threshold)
    _, registered_masked, gd_map = gd.get_map(plastic, metal, plastic_mask, metal_mask)
    mask = (registered_masked != 0) * (masked_copy(metal, metal_mask) != 0)
    print('คำนวณ GD map เสร็จแล้ว')
    return {'plastic': plastic, 'metal': metal, 'gd_map': gd_map, 'mask': mask}


def run_snr(volumes, crop, ia_result, ia_threshold):
    """ SNR map from the repeated uniform metal scans. """
    image1, image2 = prepare(volumes, ('uniform-metal', 'uniform-metal-2'), crop)
    ia_mask = get_artifact_mask(ia_result['ia_map'], ia_threshold)
    mask = get_signal_mask(ia_result['implant_mask'], artifact_masks=[ia_mask])
    snr_map = snr.get_map(image1, image2, mask)
    print('คำนวณ SNR map เสร็จแล้ว')
    return {'snr_map': snr_map, 'mask': mask}


def run_sr(volumes, crop, ia_result, gd_result, ia_threshold, gd_threshold=1, num_workers=None):
    """ Spatial resolution map from the structured metal scan and the high-resolution plastic reference. """
    if num_workers is None:
        num_workers = os.cpu_count()
    ia_mask = get_artifact_mask(ia_result['ia_map'], ia_threshold)
    gd_masks = [get_artifact_mask(gd_result['gd_map'][..., i], gd_threshold) for i in range(3)]
    mask = get_signal_mask(ia_result['implant_mask'], artifact_masks=[ia_mask] + gd_masks)
    target = volumes['structured-metal'].data
    reference = np.abs(sp.ifft(sp.resize(sp.fft(volumes['structured-plastic-reference'].data), target.shape)))
    images = equalize([reference, target])
    slc = to_slices(crop, target.shape)
    reference, target = images[0][slc], images[1][slc]
    resolution_mm = volumes['structured-metal'].meta.resolution_mm
    _, fwhm = sr.get_map(reference, target, (5, 5, 1), (14, 14, 10), resolution_mm, mask, 1, num_workers=num_workers)
    print('คำนวณ SR map เสร็จแล้ว')
    return {'fwhm': fwhm, 'mask': mask}


def _center(arr, axis):
    return arr.shape[axis] // 2


def show_ia(result, index=None, axis=2, save_folder=None):
    """ Show plastic & metal images with the IA map for one slice. """
    index = _center(result['ia_map'], axis) if index is None else index
    take = lambda arr: np.take(arr, index, axis=axis)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(take(result['plastic']), cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Plastic')
    axes[1].imshow(take(result['metal']), cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Metal')
    ia.plot_map(axes[2], take(result['ia_map']), take(result['implant_mask']))
    axes[2].set_title('Intensity artifact')
    _finish(fig, axes, '{} = {}'.format(AXIS_NAMES[axis], index), save_folder, 'ia_{}_{}.png'.format(axis, index))


def show_gd(result, index=None, axis=2, save_folder=None):
    """ Show displacement maps (pixels) along each axis for one slice. """
    index = _center(result['mask'], axis) if index is None else index
    mask = np.take(result['mask'], index, axis=axis)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        gd.plot_map(axes[i], np.take(result['gd_map'][..., i], index, axis=axis) * mask, mask, show_cbar=(i == 2))
        axes[i].set_title('Displacement along {}'.format(AXIS_NAMES[i]))
    _finish(fig, axes, '{} = {}'.format(AXIS_NAMES[axis], index), save_folder, 'gd_{}_{}.png'.format(axis, index))


def show_snr(result, index=None, axis=2, save_folder=None, ticks=(0, 50, 100)):
    """ Show the SNR map for one slice. """
    index = _center(result['snr_map'], axis) if index is None else index
    fig, ax = plt.subplots(figsize=(6, 5))
    snr.plot_map(ax, np.take(result['snr_map'], index, axis=axis), np.take(result['mask'], index, axis=axis), ticks=list(ticks))
    ax.set_title('SNR')
    _finish(fig, [ax], '{} = {}'.format(AXIS_NAMES[axis], index), save_folder, 'snr_{}_{}.png'.format(axis, index))


def show_sr(result, resolution_mm, index=None, axis=2, save_folder=None):
    """Show FWHM maps (mm) along the two in-plane axes for one slice.

    Zero FWHM means not measurable and is hidden. By default the slice with the most measurable voxels is shown.
    """
    if index is None:
        other_axes = tuple(a for a in range(3) if a != axis)
        index = int(np.argmax(np.sum(result['fwhm'][..., 0] != 0, axis=other_axes)))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for i in range(2):
        fwhm = np.take(result['fwhm'][..., i], index, axis=axis)
        sr.plot_map(axes[i], fwhm, fwhm != 0,
                    vmin=resolution_mm[i], vmax=3 * resolution_mm[i], show_cbar=(i == 1))
        axes[i].set_title('FWHM along {}'.format(AXIS_NAMES[i]))
    _finish(fig, axes, '{} = {}'.format(AXIS_NAMES[axis], index), save_folder, 'sr_{}_{}.png'.format(axis, index))


def summarize(resolution_mm, ia_result, ia_threshold, gd_result=None, snr_result=None, sr_result=None, save_folder=None):
    """ Print & save summary numbers for each completed analysis. """
    voxel_mL = np.prod(resolution_mm) / 1000
    region = ia_result['implant_mask']
    rows = [
        ('IA', 'signal loss volume (IA < -{:.0f}%)'.format(ia_threshold * 100), np.sum((ia_result['ia_map'] < -ia_threshold) & region) * voxel_mL, 'mL'),
        ('IA', 'pile-up volume (IA > +{:.0f}%)'.format(ia_threshold * 100), np.sum((ia_result['ia_map'] > ia_threshold) & region) * voxel_mL, 'mL'),
        ('IA', 'excluded volume (implant, margin, dark areas)', np.sum(~region) * voxel_mL, 'mL'),
    ]
    if gd_result is not None:
        displacement_mm = gd_result['gd_map'] * np.array(resolution_mm)
        valid = gd_result['mask']
        for i in range(3):
            rows.append(('GD', 'max |displacement| along {}'.format(AXIS_NAMES[i]), np.max(np.abs(displacement_mm[..., i][valid])), 'mm'))
        magnitude = np.linalg.norm(displacement_mm, axis=-1)[valid]
        rows.append(('GD', '95th percentile displacement magnitude', np.percentile(magnitude, 95), 'mm'))
    if snr_result is not None:
        rows.append(('SNR', 'median SNR in valid region', np.median(snr_result['snr_map'][snr_result['mask']]), ''))
    if sr_result is not None:
        for i in range(2):
            fwhm = sr_result['fwhm'][..., i][sr_result['mask']]
            fwhm = fwhm[fwhm > 0]
            if fwhm.size == 0:
                print('คำเตือน: ไม่มีบริเวณที่วัด SR ได้ (กรอบ crop อาจเล็กเกินไป หรือมี artifact ทั้งกรอบ)')
                break
            rows.append(('SR', 'median FWHM along {}'.format(AXIS_NAMES[i]), np.median(fwhm), 'mm'))
    print('{:<6}{:<45}{:>10}'.format('ชนิด', 'ค่า', 'ผล'))
    for kind, label, value, unit in rows:
        print('{:<6}{:<45}{:>10.2f} {}'.format(kind, label, value, unit))
    if save_folder is not None:
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(save_folder, 'summary.csv'), 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['analysis', 'measure', 'value', 'unit'])
            for kind, label, value, unit in rows:
                writer.writerow([kind, label, round(float(value), 4), unit])
        print('\nบันทึกตารางไว้ที่ {} (เปิดด้วย Excel ได้)'.format(os.path.join(save_folder, 'summary.csv')))
    return rows


def save_maps(save_folder, **results):
    """ Save every array of each result as .npy files, e.g. save_maps(folder, ia=ia_result). """
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    for prefix, result in results.items():
        if result is None:
            continue
        for key, value in result.items():
            np.save(os.path.join(save_folder, '{}-{}.npy'.format(prefix, key)), value)
    print('บันทึกข้อมูล map ไว้ที่ {}'.format(save_folder))


def _finish(fig, axes, title, save_folder, filename):
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    _save(fig, save_folder, filename)
    plt.show()


def _save(fig, save_folder, filename):
    if save_folder is not None:
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        fig.savefig(os.path.join(save_folder, filename), dpi=150, bbox_inches='tight')
