"""Simplified helpers for running the TONIQ analyses step by step in a notebook.

Messages are printed in Thai for the accompanying notebook; plot labels are in English since
matplotlib's default font lacks Thai glyphs.
"""
import csv
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import sigpy as sp
from skimage import filters

from toniq import ia, gd, snr, sr
from toniq.dicom import load_series_from_path, sort_series
from toniq.masks import get_implant_mask, get_signal_mask, get_artifact_mask
from toniq.util import equalize, masked_copy, normalize

AXIS_NAMES = ('axis 0 (rows)', 'axis 1 (columns)', 'axis 2 (slices)')

# chart colors (dark theme), validated for contrast & color vision deficiency on the dark surface
DARK = {'surface': '#1a1a19', 'text': '#f0efec', 'secondary': '#c3c2b7', 'muted': '#898781', 'grid': '#2c2c2a', 'axis': '#383835'}
ZONE_COLORS = ('#86b6ef', '#3987e5', '#1c5cab')  # ordinal blue ramp: near screw -> far
WHOLE_REGION = 'Whole region'
CONDITION_COLORS = ('#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767')


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


def run_sr_baseline(volumes, crop, ia_result, num_workers=None):
    """ Spatial resolution map of the structured plastic scan, used as the no-metal baseline for SR thresholds. """
    if num_workers is None:
        num_workers = os.cpu_count()
    mask = get_signal_mask(ia_result['implant_mask'])
    target = volumes['structured-plastic'].data
    reference = np.abs(sp.ifft(sp.resize(sp.fft(volumes['structured-plastic-reference'].data), target.shape)))
    images = equalize([reference, target])
    slc = to_slices(crop, target.shape)
    reference, target = images[0][slc], images[1][slc]
    resolution_mm = volumes['structured-plastic'].meta.resolution_mm
    _, fwhm = sr.get_map(reference, target, (5, 5, 1), (14, 14, 10), resolution_mm, mask, 1, num_workers=num_workers)
    print('คำนวณ SR baseline (ภาพ plastic) เสร็จแล้ว')
    return {'fwhm': fwhm, 'mask': mask}


def direction_labels(volume):
    """ Name each image axis by its encoding direction (frequency, phase, slice) from the scan metadata. """
    readout = volume.meta.readoutDirection
    if readout is None:
        return AXIS_NAMES
    labels = ['phase (axis {})'.format(i) for i in range(2)] + ['slice (axis 2)']
    labels[readout] = 'frequency (axis {})'.format(readout)
    return tuple(labels)


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


def show_gd(result, index=None, axis=2, save_folder=None, axis_labels=AXIS_NAMES):
    """ Show displacement maps (pixels) along each axis for one slice. """
    index = _center(result['mask'], axis) if index is None else index
    mask = np.take(result['mask'], index, axis=axis)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        gd.plot_map(axes[i], np.take(result['gd_map'][..., i], index, axis=axis) * mask, mask, show_cbar=(i == 2))
        axes[i].set_title('Displacement along {}'.format(axis_labels[i]))
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


def make_zones(volumes, crop, edges_mm=(10, 20)):
    """Divide the analysis region into zones by distance (mm) from the screw surface.

    The screw is taken as the largest dark connected region of the uniform plastic image.
    Zone 1 spans 0 to edges_mm[0], zone 2 spans edges_mm[0] to edges_mm[1], and zone 3 is everything farther.
    """
    (image,) = prepare(volumes, ('uniform-plastic',), crop)
    labels, count = ndi.label(image < filters.threshold_otsu(image))
    if count == 0:
        raise ValueError('ไม่พบสกรูในกรอบ crop กรุณาตรวจกรอบในขั้นที่ 3')
    sizes = ndi.sum(np.ones_like(labels), labels, index=range(1, count + 1))
    screw = ndi.binary_fill_holes(labels == 1 + int(np.argmax(sizes)))
    resolution_mm = volumes['uniform-plastic'].meta.resolution_mm
    distance_mm = ndi.distance_transform_edt(~screw, sampling=resolution_mm)
    zones = 1 + np.digitize(distance_mm, edges_mm)
    zones[screw] = 0
    names = ['Zone 1 (0-{:g} mm)'.format(edges_mm[0]),
             'Zone 2 ({:g}-{:g} mm)'.format(*edges_mm),
             'Zone 3 (>{:g} mm)'.format(edges_mm[1])]
    voxel_mL = np.prod(resolution_mm) / 1000
    print('สร้าง zone เสร็จแล้ว (ปริมาตรสกรูที่พบ {:.2f} mL)'.format(screw.sum() * voxel_mL))
    for i, name in enumerate(names):
        print('  {:<22} {:>9} voxel  ({:.1f} mL)'.format(name, np.sum(zones == i + 1), np.sum(zones == i + 1) * voxel_mL))
    return {'zones': zones, 'distance_mm': distance_mm, 'screw': screw, 'names': names, 'edges_mm': tuple(edges_mm)}


def masked_ssim_map(reference, target, mask, sigma, data_range):
    """SSIM map whose Gaussian-weighted local statistics use only voxels inside mask.

    Each local mean, variance and covariance is a weighted average over the in-mask voxels of the window
    (normalized convolution), so excluded regions such as the screw body do not contribute.
    Without excluded voxels this equals the standard SSIM (skimage / viqa, sample covariance, K1=0.01, K2=0.03).
    Voxels outside mask are set to NaN.
    """
    def smooth(arr):
        return ndi.gaussian_filter(arr, sigma, truncate=3.5)

    weight = mask.astype(np.float64)
    x = reference.astype(np.float64)
    y = target.astype(np.float64)
    norm = smooth(weight)
    with np.errstate(invalid='ignore', divide='ignore'):
        ux = smooth(x * weight) / norm
        uy = smooth(y * weight) / norm
        vx = smooth(x * x * weight) / norm - ux ** 2
        vy = smooth(y * y * weight) / norm - uy ** 2
        vxy = smooth(x * y * weight) / norm - ux * uy
    taps = 2 * int(3.5 * sigma + 0.5) + 1
    cov_norm = taps ** x.ndim / (taps ** x.ndim - 1)  # sample covariance, as in skimage & viqa
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim = ((2 * ux * uy + c1) * (2 * cov_norm * vxy + c2)) / ((ux ** 2 + uy ** 2 + c1) * (cov_norm * (vx + vy) + c2))
    ssim[~mask] = np.nan
    return ssim


def run_ssim(volumes, crop, zone_result, sigma=1.5):
    """3D SSIM map between the structured plastic (reference) and structured metal images, excluding the screw.

    The screw body is dark in both images and would otherwise raise SSIM next to the screw, so it is left out
    of every window (see masked_ssim_map). Voxels within the filter radius of the crop border are marked invalid,
    as in the standard SSIM mean.
    """
    plastic, metal = prepare(volumes, ('structured-plastic', 'structured-metal'), crop)
    data_range = float(plastic.max() - plastic.min())
    outside_screw = ~zone_result['screw']
    ssim_map = masked_ssim_map(plastic, metal, outside_screw, sigma, data_range)
    radius = int(3.5 * sigma + 0.5)
    valid = np.zeros(ssim_map.shape, dtype=bool)
    valid[tuple(slice(radius, n - radius) for n in ssim_map.shape)] = True
    valid &= outside_screw
    print('คำนวณ SSIM map เสร็จแล้ว (ไม่รวมตัวสกรู, sigma = {} voxel, SSIM เฉลี่ยทั้งกรอบ = {:.3f})'.format(sigma, ssim_map[valid].mean()))
    return {'ssim_map': ssim_map, 'valid': valid, 'plastic': plastic, 'metal': metal, 'sigma': sigma}


def ssim_sensitivity(volumes, crop, zone_result, sigmas=(1.0, 1.5, 2.0), save_folder=None):
    """Sensitivity of zonal SSIM to the window size: mean SSIM per zone for each sigma.

    The window size is also given in mm (in-plane voxel size) as the Gaussian FWHM, 2.355 sigma.
    """
    voxel_mm = volumes['structured-plastic'].meta.resolution_mm[0]
    zones = zone_result['zones']
    rows = []
    for sigma in sigmas:
        result = run_ssim(volumes, crop, zone_result, sigma=sigma)
        means = [result['ssim_map'][result['valid'] & (zones == zone)].mean() for zone in (1, 2, 3)]
        rows.append({'sigma_voxel': sigma, 'window_fwhm_mm': round(2.355 * sigma * voxel_mm, 2),
                     **{name: round(float(m), 4) for name, m in zip(zone_result['names'], means)}})
    print('\n{:<14}{:>18}'.format('sigma (voxel)', 'FWHM หน้าต่าง (mm)') + ''.join('{:>22}'.format(n) for n in zone_result['names']))
    for row in rows:
        print('{:<14}{:>18}'.format(row['sigma_voxel'], row['window_fwhm_mm']) + ''.join('{:>22.3f}'.format(row[n]) for n in zone_result['names']))
    spread = [max(r[n] for r in rows) - min(r[n] for r in rows) for n in zone_result['names']]
    print('{:<32}'.format('ช่วงที่ค่าเปลี่ยน (max - min)') + ''.join('{:>22.3f}'.format(v) for v in spread))
    if save_folder is not None:
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(save_folder, 'ssim_sensitivity.csv'), 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=['sigma_voxel', 'window_fwhm_mm'] + zone_result['names'])
            writer.writeheader()
            writer.writerows(rows)
        print('\nบันทึกตารางไว้ที่ {}'.format(os.path.join(save_folder, 'ssim_sensitivity.csv')))
    return rows


def _sr_thresholds(sr_baseline, n_sd=2):
    """ Per-axis FWHM threshold (mm) from the plastic baseline: mean + n_sd standard deviations. """
    thresholds = []
    for i in range(2):
        fwhm = sr_baseline['fwhm'][..., i]
        fwhm = fwhm[fwhm > 0]
        thresholds.append(fwhm.mean() + n_sd * fwhm.std())
    return thresholds


def _region_measures(region, resolution_mm, ia_threshold, ia_result=None, gd_result=None, snr_result=None, sr_result=None,
                     sr_thresholds=None, ssim_result=None, gd_threshold_mm=1, axis_labels=AXIS_NAMES):
    """ Each measure within one region: list of (measure, value, unit, valid voxel mask, detail). """
    voxel_cm3 = np.prod(resolution_mm) / 1000
    out = []
    if ia_result is not None:
        artifact = np.abs(ia_result['ia_map']) > ia_threshold
        out.append(('IA: % voxels with |IA| > threshold', _pct(artifact, region), '%', region, 'threshold {:.0f}%'.format(ia_threshold * 100)))
    if gd_result is not None:
        valid = region & gd_result['mask']
        detail = 'threshold {:g} mm'.format(gd_threshold_mm)
        for i in range(3):
            displacement_mm = np.abs(gd_result['gd_map'][..., i] * resolution_mm[i])
            displaced = valid & (displacement_mm > gd_threshold_mm)
            name = 'GD: volume displaced > threshold along {}'.format(axis_labels[i])
            out.append((name + ' [% of valid]', _pct(displaced, valid), '%', valid, detail))
            out.append((name + ' [voxels]', np.sum(displaced), 'voxels', valid, detail))
            out.append((name + ' [cm3]', np.sum(displaced) * voxel_cm3, 'cm3', valid, detail))
            out.append(('GD: max |displacement| along {}'.format(axis_labels[i]),
                        displacement_mm[valid].max() if valid.any() else None, 'mm', valid, ''))
    if snr_result is not None:
        valid = region & snr_result['mask'] & (snr_result['snr_map'] > 0)
        out.append(('SNR: median', np.median(snr_result['snr_map'][valid]) if valid.any() else None, '', valid, ''))
    if sr_result is not None:
        for i in range(2):
            fwhm = sr_result['fwhm'][..., i]
            valid = region & (fwhm > 0)
            out.append(('SR: median FWHM along {}'.format(axis_labels[i]), np.median(fwhm[valid]) if valid.any() else None, 'mm', valid, ''))
            for label, thresholds in (sr_thresholds or {}).items():
                blurred = valid & (fwhm > thresholds[i])
                detail = 'plastic {} = {:.2f} mm'.format(label, thresholds[i])
                name = 'SR: BV along {} (> plastic {})'.format(axis_labels[i], label)
                out.append((name + ' [% of valid]', _pct(blurred, valid), '%', valid, detail))
                out.append((name + ' [voxels]', np.sum(blurred), 'voxels', valid, detail))
                out.append((name + ' [cm3]', np.sum(blurred) * voxel_cm3, 'cm3', valid, detail))
    if ssim_result is not None:
        valid = region & ssim_result['valid']
        out.append(('SSIM: mean', ssim_result['ssim_map'][valid].mean() if valid.any() else None, '', valid,
                    'screw excluded, sigma {:g} voxel'.format(ssim_result['sigma'])))
    return out


def _all_region_rows(zone_result, by_slice, resolution_mm, ia_threshold, ia_result, gd_result, snr_result, sr_result,
                     sr_baseline, ssim_result, gd_threshold_mm, min_coverage_pct, sr_sd, axis_labels):
    sr_thresholds = {}
    if sr_result is not None and sr_baseline is not None:
        sr_thresholds = {'mean+{:g}SD'.format(k): _sr_thresholds(sr_baseline, n_sd=k) for k in sr_sd}
    zones = zone_result['zones']
    # IA, GD, SNR and SR cover the whole analysis region (crop excluding the screw); only SSIM is split by zone
    groups = [(WHOLE_REGION, zones > 0, dict(ia_result=ia_result, gd_result=gd_result, snr_result=snr_result,
                                              sr_result=sr_result, ssim_result=None))]
    if ssim_result is not None:
        groups += [(name, zones == i + 1, dict(ia_result=None, gd_result=None, snr_result=None, sr_result=None,
                                                ssim_result=ssim_result)) for i, name in enumerate(zone_result['names'])]
    rows = []
    for group_name, in_group, results in groups:
        regions = [(None, in_group)]
        if by_slice:
            regions = []
            for k in range(zones.shape[2]):
                region = np.zeros_like(in_group)
                region[..., k] = in_group[..., k]
                if region.any():
                    regions.append((k, region))
        for k, region in regions:
            size = np.sum(region)
            for measure, value, unit, valid, detail in _region_measures(
                    region, resolution_mm, ia_threshold, sr_thresholds=sr_thresholds, gd_threshold_mm=gd_threshold_mm,
                    axis_labels=axis_labels, **results):
                coverage = 100 * np.sum(valid) / size
                if coverage < min_coverage_pct:
                    value = None
                row = {'measure': measure, 'zone': group_name, 'value': None if value is None else float(value),
                       'unit': unit, 'coverage_pct': round(coverage, 1), 'detail': detail}
                if by_slice:
                    row = {'slice': k, **row}
                rows.append(row)
    return rows


def _write_rows(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, 'value': '' if row['value'] is None else round(row['value'], 4)})


def zonal_summary(zone_result, resolution_mm, ia_threshold, ia_result, gd_result=None, snr_result=None,
                  sr_result=None, sr_baseline=None, ssim_result=None, gd_threshold_mm=1, min_coverage_pct=5,
                  sr_sd=(2, 1), axis_labels=AXIS_NAMES, save_folder=None):
    """Summarize IA, GD, SNR and SR over the whole analysis region, and SSIM within each zone.

    The whole analysis region is the crop excluding the screw; zones (by distance from the screw) apply to SSIM only.
    With save_folder, per-slice values are also saved (slice_summary.csv).
    Coverage is the percentage of the region where the measure is valid; GD, SNR and SR are not measurable
    in regions of severe artifact, so a low coverage means the value describes only part of the zone.
    Values with coverage below min_coverage_pct are reported as unavailable, since too few voxels support them.
    The Blurring Volume (BV) counts valid voxels whose FWHM exceeds the plastic baseline mean + k SD for each k
    in sr_sd (the first is the primary threshold; the others are for sensitivity analysis).
    GD volume counts valid voxels displaced by more than gd_threshold_mm along each direction (frequency, phase, slice).
    axis_labels names the image axes, e.g. from direction_labels(); defaults to array axis names.
    Measure names are kept the same across runs so that conditions can be compared; settings that differ
    between runs (e.g. the SSIM sigma or the SR threshold) are recorded in the 'detail' column.
    """
    args = (resolution_mm, ia_threshold, ia_result, gd_result, snr_result, sr_result, sr_baseline, ssim_result,
            gd_threshold_mm, min_coverage_pct, sr_sd, axis_labels)
    rows = _all_region_rows(zone_result, False, *args)
    _print_zonal(rows, [WHOLE_REGION] + zone_result['names'], min_coverage_pct)
    if save_folder is not None:
        _write_rows(rows, os.path.join(save_folder, 'zonal_summary.csv'))
        _write_rows(_all_region_rows(zone_result, True, *args), os.path.join(save_folder, 'slice_summary.csv'))
        print('\nบันทึกตารางไว้ที่ {} (และค่าราย slice ใน slice_summary.csv)'.format(os.path.join(save_folder, 'zonal_summary.csv')))
    return rows


def _pct(condition, valid):
    return 100 * np.sum(condition & valid) / np.sum(valid) if np.any(valid) else None


def _print_zonal(rows, zone_names, min_coverage_pct):
    measures = list(dict.fromkeys(row['measure'] for row in rows))
    print('{:<72}'.format('ค่า') + ''.join('{:>24}'.format(name) for name in zone_names))
    for measure in measures:
        cells = []
        for name in zone_names:
            row = next((r for r in rows if r['measure'] == measure and r['zone'] == name), None)
            if row is None:
                cells.append('{:>24}'.format('-'))
                continue
            value = 'n/a' if row['value'] is None else '{:.2f} {}'.format(row['value'], row['unit']).strip()
            cells.append('{:>24}'.format('{} [{:.1f}%]'.format(value, row['coverage_pct'])))
        detail = next(r['detail'] for r in rows if r['measure'] == measure)
        print('{:<72}'.format(measure[:71]) + ''.join(cells) + ('   ({})'.format(detail) if detail else ''))
    print('\n[ตัวเลขในวงเล็บ] = ร้อยละของพื้นที่ที่วัดค่าได้ (coverage); n/a = coverage ต่ำกว่า {:g}% จึงไม่รายงานค่า; - = ไม่ได้คำนวณแยก zone'.format(min_coverage_pct))


def show_zones(zone_result, ssim_result, ia_result, index=None, axis=2, save_folder=None):
    """ Show structured plastic & metal images, their difference, the SSIM map and the IA map with zone boundaries. """
    distance = zone_result['distance_mm']
    if index is None:
        index = int(np.argmax(np.sum(zone_result['screw'], axis=tuple(a for a in range(3) if a != axis))))
    take = lambda arr: np.take(arr, index, axis=axis)
    panels = [
        ('Structured plastic', take(ssim_result['plastic']), dict(cmap='gray', vmin=0, vmax=1)),
        ('Structured metal', take(ssim_result['metal']), dict(cmap='gray', vmin=0, vmax=1)),
        ('|Metal - plastic|', take(np.abs(ssim_result['metal'] - ssim_result['plastic'])), dict(cmap='gray', vmin=0, vmax=0.5)),
        ('SSIM, screw excluded (sigma {:g})'.format(ssim_result['sigma']), take(np.where(ssim_result['valid'], ssim_result['ssim_map'], np.nan)),
         dict(cmap='Blues_r', vmin=0, vmax=1)),
        ('Intensity artifact', take(ia_result['ia_map']), dict(cmap='RdBu_r', vmin=-0.8, vmax=0.8)),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    for ax, (title, image, kwargs) in zip(axes, panels):
        im = ax.imshow(image, **kwargs)
        ax.contour(take(zone_result['screw']).astype(float), levels=[0.5], colors='#f0efec', linewidths=1.5)
        ax.contour(take(distance), levels=list(zone_result['edges_mm']), colors=['#eda100', '#e87ba4'], linewidths=1.5)
        ax.set_title(title)
        if title.startswith('SSIM') or title.startswith('Intensity'):
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.text(0.5, 0.02, 'white = screw surface,  yellow = {:g} mm,  pink = {:g} mm from screw surface'.format(*zone_result['edges_mm']),
             ha='center')
    _finish(fig, axes, '{} = {}'.format(AXIS_NAMES[axis], index), save_folder, 'zones_{}_{}.png'.format(axis, index))


def _dark_axes(ax):
    ax.set_facecolor(DARK['surface'])
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(DARK['axis'])
    ax.tick_params(colors=DARK['muted'], length=0)
    ax.yaxis.grid(True, color=DARK['grid'], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.title.set_color(DARK['text'])


def _bar_panels(measures):
    """ Pick measures to plot: percentages share a 0-100 scale; SSIM gets its own 0-1 panel. """
    # SR: only the primary (first listed) percentile threshold, as % of valid voxels
    sr_primary = next((re.search(r'\(> plastic [^)]*\)', m).group() for m in measures if m.startswith('SR: BV')), None)
    return [m for m in measures if m.startswith(('IA:', 'SSIM'))
            or (m.startswith('GD: volume') and m.endswith('[% of valid]'))
            or (sr_primary is not None and m.startswith('SR: BV') and sr_primary in m and m.endswith('[% of valid]'))]


def _wrap(text, width=34):
    words, lines, line = text.split(), [], ''
    for word in words:
        if len(line) + len(word) + 1 > width and line:
            lines.append(line)
            line = word
        else:
            line = (line + ' ' + word).strip()
    return '\n'.join(lines + [line])


def plot_zonal(rows, zone_names, save_folder=None):
    """ Dark-themed bar charts of each percentage measure and SSIM by zone (one panel per measure). """
    measures = _bar_panels(list(dict.fromkeys(row['measure'] for row in rows)))
    widths = [1.6 + 0.9 * len([r for r in rows if r['measure'] == m]) for m in measures]
    fig, axes = plt.subplots(1, len(measures), figsize=(sum(widths), 4.2), squeeze=False, gridspec_kw={'width_ratios': widths})
    fig.patch.set_facecolor(DARK['surface'])
    for ax, measure in zip(axes[0], measures):
        groups = [r['zone'] for r in rows if r['measure'] == measure]
        values = [next(r['value'] for r in rows if r['measure'] == measure and r['zone'] == name) for name in groups]
        heights = [0 if v is None else v for v in values]
        colors = [ZONE_COLORS[zone_names.index(g)] if g in zone_names else CONDITION_COLORS[0] for g in groups]
        bars = ax.bar(range(len(groups)), heights, width=0.62, color=colors, edgecolor=DARK['surface'], linewidth=2)
        for bar, value in zip(bars, values):
            label = 'n/a' if value is None else ('{:.2f}'.format(value) if measure.startswith('SSIM') else '{:.0f}%'.format(value))
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha='center', va='bottom', color=DARK['secondary'], fontsize=9)
        ax.set_ylim(0, 1.05 if measure.startswith('SSIM') else 105)
        ax.set_xlim(-0.8, len(groups) - 0.2)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([name.replace(' (', '\n(') for name in groups], color=DARK['secondary'], fontsize=8)
        ax.set_title(_wrap(measure), fontsize=9)
        _dark_axes(ax)
    fig.tight_layout()
    _save(fig, save_folder, 'zonal_summary.png')
    plt.show()


def compare_conditions(conditions, save_folder=None):
    """Compare zonal summaries from several result folders, e.g. {'Parallel to B0': folder1, 'Perpendicular': folder2}.

    Saves a combined CSV and dark-themed grouped bar charts (one panel per measure, bars grouped by zone).
    """
    if len(conditions) > len(CONDITION_COLORS):
        raise ValueError('เปรียบเทียบได้ไม่เกิน {} เงื่อนไขในกราฟเดียว'.format(len(CONDITION_COLORS)))
    tables = {}
    for name, folder in conditions.items():
        with open(os.path.join(folder, 'zonal_summary.csv'), encoding='utf-8-sig') as f:
            tables[name] = list(csv.DictReader(f))
        print('โหลด {} จาก {}'.format(name, folder))
    all_rows = [row for table in tables.values() for row in table]
    zone_names = list(dict.fromkeys(row['zone'] for row in all_rows))
    all_measures = list(dict.fromkeys(row['measure'] for row in all_rows))
    measures = _bar_panels(all_measures)

    def value(name, measure, zone):
        row = next((r for r in tables[name] if r['measure'] == measure and r['zone'] == zone), None)
        return None if row is None or row['value'] == '' else float(row['value'])

    widths = [1.6 + 1.4 * len({r['zone'] for r in all_rows if r['measure'] == m}) for m in measures]
    fig, axes = plt.subplots(1, len(measures), figsize=(sum(widths), 4.6), squeeze=False, gridspec_kw={'width_ratios': widths})
    fig.patch.set_facecolor(DARK['surface'])
    width = 0.8 / len(conditions)
    for ax, measure in zip(axes[0], measures):
        groups = list(dict.fromkeys(r['zone'] for r in all_rows if r['measure'] == measure))
        for j, name in enumerate(conditions):
            values = [value(name, measure, zone) for zone in groups]
            positions = np.arange(len(groups)) - 0.4 + width * (j + 0.5)
            ax.bar(positions, [0 if v is None else v for v in values], width=width, color=CONDITION_COLORS[j],
                   edgecolor=DARK['surface'], linewidth=2, label=name)
            for position, v in zip(positions, values):
                if v is None:
                    ax.text(position, 0, 'n/a', ha='center', va='bottom', color=DARK['muted'], fontsize=7)
        ax.set_ylim(0, 1.05 if measure.startswith('SSIM') else 105)
        ax.set_xlim(-0.6, len(groups) - 0.4)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([zone.replace(' (', '\n(') for zone in groups], color=DARK['secondary'], fontsize=8)
        ax.set_title(_wrap(measure), fontsize=9)
        _dark_axes(ax)
    legend = fig.legend(*axes[0][0].get_legend_handles_labels(), loc='upper center', ncol=len(conditions), frameon=False,
                        bbox_to_anchor=(0.5, 1.02))
    for text in legend.get_texts():
        text.set_color(DARK['text'])
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, save_folder, 'comparison.png')
    plt.show()

    if save_folder is not None:
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(save_folder, 'comparison.csv'), 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['measure', 'zone'] + list(conditions))
            for measure in all_measures:
                for zone in dict.fromkeys(r['zone'] for r in all_rows if r['measure'] == measure):
                    writer.writerow([measure, zone] + ['' if value(name, measure, zone) is None else value(name, measure, zone) for name in conditions])
        print('บันทึกตารางเปรียบเทียบไว้ที่ {}'.format(os.path.join(save_folder, 'comparison.csv')))


def _holm(p_values):
    """ Holm-adjusted p-values (family-wise error control). """
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def _read_csv(path):
    with open(path, encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def _test_group(table, names):
    """ Friedman test across protocols (columns) over blocks (rows), then pairwise Wilcoxon signed-rank with Holm. """
    from scipy.stats import friedmanchisquare, wilcoxon
    data = np.array(table, dtype=float)
    n, k = data.shape
    result = {'n_blocks': n, 'chi2': None, 'p': None, 'kendall_w': None}
    pairs = []
    if n < 3 or k < 3:
        return result, pairs
    if np.all(data == data[:, :1]):
        result.update(chi2=0.0, p=1.0, kendall_w=0.0)
    else:
        chi2, p = friedmanchisquare(*data.T)
        result.update(chi2=float(chi2), p=float(p), kendall_w=float(chi2 / (n * (k - 1))))
    raw = []
    for a in range(k):
        for b in range(a + 1, k):
            diff = data[:, a] - data[:, b]
            p_pair = 1.0 if np.all(diff == 0) else float(wilcoxon(data[:, a], data[:, b]).pvalue)
            raw.append(p_pair)
            pairs.append({'protocol_a': names[a], 'protocol_b': names[b], 'median_difference': float(np.median(diff)), 'p_raw': p_pair})
    for pair, p_holm in zip(pairs, _holm(np.array(raw))):
        pair['p_holm'] = float(p_holm)
        pair['significant'] = bool(p_holm < 0.05)
    return result, pairs


def compare_statistics(protocols, design='repeat_zone', save_folder=None):
    """Statistical comparison of protocols with the Friedman test and post-hoc Wilcoxon signed-rank (Holm).

    design 'repeat_zone': protocols maps each protocol name to a list of result folders, one per repeat, in the
        same repeat order for every protocol; blocks are repeat x zone (e.g. 3 repeats x 3 zones = 9 blocks).
    design 'zone_slice': protocols maps each protocol name to one result folder from a single phantom placement;
        blocks are slices, tested separately within each zone. Adjacent slices are correlated, so p-values are optimistic.
    Saves stats_friedman.csv, stats_posthoc.csv and stats_descriptive.csv.
    """
    names = list(protocols)
    tests, posthoc, descriptive = [], [], []
    if design == 'repeat_zone':
        repeats = {name: [_read_csv(os.path.join(folder, 'zonal_summary.csv')) for folder in folders] for name, folders in protocols.items()}
        if len({len(r) for r in repeats.values()}) != 1:
            raise ValueError('ทุก protocol ต้องมีจำนวนรอบเท่ากัน')
        lookup = {(name, i, row['measure'], row['zone']): row['value']
                  for name in names for i, rows in enumerate(repeats[name]) for row in rows}
        first = repeats[names[0]][0]
        groups = {}
        for row in first:
            for i in range(len(repeats[names[0]])):
                groups.setdefault(row['measure'], []).append((i, row['zone']))
        for measure, blocks in groups.items():
            label = 'repeat x zone' if len({z for _, z in blocks}) > 1 else 'repeat'
            if measure.endswith('[voxels]'):
                continue
            table = []
            for i, zone in blocks:
                values = [lookup.get((name, i, measure, zone), '') for name in names]
                if all(v not in ('', None) for v in values):
                    table.append([float(v) for v in values])
            result, pairs = _test_group(table, names)
            tests.append({'measure': measure, 'group': label, **result})
            posthoc += [{'measure': measure, 'group': label, **pair} for pair in pairs]
            for zone in dict.fromkeys(z for _, z in blocks):
                for name in names:
                    values = [lookup.get((name, i, measure, zone), '') for i in range(len(repeats[name]))]
                    values = [float(v) for v in values if v not in ('', None)]
                    descriptive.append({'measure': measure, 'zone': zone, 'protocol': name, 'n': len(values),
                                        'mean': np.mean(values) if values else None, 'sd': np.std(values, ddof=1) if len(values) > 1 else None})
    elif design == 'zone_slice':
        slices = {name: _read_csv(os.path.join(folder, 'slice_summary.csv')) for name, folder in protocols.items()}
        lookup = {(name, row['measure'], row['zone'], row['slice']): row['value'] for name in names for row in slices[name]}
        groups = {}
        for row in slices[names[0]]:
            groups.setdefault((row['measure'], row['zone']), []).append(row['slice'])
        for (measure, zone), slice_ids in groups.items():
            if measure.endswith('[voxels]'):
                continue
            table = []
            for k in slice_ids:
                values = [lookup.get((name, measure, zone, k), '') for name in names]
                if all(v not in ('', None) for v in values):
                    table.append([float(v) for v in values])
            result, pairs = _test_group(table, names)
            tests.append({'measure': measure, 'group': zone, **result})
            posthoc += [{'measure': measure, 'group': zone, **pair} for pair in pairs]
            column = np.array(table) if table else np.zeros((0, len(names)))
            for j, name in enumerate(names):
                descriptive.append({'measure': measure, 'zone': zone, 'protocol': name, 'n': len(column),
                                    'mean': column[:, j].mean() if len(column) else None,
                                    'sd': column[:, j].std(ddof=1) if len(column) > 1 else None})
        print('คำเตือน: slice ที่อยู่ติดกันมีค่าสัมพันธ์กัน ค่า p จากแบบ zone x slice จึงต่ำกว่าความเป็นจริง ต้องรายงานเป็นข้อจำกัด\n')
    else:
        raise ValueError("design ต้องเป็น 'repeat_zone' หรือ 'zone_slice'")

    shown = _bar_panels(list(dict.fromkeys(t['measure'] for t in tests)))
    print('{:<66}{:<20}{:>4}{:>9}{:>10}{:>8}'.format('ค่า', 'กลุ่ม', 'n', 'chi2', 'p', 'W'))
    for t in tests:
        if t['measure'] in shown:
            if t['p'] is None:
                print('{:<66}{:<20}{:>4}   ทดสอบไม่ได้ (ต้องมีอย่างน้อย 3 block และ 3 protocol)'.format(t['measure'][:65], t['group'][:19], t['n_blocks']))
            else:
                print('{:<66}{:<20}{:>4}{:>9.2f}{:>10.2g}{:>8.2f}'.format(t['measure'][:65], t['group'][:19], t['n_blocks'], t['chi2'], t['p'], t['kendall_w']))
    print('\nแสดงเฉพาะค่าหลัก ผลครบทุกค่าและ post-hoc อยู่ในไฟล์ CSV;  W = Kendall\'s W (0 = ไม่สม่ำเสมอ, 1 = อันดับเหมือนกันทุก block)')
    if save_folder is not None:
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        for filename, rows in (('stats_friedman.csv', tests), ('stats_posthoc.csv', posthoc), ('stats_descriptive.csv', descriptive)):
            if rows:
                with open(os.path.join(save_folder, filename), 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)
        print('บันทึกไว้ที่ {} (stats_friedman.csv, stats_posthoc.csv, stats_descriptive.csv)'.format(save_folder))
    return tests, posthoc, descriptive
