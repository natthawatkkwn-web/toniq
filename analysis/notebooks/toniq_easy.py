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


def make_zones(volumes, crop, edges_mm=(5, 15)):
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


def _sr_thresholds(sr_baseline, percentile=75):
    """ Per-axis FWHM threshold (mm) from the plastic baseline. """
    thresholds = []
    for i in range(2):
        fwhm = sr_baseline['fwhm'][..., i]
        thresholds.append(np.percentile(fwhm[fwhm > 0], percentile))
    return thresholds


def zonal_summary(zone_result, resolution_mm, ia_threshold, ia_result, gd_result=None, snr_result=None,
                  sr_result=None, sr_baseline=None, ssim_result=None, gd_threshold_px=1, min_coverage_pct=5,
                  sr_percentiles=(95, 75), save_folder=None):
    """Summarize each analysis within each zone.

    Coverage is the percentage of the zone where the measure is valid; GD, SNR and SR are not measurable
    in regions of severe artifact, so a low coverage means the value describes only part of the zone.
    Values with coverage below min_coverage_pct are reported as unavailable, since too few voxels support them.
    The Blurring Volume (BV) counts valid voxels whose FWHM exceeds a percentile of the plastic baseline FWHM,
    for each of sr_percentiles (the first is the primary threshold; the others are for sensitivity analysis).
    Measure names are kept the same across runs so that conditions can be compared; settings that differ
    between runs (e.g. the SSIM sigma or the SR threshold) are recorded in the 'detail' column.
    """
    zones = zone_result['zones']
    rows = []

    def add(measure, zone, value, unit, valid, detail=''):
        zone_size = np.sum(zones == zone)
        coverage = 100 * np.sum(valid) / zone_size if zone_size > 0 else 0
        if coverage < min_coverage_pct:
            value = None
        rows.append({'measure': measure, 'zone': zone_result['names'][zone - 1], 'value': float(value) if value is not None else None,
                     'unit': unit, 'coverage_pct': round(coverage, 1), 'detail': detail})

    sr_thresholds = {}
    if sr_result is not None and sr_baseline is not None:
        sr_thresholds = {p: _sr_thresholds(sr_baseline, percentile=p) for p in sr_percentiles}
    voxel_cm3 = np.prod(resolution_mm) / 1000
    for zone in (1, 2, 3):
        in_zone = zones == zone
        artifact = np.abs(ia_result['ia_map']) > ia_threshold
        add('IA: % voxels with |IA| > threshold', zone, _pct(artifact, in_zone), '%', in_zone, 'threshold {:.0f}%'.format(ia_threshold * 100))
        if gd_result is not None:
            valid = in_zone & gd_result['mask']
            displacement_px = np.linalg.norm(gd_result['gd_map'], axis=-1)
            displacement_mm = np.linalg.norm(gd_result['gd_map'] * np.array(resolution_mm), axis=-1)
            add('GD: % voxels displaced > threshold', zone, _pct(displacement_px > gd_threshold_px, valid), '%', valid,
                'threshold {:g} px'.format(gd_threshold_px))
            add('GD: max displacement', zone, displacement_mm[valid].max() if valid.any() else None, 'mm', valid)
        if snr_result is not None:
            valid = in_zone & snr_result['mask'] & (snr_result['snr_map'] > 0)
            add('SNR: median', zone, np.median(snr_result['snr_map'][valid]) if valid.any() else None, '', valid)
        if sr_result is not None:
            for i in range(2):
                fwhm = sr_result['fwhm'][..., i]
                valid = in_zone & (fwhm > 0)
                add('SR: median FWHM along {}'.format(AXIS_NAMES[i]), zone, np.median(fwhm[valid]) if valid.any() else None, 'mm', valid)
                for p, thresholds in sr_thresholds.items():
                    blurred = valid & (fwhm > thresholds[i])
                    detail = 'plastic P{} = {:.2f} mm'.format(p, thresholds[i])
                    name = 'SR: BV along {} (> plastic P{})'.format(AXIS_NAMES[i], p)
                    add(name + ' [% of valid]', zone, _pct(blurred, valid), '%', valid, detail)
                    add(name + ' [voxels]', zone, np.sum(blurred), 'voxels', valid, detail)
                    add(name + ' [cm3]', zone, np.sum(blurred) * voxel_cm3, 'cm3', valid, detail)
        if ssim_result is not None:
            valid = in_zone & ssim_result['valid']
            add('SSIM: mean', zone, ssim_result['ssim_map'][valid].mean() if valid.any() else None, '', valid,
                'screw excluded, sigma {:g} voxel'.format(ssim_result['sigma']))

    _print_zonal(rows, zone_result['names'], min_coverage_pct)
    if save_folder is not None:
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(save_folder, 'zonal_summary.csv'), 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=['measure', 'zone', 'value', 'unit', 'coverage_pct', 'detail'])
            writer.writeheader()
            for row in rows:
                writer.writerow({**row, 'value': '' if row['value'] is None else round(row['value'], 4)})
        print('\nบันทึกตารางไว้ที่ {}'.format(os.path.join(save_folder, 'zonal_summary.csv')))
    return rows


def _pct(condition, valid):
    return 100 * np.sum(condition & valid) / np.sum(valid) if np.any(valid) else None


def _print_zonal(rows, zone_names, min_coverage_pct):
    measures = list(dict.fromkeys(row['measure'] for row in rows))
    print('{:<62}'.format('ค่า') + ''.join('{:>24}'.format(name) for name in zone_names))
    for measure in measures:
        cells = []
        for name in zone_names:
            row = next(r for r in rows if r['measure'] == measure and r['zone'] == name)
            value = 'n/a' if row['value'] is None else '{:.2f} {}'.format(row['value'], row['unit']).strip()
            cells.append('{:>24}'.format('{} [{:.1f}%]'.format(value, row['coverage_pct'])))
        detail = next(r['detail'] for r in rows if r['measure'] == measure)
        print('{:<62}'.format(measure[:61]) + ''.join(cells) + ('   ({})'.format(detail) if detail else ''))
    print('\n[ตัวเลขในวงเล็บ] = ร้อยละของ zone ที่วัดค่าได้ (coverage); n/a = coverage ต่ำกว่า {:g}% จึงไม่รายงานค่า'.format(min_coverage_pct))


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
    sr_primary = next((re.search(r'plastic P\d+', m).group() for m in measures if m.startswith('SR: BV')), None)
    return [m for m in measures if m.startswith(('IA:', 'GD: %', 'SSIM'))
            or (sr_primary is not None and m.startswith('SR: BV') and sr_primary + ')' in m and m.endswith('[% of valid]'))]


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
    fig, axes = plt.subplots(1, len(measures), figsize=(3.6 * len(measures), 4.2), squeeze=False)
    fig.patch.set_facecolor(DARK['surface'])
    for ax, measure in zip(axes[0], measures):
        values = [next(r['value'] for r in rows if r['measure'] == measure and r['zone'] == name) for name in zone_names]
        heights = [0 if v is None else v for v in values]
        bars = ax.bar(range(3), heights, width=0.62, color=ZONE_COLORS, edgecolor=DARK['surface'], linewidth=2)
        for bar, value in zip(bars, values):
            label = 'n/a' if value is None else ('{:.2f}'.format(value) if measure.startswith('SSIM') else '{:.0f}%'.format(value))
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha='center', va='bottom', color=DARK['secondary'], fontsize=9)
        ax.set_ylim(0, 1.05 if measure.startswith('SSIM') else 105)
        ax.set_xticks(range(3))
        ax.set_xticklabels([name.replace(' (', '\n(') for name in zone_names], color=DARK['secondary'], fontsize=8)
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

    fig, axes = plt.subplots(1, len(measures), figsize=(4.2 * len(measures), 4.6), squeeze=False)
    fig.patch.set_facecolor(DARK['surface'])
    width = 0.8 / len(conditions)
    for ax, measure in zip(axes[0], measures):
        for j, name in enumerate(conditions):
            values = [value(name, measure, zone) for zone in zone_names]
            positions = np.arange(len(zone_names)) - 0.4 + width * (j + 0.5)
            ax.bar(positions, [0 if v is None else v for v in values], width=width, color=CONDITION_COLORS[j],
                   edgecolor=DARK['surface'], linewidth=2, label=name)
            for position, v in zip(positions, values):
                if v is None:
                    ax.text(position, 0, 'n/a', ha='center', va='bottom', color=DARK['muted'], fontsize=7)
        ax.set_ylim(0, 1.05 if measure.startswith('SSIM') else 105)
        ax.set_xticks(range(len(zone_names)))
        ax.set_xticklabels([zone.replace(' (', '\n(') for zone in zone_names], color=DARK['secondary'], fontsize=8)
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
                for zone in zone_names:
                    writer.writerow([measure, zone] + ['' if value(name, measure, zone) is None else value(name, measure, zone) for name in conditions])
        print('บันทึกตารางเปรียบเทียบไว้ที่ {}'.format(os.path.join(save_folder, 'comparison.csv')))
