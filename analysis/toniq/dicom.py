"""Functions for loading & parsing DICOM data.

"""
import numpy.typing as npt
import numpy as np

from pathlib import Path
from pydicom import dcmread
from pydicom.dataset import Dataset
from pydicom.errors import InvalidDicomError
from os import path

from toniq.data import Metadata, ImageVolume

MR_IMAGE_STORAGE = '1.2.840.10008.5.1.4.1.1.4'
ENHANCED_MR_IMAGE_STORAGE = '1.2.840.10008.5.1.4.1.1.4.1'
PHILIPS_SCANNING_TECHNIQUE = (0x2001, 0x1020)

def load_series(files: list[str], dtype=float) -> ImageVolume:
    """ Load DICOM series a list of files. """
    data = []
    slice_indices = []
    meta = None
    for f in files:
        slice_data = read_data(f)
        slice_meta, slice_index = read_meta(f)
        if meta is None:
            meta = slice_meta
        elif meta != slice_meta:
            raise ValueError('Metadata disagreement between slices')
        data.append(slice_data)
        slice_indices.append(slice_index)
    order = np.argsort(slice_indices)
    data = np.stack([data[i] for i in order], axis=-1)
    data = data.astype(dtype)
    return ImageVolume(data, meta)

def load_series_from_path(series_path: str) -> ImageVolume:
    """ Load DICOM series located on a given path.

    Files named in the GE export convention (*MRDC*) are loaded with GE-specific parsing.
    Otherwise all files in the folder are loaded with standard DICOM parsing (e.g. for Philips).
    """
    files = list(Path(series_path).glob('*MRDC*'))
    if len(files) > 0:
        image = load_series(files)
    else:
        files = sorted(f for f in Path(series_path).iterdir() if f.is_file())
        image = load_series_standard(files)
    return image

def read_data(file: str) -> npt.NDArray:
    """ Read image data from DICOM file. """
    return dcmread(file).pixel_array

def read_meta(file: str) -> tuple[Metadata, int]:
    """ Read metadata from DICOM file. """
    dicom = dcmread(file)
    acqMatrix = np.array(dicom.AcquisitionMatrix)
    inplaneMatrixSize = acqMatrix[np.nonzero(acqMatrix)]

    meta_dict = {
        'date_YYYYMMDD': dicom.StudyDate,
        'scanner': dicom.ManufacturerModelName,
        'staticFieldStrength_T': float(dicom.MagneticFieldStrength),

        'seriesName': dicom.SeriesDescription,
        'dimensionality': dicom.MRAcquisitionType,
        'pulseSequenceName': dicom[0x0019, 0x109c].value,
        'duration_s': np.round(dicom[0x0019, 0x105a].value * 1e-6),

        'acqMatrixShape': tuple(inplaneMatrixSize) + (int(dicom.ImagesInAcquisition),),
        'resolution_mm': tuple(map(float, dicom.PixelSpacing)) + (float(dicom.SliceThickness),),
        'refocusFlipAngle_deg': float(dicom.FlipAngle),
        'echoTrainLength': int(dicom.EchoTrainLength),
        'echoTime_ms': float(dicom.EchoTime),
        'repetitionTime_ms': float(dicom.RepetitionTime),
        'centerFrequency_Hz': float(dicom.ImagingFrequency * 1e6),
        'pixelBandwidth_Hz': float(dicom.PixelBandwidth),
        'readoutDirection': 0 if dicom.InPlanePhaseEncodingDirection == 'ROW' else 1,

        'containsMetal': 'PLA' not in dicom.SeriesDescription
    }
    
    if meta_dict['dimensionality'] == '2D' and dicom.SpacingBetweenSlices != meta_dict['resolution_mm'][-1]:
        raise ValueError('Multi-slice volume has gaps between slices')
    meta = Metadata(**meta_dict)
    try:
        sliceIndex = dicom.InStackPositionNumber
    except:
        _, sliceIndex = path.splitext(file)
        sliceIndex = int(sliceIndex[1:])
    return meta, sliceIndex

def load_series_standard(files: list[str], dtype=float) -> ImageVolume:
    """Load DICOM series from a list of files using standard DICOM attributes (e.g. Philips).

    Supports classic MR storage (one file per slice) and enhanced MR storage (multi-frame files).
    Files that are not MR images (e.g. DICOMDIR) are skipped.
    Slices are ordered by position along the slice normal, and pixel values are rescaled to real-world values.

    Args:
        files (list[str]): files of the series
        dtype (optional): data type of the returned volume. Defaults to float.

    Returns:
        ImageVolume: image volume
    """
    data = []
    positions = []
    meta_dict = None
    for f in files:
        dicom = read_mr_image(f)
        if dicom is None:
            continue
        if dicom.SOPClassUID == ENHANCED_MR_IMAGE_STORAGE:
            file_data, file_positions = read_frames_enhanced(dicom)
        else:
            file_data, file_positions = read_frames_classic(dicom)
        file_meta_dict = read_meta_standard(dicom)
        if meta_dict is None:
            meta_dict = file_meta_dict
        elif meta_dict != file_meta_dict:
            raise ValueError('Metadata disagreement between slices')
        data += file_data
        positions += file_positions
    if meta_dict is None:
        raise ValueError('No MR image files found in series')
    order = np.argsort(positions)
    positions = np.array(positions)[order]
    spacing = np.diff(positions)
    if np.any(spacing < 1e-3):
        raise ValueError('Multiple images at the same slice position; export one image type (e.g. magnitude) and one echo per series')
    if len(spacing) > 0 and not np.allclose(spacing, spacing[0], atol=1e-2):
        raise ValueError('Slices are not evenly spaced')
    thickness = meta_dict.pop('sliceThickness_mm')
    slice_spacing = float(np.round(spacing[0], decimals=3)) if len(spacing) > 0 else thickness
    if meta_dict['dimensionality'] == '2D' and slice_spacing > thickness + 1e-2:
        raise ValueError('Multi-slice volume has gaps between slices')
    meta_dict['resolution_mm'] = meta_dict['resolution_mm'] + (slice_spacing,)
    meta_dict['acqMatrixShape'] = meta_dict['acqMatrixShape'] + (len(positions),)
    data = np.stack([data[i] for i in order], axis=-1)
    data = data.astype(dtype)
    return ImageVolume(data, Metadata(**meta_dict))

def read_mr_image(file: str) -> Dataset | None:
    """ Read DICOM file, returning None if it is not an MR image. """
    try:
        dicom = dcmread(file)
    except InvalidDicomError:
        return None
    if dicom.get('SOPClassUID') not in (MR_IMAGE_STORAGE, ENHANCED_MR_IMAGE_STORAGE):
        return None
    return dicom

def get_attribute(dicom: Dataset, keyword: str, frame: int = 0, default=None):
    """ Get attribute by keyword from the top level, or else from the functional groups of an enhanced DICOM. """
    if keyword in dicom:
        return getattr(dicom, keyword)
    groups = []
    if 'PerFrameFunctionalGroupsSequence' in dicom:
        groups.append(dicom.PerFrameFunctionalGroupsSequence[frame])
    if 'SharedFunctionalGroupsSequence' in dicom:
        groups.append(dicom.SharedFunctionalGroupsSequence[0])
    for group in groups:
        for element in group:
            if element.VR == 'SQ' and len(element.value) > 0 and keyword in element.value[0]:
                return getattr(element.value[0], keyword)
    return default

def get_slice_position(dicom: Dataset, frame: int = 0) -> float:
    """ Get position (mm) of an image along the slice normal. """
    orientation = np.array(get_attribute(dicom, 'ImageOrientationPatient', frame=frame), dtype=float)
    position = np.array(get_attribute(dicom, 'ImagePositionPatient', frame=frame), dtype=float)
    normal = np.cross(orientation[:3], orientation[3:])
    return float(np.dot(normal, position))

def rescale(pixels: npt.NDArray, dicom: Dataset, frame: int = 0) -> npt.NDArray[np.float64]:
    """ Convert stored pixel values to real-world values using the rescale slope & intercept. """
    slope = float(get_attribute(dicom, 'RescaleSlope', frame=frame, default=1))
    intercept = float(get_attribute(dicom, 'RescaleIntercept', frame=frame, default=0))
    return pixels * slope + intercept

def read_frames_classic(dicom: Dataset) -> tuple[list[npt.NDArray], list[float]]:
    """ Read image data & slice position from a single-frame DICOM. """
    return [rescale(dicom.pixel_array, dicom)], [get_slice_position(dicom)]

def read_frames_enhanced(dicom: Dataset) -> tuple[list[npt.NDArray], list[float]]:
    """ Read image data & slice positions from a multi-frame (enhanced) DICOM. """
    pixels = dicom.pixel_array
    if pixels.ndim == 2:
        pixels = pixels[np.newaxis]
    data = [rescale(pixels[i], dicom, frame=i) for i in range(pixels.shape[0])]
    positions = [get_slice_position(dicom, frame=i) for i in range(pixels.shape[0])]
    return data, positions

def read_meta_standard(dicom: Dataset) -> dict:
    """ Read metadata from a DICOM file using standard attributes (e.g. Philips).

    Returns a dictionary of Metadata fields, except that the number of slices and slice spacing are added by the caller.
    The slice thickness is included under the extra key 'sliceThickness_mm'. Unavailable fields are set to None.
    """
    def get(keyword):
        return get_attribute(dicom, keyword)

    def get_float(keyword):
        value = get(keyword)
        return None if value is None else float(value)

    if 'AcquisitionMatrix' in dicom:
        acqMatrix = np.array(dicom.AcquisitionMatrix)
        inplaneMatrixSize = tuple(int(n) for n in acqMatrix[np.nonzero(acqMatrix)])
    else:
        inplaneMatrixSize = (int(get('MRAcquisitionFrequencyEncodingSteps')), int(get('MRAcquisitionPhaseEncodingStepsInPlane')))

    pulseSequenceName = get('PulseSequenceName')
    if pulseSequenceName is None and PHILIPS_SCANNING_TECHNIQUE in dicom:
        pulseSequenceName = dicom[PHILIPS_SCANNING_TECHNIQUE].value
        if isinstance(pulseSequenceName, bytes):
            pulseSequenceName = pulseSequenceName.decode(errors='ignore').strip()
    if pulseSequenceName is None:
        pulseSequenceName = get('SequenceName')

    duration_s = get_float('AcquisitionDuration')
    imagingFrequency_MHz = get_float('ImagingFrequency') or get_float('TransmitterFrequency')
    echoTime_ms = get_float('EchoTime') or get_float('EffectiveEchoTime')
    echoTrainLength = get('EchoTrainLength')
    phaseDirection = get('InPlanePhaseEncodingDirection')
    seriesName = dicom.get('SeriesDescription') or dicom.get('ProtocolName', '')

    return {
        'date_YYYYMMDD': dicom.StudyDate,
        'scanner': dicom.ManufacturerModelName,
        'staticFieldStrength_T': float(dicom.MagneticFieldStrength),

        'seriesName': seriesName,
        'dimensionality': dicom.MRAcquisitionType,
        'pulseSequenceName': pulseSequenceName,
        'duration_s': None if duration_s is None else np.round(duration_s),

        'acqMatrixShape': inplaneMatrixSize,
        'resolution_mm': tuple(map(float, get('PixelSpacing'))),
        'sliceThickness_mm': float(get('SliceThickness')),
        'refocusFlipAngle_deg': get_float('FlipAngle'),
        'echoTrainLength': None if echoTrainLength is None else int(echoTrainLength),
        'echoTime_ms': echoTime_ms,
        'repetitionTime_ms': get_float('RepetitionTime'),
        'centerFrequency_Hz': None if imagingFrequency_MHz is None else imagingFrequency_MHz * 1e6,
        'pixelBandwidth_Hz': get_float('PixelBandwidth'),
        'readoutDirection': None if phaseDirection is None else (0 if phaseDirection == 'ROW' else 1),

        'containsMetal': 'PLA' not in seriesName
    }
