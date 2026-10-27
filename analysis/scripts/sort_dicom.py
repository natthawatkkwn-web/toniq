import argparse
import shutil
from collections import defaultdict
from pathlib import Path

from pydicom import dcmread
from pydicom.errors import InvalidDicomError

from toniq.dicom import MR_IMAGE_STORAGE, ENHANCED_MR_IMAGE_STORAGE

p = argparse.ArgumentParser(description='Sort an exported DICOM folder (e.g. from Philips) into one folder per series, named Series<SeriesNumber> for use in config files')
p.add_argument('in_dir', type=str, help='exported DICOM folder; searched recursively')
p.add_argument('out_dir', type=str, help='folder where series folders are created; use as root in config file')

if __name__ == '__main__':

    args = p.parse_args()
    out_dir = Path(args.out_dir)

    series = defaultdict(lambda: {'description': '', 'files': 0, 'frames': 0})
    for f in sorted(Path(args.in_dir).rglob('*')):
        if not f.is_file():
            continue
        try:
            dicom = dcmread(f, stop_before_pixels=True)
        except InvalidDicomError:
            continue
        if dicom.get('SOPClassUID') not in (MR_IMAGE_STORAGE, ENHANCED_MR_IMAGE_STORAGE):
            continue
        name = 'Series{}'.format(dicom.SeriesNumber)
        series_dir = out_dir / name
        series_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, series_dir / '{}.dcm'.format(dicom.SOPInstanceUID))
        series[name]['description'] = dicom.get('SeriesDescription') or dicom.get('ProtocolName', '')
        series[name]['files'] += 1
        series[name]['frames'] += int(dicom.get('NumberOfFrames', 1))

    print('{:<14}{:>7}{:>8}  {}'.format('folder', 'files', 'frames', 'description'))
    for name in sorted(series, key=lambda s: int(s[len('Series'):])):
        info = series[name]
        print('{:<14}{:>7}{:>8}  {}'.format(name, info['files'], info['frames'], info['description']))
