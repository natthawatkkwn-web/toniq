import argparse

from toniq.dicom import sort_series

p = argparse.ArgumentParser(description='Sort an exported DICOM folder (e.g. from Philips) into one folder per series, named Series<SeriesNumber> for use in config files')
p.add_argument('in_dir', type=str, help='exported DICOM folder; searched recursively')
p.add_argument('out_dir', type=str, help='folder where series folders are created; use as root in config file')

if __name__ == '__main__':

    args = p.parse_args()
    series = sort_series(args.in_dir, args.out_dir)
    print('{:<14}{:>7}{:>8}  {}'.format('folder', 'files', 'frames', 'description'))
    for info in series:
        print('{:<14}{:>7}{:>8}  {}'.format(info['folder'], info['files'], info['frames'], info['description']))
