"""
ECG Image Digitizer - Local inference script
Uses hengck23's 3-stage pipeline to convert ECG images to 12-lead CSV signals.

Usage:
    python ecg_digitize.py --input <image_path> --output <csv_path>

Output CSV columns: Time, I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
"""

import sys, os, argparse
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F

# Add the hengck23 library to path
LIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hengck23-submit-physionet')
WEIGHT_DIR = os.path.join(LIB_DIR, 'weight')
sys.path.insert(0, LIB_DIR)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_stage0():
    from stage0_model import Net as Stage0Net
    import stage0_common
    net = Stage0Net(pretrained=False)
    stage0_common.load_net(net, os.path.join(WEIGHT_DIR, 'stage0-last.checkpoint.pth'))
    net.to(DEVICE).eval()
    return net


def load_stage1():
    from stage1_model import Net as Stage1Net
    import stage1_common
    net = Stage1Net(pretrained=False)
    stage1_common.load_net(net, os.path.join(WEIGHT_DIR, 'stage1-last.checkpoint.pth'))
    net.to(DEVICE).eval()
    return net


def load_stage2():
    from stage2_model import Net as Stage2Net
    import stage2_common
    net = Stage2Net(pretrained=False)
    stage2_common.load_net(net, os.path.join(WEIGHT_DIR, 'stage2-00005810.checkpoint.pth'))
    net.to(DEVICE).eval()
    return net


def run_stage0(net, image):
    import stage0_common
    batch = stage0_common.image_to_batch(image)
    with torch.no_grad():
        output = net(batch)
    rotated, keypoint = stage0_common.output_to_predict(image, batch, output)
    normalised, keypoint, homo = stage0_common.normalise_by_homography(rotated, keypoint)
    return normalised


def run_stage1(net, normalised):
    import stage1_common
    batch = {
        'image': torch.from_numpy(
            np.ascontiguousarray(normalised.transpose(2, 0, 1))
        ).unsqueeze(0).to(DEVICE),
    }
    with torch.no_grad():
        output = net(batch)
    try:
        gridpoint_xy, _ = stage1_common.output_to_predict(normalised, batch, output)
    except ValueError as e:
        raise RuntimeError(
            'Stage 1 failed to detect grid points in the image. '
            'Please make sure the input is a standard 12-lead ECG image.'
        ) from e
    rectified = stage1_common.rectify_image(normalised, gridpoint_xy)
    return rectified


def run_stage2(net, rectified, num_samples=None):
    import stage2_common

    # Fixed parameters from the hengck23 model
    y0, y1 = 0, 1696
    x0, x1 = 0, 2176
    zero_mv = [703.5, 987.5, 1271.5, 1531.5]
    mv_to_pixel = 78.0
    t0, t1 = 118, 2080

    crop = rectified[y0:y1, x0:x1]
    batch = {
        'image': torch.from_numpy(
            np.ascontiguousarray(crop.transpose(2, 0, 1))
        ).unsqueeze(0),
    }

    with torch.no_grad():
        output = net(batch)

    pixel = output['pixel'].float().data.cpu().numpy()[0]
    series_in_pixel = stage2_common.pixel_to_series(
        pixel[..., t0:t1], zero_mv, num_samples
    )
    series = (np.array(zero_mv).reshape(4, 1) - series_in_pixel) / mv_to_pixel
    series = stage2_common.filter_series_by_limits(series)
    return series


def series_to_dataframe(series, fs=200):
    """
    Convert the (4, L) series array to a 12-lead DataFrame.

    fs=200: matches ecg-image-kit generator's `-r 200` render rate used in
    generate_5class_images.py. Stage 2 crop (118..2080 px) represents a
    10-second rhythm strip → 1962 samples ÷ 10 s ≈ 196 Hz; 200 is the
    nominal rate. Previously fs=500 was wrong and caused downstream boki
    classification to see heart rates around 300-400 bpm.

    series layout:
        series[0] → I,   aVR, V1, V4  (4 equal segments)
        series[1] → II,  aVL, V2, V5  (4 equal segments)
        series[2] → III, aVF, V3, V6  (4 equal segments)
        series[3] → II rhythm strip    (full length)
    """
    lead_order = [
        ['I',   'aVR', 'V1', 'V4'],
        ['II',  'aVL', 'V2', 'V5'],
        ['III', 'aVF', 'V3', 'V6'],
    ]
    lead_data = {}
    for row_idx, leads in enumerate(lead_order):
        segments = np.array_split(series[row_idx], 4)
        for lead, seg in zip(leads, segments):
            lead_data[lead] = seg

    # Override II with full rhythm strip
    lead_data['II'] = series[3]

    # Build DataFrame — use II (longest) to set the time axis
    n = len(lead_data['II'])
    time = np.arange(n) / float(fs)

    rows = {'Time': time}
    for lead in ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']:
        s = lead_data[lead]
        if len(s) < n:
            s = np.pad(s, (0, n - len(s)), mode='edge')
        rows[lead] = s[:n]

    return pd.DataFrame(rows)


def digitize(image_path, output_csv, num_samples=None, verbose=True):
    """
    Main entry point.

    Args:
        image_path  : Path to input ECG image (PNG/JPG)
        output_csv  : Path to output CSV file
        num_samples : Expected samples per lead (None = use pixel width)
        verbose     : Print progress
    """
    if verbose:
        print(f'Device: {DEVICE}')
        print(f'Loading image: {image_path}')

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f'Cannot read image: {image_path}')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if verbose: print('Loading Stage 0 model ...')
    stage0_net = load_stage0()
    if verbose: print('Running Stage 0 (orientation & perspective correction) ...')
    normalised = run_stage0(stage0_net, image)
    del stage0_net
    torch.cuda.empty_cache() if DEVICE == 'cuda' else None

    if verbose: print('Loading Stage 1 model ...')
    stage1_net = load_stage1()
    if verbose: print('Running Stage 1 (grid detection & rectification) ...')
    rectified = run_stage1(stage1_net, normalised)
    del stage1_net
    torch.cuda.empty_cache() if DEVICE == 'cuda' else None

    if verbose: print('Loading Stage 2 model ...')
    stage2_net = load_stage2()
    if verbose: print('Running Stage 2 (signal extraction) ...')
    series = run_stage2(stage2_net, rectified, num_samples)
    del stage2_net
    torch.cuda.empty_cache() if DEVICE == 'cuda' else None

    if verbose: print('Saving CSV ...')
    df = series_to_dataframe(series)
    df.to_csv(output_csv, index=False)

    if verbose:
        print(f'Done! Output: {output_csv}')
        print(f'  Shape: {df.shape}  ({len(df)} time steps, {len(df.columns)-1} leads)')

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Digitize ECG image to CSV')
    parser.add_argument('--input',   required=True,  help='Input ECG image path')
    parser.add_argument('--output',  required=True,  help='Output CSV path')
    parser.add_argument('--samples', type=int, default=None,
                        help='Expected samples per lead (default: auto)')
    parser.add_argument('--quiet',   action='store_true', help='Suppress progress output')
    args = parser.parse_args()

    digitize(args.input, args.output, args.samples, verbose=not args.quiet)
