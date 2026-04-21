"""
Quick ECG waveform visualizer.
Usage: python visualize_ecg.py result.csv
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Standard 12-lead layout
LAYOUT = [
    ['I',   'aVR', 'V1', 'V4'],
    ['II',  'aVL', 'V2', 'V5'],
    ['III', 'aVF', 'V3', 'V6'],
]
RHYTHM_LEAD = 'II'


def plot_ecg(csv_path):
    df = pd.read_csv(csv_path)
    time = df['Time'].values

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(f'ECG — {csv_path}', fontsize=13)

    # 3 rows of 4 leads + 1 rhythm strip row
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.55, wspace=0.3)

    def plot_lead(ax, lead, t, signal):
        ax.plot(t, signal, lw=0.8, color='black')
        ax.set_title(lead, fontsize=10, pad=2)
        ax.set_xlabel('s', fontsize=7)
        ax.set_ylabel('mV', fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle='--', alpha=0.4)
        # Reference lines at ±1 mV
        ax.axhline(1,  color='red',  lw=0.5, ls=':', alpha=0.6)
        ax.axhline(-1, color='red',  lw=0.5, ls=':', alpha=0.6)
        ax.axhline(0,  color='gray', lw=0.5, alpha=0.5)

    # 3 × 4 grid
    for row_idx, leads in enumerate(LAYOUT):
        n_cols = df[leads[0]].values.shape[0]  # just to get length
        # Each lead occupies ~1/4 of the time axis
        for col_idx, lead in enumerate(leads):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            seg = df[lead].values
            seg_t = time[:len(seg)]
            plot_lead(ax, lead, seg_t, seg)

    # Rhythm strip (II, full width)
    ax_rhythm = fig.add_subplot(gs[3, :])
    plot_lead(ax_rhythm, f'{RHYTHM_LEAD} (rhythm strip)', time, df[RHYTHM_LEAD].values)

    plt.savefig('ecg_preview.png', dpi=150, bbox_inches='tight')
    print('Saved: ecg_preview.png')
    plt.show()


if __name__ == '__main__':
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'result.csv'
    plot_ecg(csv_path)
