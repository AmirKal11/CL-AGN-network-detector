"""
Quick plot: spectrum before vs after morphological_continuum_subtraction.
"""
import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'src'))

from utils import load_config


def morphological_continuum_subtraction(x, window_size=151):
    pad = window_size // 2
    x_padded = F.pad(x, (pad, pad), mode='reflect')
    continuum = F.avg_pool1d(x_padded, kernel_size=window_size, stride=1)
    x_flattened = x - continuum
    mean = x_flattened.mean(dim=-1, keepdim=True)
    std = x_flattened.std(dim=-1, keepdim=True)
    return (x_flattened - mean) / (std + 1e-8)


def main():
    config = load_config(os.path.join(BASE_DIR, 'config.yml'))
    df = pd.read_parquet(os.path.join(BASE_DIR, config['data']['processed_catalog']))
    df = df.sort_values('snr', ascending=False).drop_duplicates(subset=['obj_id'])

    meta_cols = ['filename', 'agn_type', 'z', 'snr', 'obj_id']
    flux_cols = [c for c in df.columns if c not in meta_cols]
    wavelengths = np.array(flux_cols, dtype=float)

    # Pick one Type 1 and one Type 2 spectrum (highest SNR)
    idx_t1 = df[df['agn_type'] == 1].index[0]
    idx_t2 = df[df['agn_type'] == 2].index[0]

    fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=True)

    for row, (idx, label) in enumerate([(idx_t1, 'Type 1'), (idx_t2, 'Type 2')]):
        obj_id = df.loc[idx, 'obj_id']
        raw_flux = df.loc[idx, flux_cols].values.astype(np.float32)
        print(obj_id)
        # Convert to tensor [1, 1, Seq_Len]
        x_tensor = torch.tensor(raw_flux).unsqueeze(0).unsqueeze(0)
        #x_processed = morphological_continuum_subtraction(x_tensor)
        #processed_flux = x_processed.squeeze().numpy()
        processed_flux = raw_flux

        # --- Before ---
        ax_before = axes[row, 0]
        ax_before.plot(wavelengths, raw_flux, color='steelblue', linewidth=1.0)
        ax_before.set_ylabel('Raw Flux', fontsize=12)
        ax_before.set_title(f'{label} (obj_id={obj_id}) — Before (Raw)', fontsize=14, fontweight='bold')
        ax_before.grid(True, linestyle='--', alpha=0.4)

        # --- After ---
        ax_after = axes[row, 1]
        ax_after.plot(wavelengths, processed_flux, color='darkorange', linewidth=1.0)
        ax_after.set_ylabel('Normalized Flux', fontsize=12)
        ax_after.set_title(f'{label} (obj_id={obj_id}) — After (Continuum-Subtracted & Normalized)', fontsize=14, fontweight='bold')
        ax_after.grid(True, linestyle='--', alpha=0.4)

    axes[1, 0].set_xlabel('Rest Wavelength (Å)', fontsize=12)
    axes[1, 1].set_xlabel('Rest Wavelength (Å)', fontsize=12)

    fig.suptitle('Morphological Continuum Subtraction (window=151)', fontsize=12, y=1.01)
    plt.tight_layout()

    save_path = os.path.join(BASE_DIR, 'models', 'continuum_subtraction_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {save_path}")
    plt.show()


if __name__ == '__main__':
    main()
