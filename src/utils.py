import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def plot_spectrum(df, row_index=0):
    """
    Plots a single spectrum from the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The loaded processed_agn_catalog
    row_index (int): The row number to plot
    """
    # 1. Identify flux columns (all columns except the metadata)
    meta_cols = ['filename', 'agn_type', 'z', 'snr']
    flux_cols = [c for c in df.columns if c not in meta_cols]
    
    # 2. Convert column names back to floats for the x-axis
    wavelengths = np.array(flux_cols).astype(float)
    
    # 3. Extract the specific row
    row = df.iloc[row_index]
    flux_values = row[flux_cols].values
    
    # 4. Plotting
    plt.figure(figsize=(12, 5), dpi=100)
    plt.plot(wavelengths, flux_values, color='#2c3e50', lw=1, alpha=0.8)
    
    # Aesthetics
    plt.title(f"Spectrum: {row['filename']} (Type {row['agn_type']}, z={row['z']:.3f})", 
              fontsize=14, fontweight='bold')
    plt.xlabel("Rest-frame Wavelength [Å]", fontsize=12)
    plt.ylabel("Standardized Flux", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def load_config(config_path='../config.yml'):
    """
    Loads configurations from a YAML file.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
