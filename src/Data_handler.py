import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from utils import load_config
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_spectrum_limits(df):
    meta_cols = ['filename', 'agn_type', 'z', 'snr', 'obj_id']
    flux_cols = [c for c in df.columns if c not in meta_cols]
    wavelengths = np.array(flux_cols).astype(float)

    # Identify which pixels are NEVER zero across the whole dataset
    # (We check for != 0.0 across all rows)
    is_never_zero = (df[flux_cols] != 0.0).all(axis=0)

    # The 'Safe Box' is the range where this mask is True
    safe_wavelengths = wavelengths[is_never_zero]
    blue_limit = safe_wavelengths.min()
    red_limit = safe_wavelengths.max()

    print(f"--- Safe Overlap Analysis ---")
    print(f"Safe Blue Limit: {blue_limit:.1f} Å")
    print(f"Safe Red Limit:  {red_limit:.1f} Å")
    print(f"Total valid range: {red_limit - blue_limit:.1f} Å")
    return blue_limit, red_limit


def get_spectrum(df, identifier):
    """
    Retrieves the wavelength grid and flux array for a specific spectrum.
    
    Parameters:
    df (pd.DataFrame): The processed AGN catalog.
    identifier (int or str): The row index (int) or the filename (str).
    
    Returns:
    tuple: (wavelengths, flux, metadata)
    """
    # Define metadata columns to separate them from flux data
    meta_cols = ['filename', 'agn_type', 'z', 'snr', 'obj_id']
    flux_cols = [c for c in df.columns if c not in meta_cols]
    
    # 1. Locate the row
    if isinstance(identifier, (int, np.integer)):
        row = df.iloc[identifier]
    elif isinstance(identifier, str):
        match = df[df['filename'] == identifier]
        if match.empty:
            raise ValueError(f"Filename '{identifier}' not found in DataFrame.")
        row = match.iloc[0]
    else:
        raise TypeError("Identifier must be an integer (row index) or string (filename).")
    
    # 2. Extract data
    wavelengths = np.array(flux_cols).astype(float)
    flux = row[flux_cols].values.astype(np.float32)
    metadata = row[meta_cols].to_dict()
    
    return wavelengths, flux, metadata

def prepare_for_nn(df_clean):
    """
    Prepares the cleaned DataFrame for a neural network.
    Returns X (reshaped for PyTorch 1D CNNs) and y (binary labels).
    """
    meta_cols = ['filename', 'agn_type', 'z', 'snr','obj_id']
    flux_cols = [c for c in df_clean.columns if c not in meta_cols]
    

    X = df_clean[flux_cols].values
    # Add channel dimension: (Samples, Channels, Sequence_Length)
    # This creates shape (N, 1, 5001)
    X = np.expand_dims(X, axis=1)
    
    # Encode Type 1 -> 0, Type 2 -> 1
    y = (df_clean['agn_type'] == 2).astype(int).values
    
    z = df_clean['z'].values
    

    # Randomly shuffle the entire dataset
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    z = z[indices]
    
    z_mean = np.mean(z)
    z_std = np.std(z)
    z_scaled = (z - z_mean) / (z_std + 1e-6)
    

    return X, y, z_scaled
    
class AGNSpectraDataset(Dataset):
    def __init__(self, X, y, wavelengths=None, z=None, apply_masking=False, mask_lines=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        
        # Ensure z is passed and converted to a tensor
        if z is None:
            raise ValueError("Redshift (z) must be provided for adversarial training.")
        self.z = torch.tensor(z, dtype=torch.float32).view(-1, 1)
        
        self.apply_masking = apply_masking
        self.mask_lines = mask_lines
        self.wavelengths = torch.tensor(wavelengths, dtype=torch.float32) if wavelengths is not None else None
        
        self.mask_ranges = [(4800, 5100), (6500, 6650), (4575, 4800)]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        y = self.y[idx]
        z = self.z[idx] # FIXED: Changed self.z_scaled to self.z

        if self.mask_lines and self.wavelengths is not None:
            for start_wave, end_wave in self.mask_ranges:
                mask_idx = (self.wavelengths >= start_wave) & (self.wavelengths <= end_wave)
                x[..., mask_idx] = 0.0

        if self.apply_masking:
            seq_len = x.shape[-1]
            
            # 1. TARGETED H-ALPHA DROPOUT (50% chance)
            # Force the network to survive without the red edge crutch
            if torch.rand(1).item() < 0.5:
                x[..., -250:] = 0.0
                
            # 2. RANDOM SPECAUGMENT (The Whack-a-Mole Defense)
            # Drop 1 or 2 smaller random masks to force continuum learning
            num_masks = torch.randint(1, 3, (1,)).item()
            for _ in range(num_masks):
                mask_len = torch.randint(30, 100, (1,)).item()
                start_idx = torch.randint(0, seq_len - mask_len, (1,)).item()
                x[..., start_idx:start_idx + mask_len] = 0.0

        return x, y, z

        
def prepare_agn_data(df, batch_size=256, random_state=42, mask_lines=False):
    meta_cols = ['filename', 'agn_type', 'z', 'snr', 'obj_id']
    flux_cols = [c for c in df.columns if c not in meta_cols]
    wavelengths = np.array(flux_cols).astype(float)
    config = load_config(os.path.join(BASE_DIR, 'config.yml'))


    X = df[flux_cols].values
    X = np.expand_dims(X, axis=1) # (N, 1, 1024)
    y = (df['agn_type'] == 2).astype(int).values # Binary Targets
    
    # --- NEW: Extract and Scale Redshift ---
    z_raw = df['z'].values
    z_mean, z_std = z_raw.mean(), z_raw.std()
    z_scaled = (z_raw - z_mean) / (z_std + 1e-8)

    # --- NEW: Synced Splitting (X, y, and z) ---
    # Split 1: 70% Train, 30% Temp
    X_train, X_temp, y_train, y_temp, z_train, z_temp = train_test_split(
        X, y, z_scaled, test_size=0.30, random_state=random_state, stratify=y, shuffle=True
    )

    # Split 2: 15% Val, 15% Test
    X_val, X_test, y_val, y_test, z_val, z_test = train_test_split(
        X_temp, y_temp, z_temp, test_size=0.50, random_state=random_state, stratify=y_temp, shuffle=True
    )

    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32)

    # --- NEW: Pass z to Datasets ---
    train_ds = AGNSpectraDataset(X_train, y_train, z=z_train, wavelengths=wavelengths, apply_masking=True, mask_lines=True)
    val_ds   = AGNSpectraDataset(X_val, y_val, z=z_val, wavelengths=wavelengths, apply_masking=True, mask_lines=True)
    test_ds  = AGNSpectraDataset(X_test, y_test, z=z_test, wavelengths=wavelengths, apply_masking=False, mask_lines=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, pos_weight


class SyntheticSiameseDataset(Dataset):
    def __init__(self, df, flux_cols, epoch_size=2000):
        """
        df: The pandas dataframe (df_clean)
        flux_cols: List of the wavelength column names
        epoch_size: Arbitrary number of pairs to generate per epoch
        """
        self.df = df
        self.flux_cols = flux_cols
        self.epoch_size = epoch_size
        
        # Pre-sort indices by class so we can sample them instantly
        self.t1_indices = self.df[self.df['agn_type'] == 1].index.values
        self.t2_indices = self.df[self.df['agn_type'] == 2].index.values

    def __len__(self):
        # Because we are generating pairs randomly, the concept of "dataset length" 
        # is arbitrary. We set it to a fixed size per epoch.
        return self.epoch_size

    def get_spectrum(self, idx):
        # Extract, convert to float32, and add the channel dimension [1, Seq_Len]
        x = self.df.loc[idx, self.flux_cols].values.astype(np.float32)
        return torch.tensor(x).unsqueeze(0) 

    def __getitem__(self, i):
        # 1. Flip a coin: 50% chance of Change (1) or Static (0)
        target = np.random.randint(0, 2)
        
        if target == 1: 
            # STATE CHANGE: Pick one random Type 1 and one random Type 2
            idx1 = np.random.choice(self.t1_indices)
            idx2 = np.random.choice(self.t2_indices)
        else: 
            # STATIC: 50% chance of T1+T1, 50% chance of T2+T2
            if np.random.rand() > 0.5:
                idx1 = np.random.choice(self.t1_indices)
                idx2 = np.random.choice(self.t1_indices)
            else:
                idx1 = np.random.choice(self.t2_indices)
                idx2 = np.random.choice(self.t2_indices)

        x1 = self.get_spectrum(idx1)
        x2 = self.get_spectrum(idx2)
        
        # 2. Randomly swap the chronological order
        # We don't want the network memorizing that Type 1 is always 'x1'.
        # A change is a change, regardless of direction.
        if np.random.rand() > 0.5:
            x1, x2 = x2, x1

        return x1, x2, torch.tensor([target], dtype=torch.float32)