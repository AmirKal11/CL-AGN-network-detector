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
        
        # Mask ranges widened to cover BROAD LINE WINGS, not just cores:
        # (4575, 5550): Hβ broad wings + [O III] + Fe II pseudo-continuum
        # (5800, 5950): He I 5876 broad
        # (6200, 6699): [O I] 6300 + [N II] + Hα broad wings + [S II]
        self.mask_ranges = [(4575, 5550), (5800, 5950), (6200, 6699)]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        y = self.y[idx]
        z = self.z[idx] # FIXED: Changed self.z_scaled to self.z

        if self.mask_lines and self.wavelengths is not None:
            # 1. Build a boolean mask for ALL line regions at once
            line_mask = torch.zeros(self.wavelengths.shape, dtype=torch.bool)
            for start_wave, end_wave in self.mask_ranges:
                line_mask |= (self.wavelengths >= start_wave) & (self.wavelengths <= end_wave)
            
            # 2. Zero out the line regions
            x[..., line_mask] = 0.0
            
            # 3. RE-NORMALIZE using only the UNMASKED pixels
            #    This removes the global statistical fingerprint that the original
            #    normalization (computed with lines present) baked into the continuum.
            unmasked = x[..., ~line_mask]
            if unmasked.numel() > 0:
                mean = unmasked.mean()
                std = unmasked.std()
                x[..., ~line_mask] = (unmasked - mean) / (std + 1e-8)

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
    train_ds = AGNSpectraDataset(X_train, y_train, z=z_train, wavelengths=wavelengths, apply_masking=True, mask_lines=mask_lines)
    val_ds   = AGNSpectraDataset(X_val, y_val, z=z_val, wavelengths=wavelengths, apply_masking=True, mask_lines=mask_lines)
    test_ds  = AGNSpectraDataset(X_test, y_test, z=z_test, wavelengths=wavelengths, apply_masking=False, mask_lines=mask_lines)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, pos_weight

class SyntheticSiameseDataset(Dataset):
    def __init__(
        self,
        df,
        flux_cols,
        epoch_size=2000,
        k_neighbors=20,
        mode="train",
        change_pair_prob=0.15,
        mask_lines=False,
    ):
        """
        Dynamically generates Siamese pairs.

        label 0 = static-like pair:
            Type 1 + Type 1
            or Type 2 + Type 2

        label 1 = change-like pair:
            Type 1 + Type 2
            or Type 2 + Type 1

        Args:
            df:
                DataFrame containing spectra and metadata.

            flux_cols:
                List of wavelength/flux columns.

            epoch_size:
                Number of generated pairs per epoch.

            k_neighbors:
                For train mode, choose randomly from nearest k matched objects.

            mode:
                "train" -> random neighbor among nearest k
                "val" / "test" -> deterministic nearest neighbor

            change_pair_prob:
                Probability of generating a change-like pair.
                Example:
                    0.15 means 15% change-like, 85% static-like.
        """
        self.df = df.reset_index(drop=True)
        self.flux_cols = flux_cols
        self.epoch_size = epoch_size
        self.k_neighbors = k_neighbors
        self.mode = mode
        self.change_pair_prob = change_pair_prob
        self.mask_lines = mask_lines
        self.mask_ranges = [(4800, 5100), (6500, 6650), (4575, 4800)]
        self.wavelengths = np.array(self.flux_cols).astype(float)

        if self.mode not in ["train", "val", "test"]:
            raise ValueError(f"mode must be 'train', 'val', or 'test', got {mode}")

        if not (0.0 <= self.change_pair_prob <= 1.0):
            raise ValueError(
                f"change_pair_prob must be between 0 and 1, got {change_pair_prob}"
            )

        self.indices_by_type = {
            1: self.df[self.df["agn_type"] == 1].index.values,
            2: self.df[self.df["agn_type"] == 2].index.values,
        }

        if len(self.indices_by_type[1]) == 0:
            raise ValueError("No Type 1 AGN found in dataframe.")

        if len(self.indices_by_type[2]) == 0:
            raise ValueError("No Type 2 AGN found in dataframe.")

        self.z_std = self.df["z"].std() + 1e-8
        self.snr_std = self.df["snr"].std() + 1e-8

    def __len__(self):
        return self.epoch_size

    def get_spectrum(self, idx):
        x = self.df.loc[idx, self.flux_cols].values.astype(np.float32)
        if self.mask_lines:
            for start_wave, end_wave in self.mask_ranges:
                mask_idx = (self.wavelengths >= start_wave) & (self.wavelengths <= end_wave)
                x[mask_idx] = 0.0
        return torch.tensor(x).unsqueeze(0)  # [1, SeqLen]

    def find_matched_partner(self, anchor_idx, partner_type):
        anchor = self.df.loc[anchor_idx]

        candidate_indices = self.indices_by_type[partner_type]

        # Avoid pairing object with itself when possible.
        if "obj_id" in self.df.columns:
            candidate_indices = np.array([
                idx for idx in candidate_indices
                if self.df.loc[idx, "obj_id"] != anchor["obj_id"]
            ])
        else:
            candidate_indices = np.array([
                idx for idx in candidate_indices
                if idx != anchor_idx
            ])

        if len(candidate_indices) == 0:
            raise RuntimeError(
                f"No valid partner candidates found for "
                f"anchor_idx={anchor_idx}, partner_type={partner_type}"
            )

        candidates = self.df.loc[candidate_indices]

        dz = (candidates["z"].values - anchor["z"]) / self.z_std
        dsnr = (candidates["snr"].values - anchor["snr"]) / self.snr_std

        dist = dz**2 + dsnr**2

        if self.mode == "train":
            k = min(self.k_neighbors, len(candidate_indices))
            nearest_positions = np.argsort(dist)[:k]
            chosen_position = np.random.choice(nearest_positions)
        else:
            chosen_position = np.argmin(dist)

        partner_idx = candidate_indices[chosen_position]
        return partner_idx

    def __getitem__(self, i):
        # 1. Choose pair label according to desired prior.
        # target = 1 means change-like.
        # target = 0 means static-like.
        target = 1 if np.random.rand() < self.change_pair_prob else 0

        # 2. Choose anchor type.
        anchor_type = np.random.choice([1, 2])

        # 3. Choose anchor object.
        anchor_idx = np.random.choice(self.indices_by_type[anchor_type])

        # 4. Choose partner type based on target.
        if target == 0:
            partner_type = anchor_type
        else:
            partner_type = 2 if anchor_type == 1 else 1

        # 5. Find matched partner in z/SNR space.
        partner_idx = self.find_matched_partner(
            anchor_idx=anchor_idx,
            partner_type=partner_type,
        )

        x1 = self.get_spectrum(anchor_idx)
        x2 = self.get_spectrum(partner_idx)

        # 6. Randomly swap order during training only.
        if self.mode == "train" and np.random.rand() > 0.5:
            x1, x2 = x2, x1

        y = torch.tensor([target], dtype=torch.float32)

        return x1, x2, y