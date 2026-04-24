import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'src'))

from utils import load_config
from architectures import SpectraNet
import pandas as pd
from Data_handler import prepare_for_nn

def morphological_continuum_subtraction(x, window_size=151):
    """
    Identical filter used in train.py to ensure the network sees the flattened data.
    """
    pad = window_size // 2
    x_padded = F.pad(x, (pad, pad), mode='reflect')
    continuum = F.avg_pool1d(x_padded, kernel_size=window_size, stride=1)
    x_flattened = x - continuum
    mean = x_flattened.mean(dim=-1, keepdim=True)
    std = x_flattened.std(dim=-1, keepdim=True)
    x_normalized = (x_flattened - mean) / (std + 1e-8)
    return x_normalized


class GradCAM:
    """
    Grad-CAM for 1D Convolutional Networks.
    Highlights the regions of the 1D sequence that contributed most to the prediction.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def __call__(self, x, target_class=1):
        """
        x: Input tensor of shape (1, 1, Sequence_Length)
        target_class: 1 for Type 2, 0 for Type 1.
        """
        self.model.eval()
        self.model.zero_grad()
        
        # FIX: Unpack the dual outputs from the new SpectraNet architecture
        class_output, _ = self.model(x)
        
        pred_prob = torch.sigmoid(class_output).item()
        
        # For binary classification with a single logit:
        # If target_class == 1, we want gradients that increase the logit.
        # If target_class == 0, we want gradients that decrease the logit.
        if target_class == 0:
            loss = -class_output
        else:
            loss = class_output
            
        # Backward pass to get gradients
        loss.backward(retain_graph=True)
        
        # Calculate Grad-CAM
        # Global average pool the gradients across the sequence dimension
        weights = torch.mean(self.gradients, dim=2, keepdim=True) # (1, Channels, 1)
        
        # Weighted combination of activations
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True) # (1, 1, L)
        
        # Apply ReLU to keep only features that have a positive influence
        cam = F.relu(cam)
        
        # Normalize between 0 and 1
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-8)
        
        # Interpolate back to original sequence length (1024)
        # MPS doesn't support 1D linear interpolation, so we do it on CPU!
        cam = cam.cpu()
        cam = F.interpolate(cam, size=x.shape[2], mode='linear', align_corners=False)
        
        return cam.detach().squeeze().numpy(), pred_prob

def plot_combined_gradcam(results, wavelengths, filename="gradcam_combined.png"):
    """
    Plots multiple spectra and overlays the Grad-CAM heatmap in subplots.
    """
    fig, axes = plt.subplots(len(results), 1, figsize=(14, 5 * len(results)), sharex=True)
    if len(results) == 1:
        axes = [axes]
        
    for i, res in enumerate(results):
        ax = axes[i]
        x_array = res['x_array']
        cam = res['cam']
        pred_prob = res['pred_prob']
        actual_class = res['actual_class']
        target_class = res['target_class']
        
        # Plot the original normalized flux
        ax.plot(wavelengths, x_array, color='black', linewidth=1.2, label='Spectrum')
        
        # Create a colored heatmap over the spectrum
        sc = ax.scatter(wavelengths, x_array, c=cam, cmap='jet', alpha=0.6, s=15, zorder=2)
        
        # Also add a colorbar to explain the colormap
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label('Grad-CAM Importance', rotation=270, labelpad=15)
        
        pred_label = "Type 2" if pred_prob >= 0.5 else "Type 1"
        actual_label = "Type 2" if actual_class == 1 else "Type 1"
        target_label = "Type 2" if target_class == 1 else "Type 1"
        
        title = (f"Grad-CAM (Target: {target_label}) | "
                 f"Pred: {pred_label} (Prob: {pred_prob:.3f}) | Actual: {actual_label}")
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('Normalized Flux', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right')
        
    axes[-1].set_xlabel('Rest Wavelength (Å)', fontsize=12)
    plt.tight_layout()
    
    plt.savefig(filename, dpi=300)
    print(f"Saved combined Grad-CAM plot to {filename}")
    plt.show()
def run_gradcam_analysis():
    config = load_config(os.path.join(BASE_DIR, 'config.yml'))
    
    # Setup Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Model
    model_path = os.path.join(BASE_DIR, config['model']['model_path'])
    model = SpectraNet(config)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded best trained model.")
    else:
        print("WARNING: Model weights not found. Running with untrained weights!")
        
    model = model.to(device)
    model.eval()
    
    # FIX: Point the Grad-CAM hook to the final SpectraBlock (index 2)
    target_layer = model.feature_extractor[2]
    grad_cam = GradCAM(model, target_layer)
    
    # Load some data to visualize
    data_path = os.path.join(BASE_DIR, config['data']['processed_catalog'])
    df_clean = pd.read_parquet(data_path)
    
    meta_cols = ['filename', 'agn_type', 'z', 'snr', 'obj_id']
    flux_cols = [c for c in df_clean.columns if c not in meta_cols]
    wavelengths = np.array(flux_cols, dtype=float)
    
    # Grab one Type 1 and one Type 2 spectrum
    type1_idx = df_clean[df_clean['agn_type'] == 1].index[0]
    type2_idx = df_clean[df_clean['agn_type'] == 2].index[0]
    
    results = []
    for idx in [type1_idx, type2_idx]:
        sample_row = df_clean.loc[idx]
        x_array = sample_row[flux_cols].values.astype(np.float32)
        
        # Convert to tensor (1, 1, Seq_Len)
        x_tensor = torch.tensor(x_array).unsqueeze(0).unsqueeze(0).to(device)
        
        # FIX: Apply the morphological filter so the network sees the flattened continuum
        x_tensor_processed = morphological_continuum_subtraction(x_tensor)
        
        actual_class = 1 if sample_row['agn_type'] == 2 else 0
        
        # Run Grad-CAM 
        cam, pred_prob = grad_cam(x_tensor_processed, target_class=actual_class)
        
        # Extract the processed 1D array to plot as the background spectrum
        processed_array = x_tensor_processed.squeeze().cpu().numpy()
        
        results.append({
            'x_array': processed_array,
            'cam': cam,
            'pred_prob': pred_prob,
            'actual_class': actual_class,
            'target_class': actual_class
        })
        
    # Plot combined
    plot_path = os.path.join(BASE_DIR, 'models', 'gradcam_combined.png')
    plot_combined_gradcam(results, wavelengths, filename=plot_path)




def plot_attention_map(model, spectrum, wavelengths, title="Spectral Attention Map"):
    """
    Captures and plots the attention weights from the first stage of SpectraNet.
    """
    model.eval()
    attention_weights = []

    # 1. Define the Hook Function
    def hook_fn(module, input, output):
        # The output of our SpatialAttention1d forward pass is (x * mask)
        # We want to capture the MASK itself.
        # We'll re-calculate it quickly from the input to the attention layer
        x = input[0]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        mask = module.sigmoid(module.conv(attn))
        attention_weights.append(mask.detach().cpu().numpy())

    # 2. Register hook to the first block's attention module
    # feature_extractor[0] is Stage 1
    handle = model.feature_extractor[0].attn.register_forward_hook(hook_fn)

    # 3. Forward Pass
    with torch.no_grad():
        # Ensure spectrum is (1, 1, 1024)
        input_tensor = spectrum.unsqueeze(0) if spectrum.ndim == 2 else spectrum
        _ = model(input_tensor)

    # Remove hook
    handle.remove()

    # 4. Process Weights
    # attn_map shape: [1, 1, 1024] -> flatten to [1024]
    attn_map = attention_weights[0].flatten()

    # 5. Plotting
    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Plot the original spectrum
    color_flux = 'black'
    ax1.set_xlabel('Rest Wavelength (Å)')
    ax1.set_ylabel('Normalized Flux', color=color_flux)
    ax1.plot(wavelengths, spectrum.flatten().cpu().numpy(), color=color_flux, alpha=0.7, label='Spectrum')
    ax1.tick_params(axis='y', labelcolor=color_flux)

    # Plot the attention map on a secondary axis
    ax2 = ax1.twinx()
    color_attn = 'crimson'
    ax2.set_ylabel('Attention Weight (0-1)', color=color_attn)
    ax2.fill_between(wavelengths, attn_map, color=color_attn, alpha=0.3, label='Attention')
    ax2.set_ylim(0, 1.1)
    ax2.tick_params(axis='y', labelcolor=color_attn)

    # Physics Markers (Optional but helpful for your thesis!)
    lines = {'Hβ': 4861, 'OIII': 5007, 'Hα': 6563}
    for name, wave in lines.items():
        ax1.axvline(x=wave, color='blue', linestyle='--', alpha=0.4)
        ax1.text(wave, ax1.get_ylim()[1]*0.9, name, color='blue', fontsize=10)

    plt.title(title)
    plt.show()

# Usage:
# spectrum, label = next(iter(test_loader))
# plot_attention_map(model, spectrum[0], master_grid)

def plot_transformer_attention(model, spec1, spec2, wavelengths):
    model.eval()
    
    def get_attn_1d(spec):
        # 1. Forward pass to get features and attention weights
        with torch.no_grad():
            x = spec.unsqueeze(0) if spec.ndim == 2 else spec # [1, 1, 1024]
            features = model.feature_extractor(x) # [1, 384, 64]
            x_trans = features.permute(0, 2, 1) # [1, 64, 384]
            _, weights = model.global_corr.mha(x_trans, x_trans, x_trans)
        
        attn_matrix = weights[0].cpu().numpy() # [64, 64]
        
        # 2. Sum along the query axis to see how much attention each token *received*
        token_importance = attn_matrix.sum(axis=0)
        
        # 3. Normalize between 0 and 1
        token_importance = token_importance - token_importance.min()
        token_importance = token_importance / (token_importance.max() + 1e-8)
        
        # 4. Interpolate from 64 tokens back to 1024 pixels
        token_tensor = torch.tensor(token_importance, dtype=torch.float32).view(1, 1, -1)
        attn_map = torch.nn.functional.interpolate(
            token_tensor, size=len(wavelengths), mode='linear', align_corners=False
        ).squeeze().numpy()
        
        return attn_map
        
    attn_map1 = get_attn_1d(spec1)
    attn_map2 = get_attn_1d(spec2)
    
    # 5. Plotting as 1D Overlays
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    def plot_overlay(ax, spec, attn_map, title):
        color_flux = 'black'
        ax.set_xlabel('Rest Wavelength (Å)')
        ax.set_ylabel('Normalized Flux', color=color_flux)
        ax.plot(wavelengths, spec.flatten().cpu().numpy(), color=color_flux, alpha=0.7, label='Spectrum')
        ax.tick_params(axis='y', labelcolor=color_flux)

        ax2 = ax.twinx()
        color_attn = 'crimson'
        ax2.set_ylabel('Received Global Attention (0-1)', color=color_attn)
        ax2.fill_between(wavelengths, attn_map, color=color_attn, alpha=0.3, label='MHA Attention')
        ax2.set_ylim(0, 1.1)
        ax2.tick_params(axis='y', labelcolor=color_attn)

        lines = {'Hβ': 4861, 'OIII': 5007, 'Hα': 6563}
        for name, wave in lines.items():
            ax.axvline(x=wave, color='blue', linestyle='--', alpha=0.4)
            ax.text(wave, ax.get_ylim()[1]*0.9, name, color='blue', fontsize=10)
        
        ax.set_title(title)
        
    plot_overlay(axes[0], spec1, attn_map1, "Type 1: Global Attention Overlay")
    plot_overlay(axes[1], spec2, attn_map2, "Type 2: Global Attention Overlay")
    
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    run_gradcam_analysis()
