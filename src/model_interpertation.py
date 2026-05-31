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

class SignedType2GradCAM1D:
    """
    Signed Grad-CAM for a single-logit binary classifier.

    Convention:
        sigmoid(logit) = P(Type 2)

    IMPORTANT:
    We ALWAYS backpropagate from the raw Type 2 logit.

    Therefore:
        positive CAM  -> evidence toward Type 2
        negative CAM  -> evidence toward Type 1
        |CAM|         -> importance magnitude
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.forward_handle = self.target_layer.register_forward_hook(
            self._save_activation
        )
        self.backward_handle = self.target_layer.register_full_backward_hook(
            self._save_gradient
        )

    def _save_activation(self, module, inputs, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, x):
        """
        x: [1, 1, SeqLen]
        Returns a signed CAM normalized to [-1, 1].
        """
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        class_logit, _ = self.model(x)
        class_logit = class_logit.squeeze()

        pred_prob_type2 = torch.sigmoid(class_logit).item()
        pred_class = 1 if pred_prob_type2 >= 0.5 else 0

        # Uniform convention for ALL samples:
        # backprop from the Type 2 logit
        class_logit.backward(retain_graph=False)

        if self.gradients is None or self.activations is None:
            raise RuntimeError(
                "Gradients/activations were not captured. "
                "Check the target layer."
            )

        # Global-average-pool gradients over sequence dimension
        weights = self.gradients.mean(dim=2, keepdim=True)   # [1, C, 1]

        # Weighted combination of activations
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # [1, 1, L]

        # Upsample to input length
        cam = cam.detach().cpu()
        cam = F.interpolate(
            cam,
            size=x.shape[-1],
            mode='linear',
            align_corners=False
        )

        cam = cam.squeeze().numpy()

        # Signed normalization: preserve sign
        max_abs = np.max(np.abs(cam)) + 1e-8
        cam = cam / max_abs

        return {
            "cam": cam,
            "pred_prob_type2": pred_prob_type2,
            "pred_class": pred_class
        }


def get_flux_columns_and_wavelengths(df):
    meta_cols = ['filename', 'agn_type', 'z', 'snr', 'obj_id']
    flux_cols = [c for c in df.columns if c not in meta_cols]
    wavelengths = np.array(flux_cols, dtype=float)
    return flux_cols, wavelengths


def collect_signed_cams_by_true_class(
    df,
    flux_cols,
    device,
    grad_cam,
    n_per_class=50,
    only_correct=True,
    random_state=42
):
    """
    Collect signed CAMs for true Type 1 and true Type 2 spectra.

    Returns:
        cams_by_class = {
            0: [cam1, cam2, ...],  # true Type 1
            1: [cam1, cam2, ...]   # true Type 2
        }
    """
    cams_by_class = {0: [], 1: []}

    df_shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    for _, row in df_shuffled.iterrows():
        actual_class = 1 if row['agn_type'] == 2 else 0

        if len(cams_by_class[actual_class]) >= n_per_class:
            if len(cams_by_class[0]) >= n_per_class and len(cams_by_class[1]) >= n_per_class:
                break
            continue

        x_array = row[flux_cols].values.astype(np.float32)
        x_tensor = torch.tensor(x_array).view(1, 1, -1).to(device)

        #x_processed = morphological_continuum_subtraction(x_tensor)

        result = grad_cam(x_tensor)

        if only_correct and result["pred_class"] != actual_class:
            continue

        cams_by_class[actual_class].append(result["cam"])

    print(f"Collected {len(cams_by_class[0])} correct Type 1 CAMs")
    print(f"Collected {len(cams_by_class[1])} correct Type 2 CAMs")

    return cams_by_class


def plot_signed_cam_subplots(
    cams_by_class,
    wavelengths,
    filename=None,
    title="Average Signed Grad-CAM by True Class"
):
    """
    Two-panel plot:
        Top    -> true Type 1 spectra
        Bottom -> true Type 2 spectra

    Uniform sign convention in BOTH panels:
        positive -> evidence toward Type 2
        negative -> evidence toward Type 1
    """
    fig, axes = plt.subplots(
        2, 1,
        figsize=(14, 9),
        sharex=True,
        sharey=True
    )

    class_names = {
        0: "True Type 1 spectra",
        1: "True Type 2 spectra"
    }

    spectral_lines = {
        "Hβ": 4861,
        "[O III]": 5007,
        "Hα": 6563
    }

    all_means = []
    all_stds = []

    # First pass to compute shared y-limits
    for class_id in [0, 1]:
        cams = np.array(cams_by_class[class_id])
        if len(cams) == 0:
            all_means.append(None)
            all_stds.append(None)
            continue

        mean_cam = cams.mean(axis=0)
        std_cam = cams.std(axis=0)
        all_means.append(mean_cam)
        all_stds.append(std_cam)

    # Shared y-limits
    ymin, ymax = 1e9, -1e9
    for mean_cam, std_cam in zip(all_means, all_stds):
        if mean_cam is None:
            continue
        ymin = min(ymin, np.min(mean_cam - std_cam))
        ymax = max(ymax, np.max(mean_cam + std_cam))

    if ymin == 1e9:
        ymin, ymax = -1.0, 1.0

    pad = 0.08 * max(abs(ymin), abs(ymax), 1.0)
    ymin -= pad
    ymax += pad

    for ax, class_id, mean_cam, std_cam in zip(axes, [0, 1], all_means, all_stds):
        if mean_cam is None:
            ax.set_title(f"{class_names[class_id]} (no samples)")
            continue

        ax.plot(
            wavelengths,
            mean_cam,
            linewidth=2.2,
            label=class_names[class_id]
        )

        ax.fill_between(
            wavelengths,
            mean_cam - std_cam,
            mean_cam + std_cam,
            alpha=0.2
        )

        ax.axhline(0, color='black', linestyle='-', linewidth=1.0, alpha=0.8)

        for name, wave in spectral_lines.items():
            ax.axvline(wave, color='steelblue', linestyle='--', alpha=0.6)
            ax.text(
                wave,
                ymax * 0.92,
                name,
                rotation=90,
                verticalalignment='top',
                horizontalalignment='left',
                color='black',
                fontsize=11
            )

        ax.set_ylim(ymin, ymax)
        ax.set_ylabel("Signed Grad-CAM")
        ax.set_title(class_names[class_id], fontsize=13, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.35)
        ax.legend(loc='best')

    axes[-1].set_xlabel("Rest Wavelength (Å)")

    fig.suptitle(title, fontsize=15, fontweight='bold', y=0.98)

    # Helpful annotation with the uniform convention
    fig.text(
        0.5, 0.02,
        "Positive values = evidence toward Type 2   |   Negative values = evidence toward Type 1",
        ha='center',
        fontsize=11
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    if filename is not None:
        plt.savefig(filename, dpi=300)
        print(f"Saved subplot Grad-CAM figure to: {filename}")

    plt.show()


def run_signed_gradcam_subplot_analysis(
    n_per_class=50,
    only_correct=True
):
    config = load_config(os.path.join(BASE_DIR, 'config.yml'))

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Load model
    model_path = os.path.join(BASE_DIR, config['model']['model_path'])
    model = SpectraNet(config)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print("Loaded trained model.")

    # Better Grad-CAM hook target than the whole block
    target_layer = model.feature_extractor[2].fusion
    grad_cam = SignedType2GradCAM1D(model, target_layer)

    # Load data
    data_path = os.path.join(BASE_DIR, config['data']['processed_catalog'])
    df_clean = pd.read_parquet(data_path)

    flux_cols, wavelengths = get_flux_columns_and_wavelengths(df_clean)

    cams_by_class = collect_signed_cams_by_true_class(
        df=df_clean,
        flux_cols=flux_cols,
        device=device,
        grad_cam=grad_cam,
        n_per_class=n_per_class,
        only_correct=only_correct,
        random_state=42
    )

    save_path = os.path.join(BASE_DIR, 'models', 'signed_gradcam_true_class_subplots.png')

    plot_signed_cam_subplots(
        cams_by_class=cams_by_class,
        wavelengths=wavelengths,
        filename=save_path,
        title="Average Signed Grad-CAM by True Class"
    )

    grad_cam.remove_hooks()

    return cams_by_class




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
        

            print(f"DEBUG: Raw weights shape: {weights.shape}")
                
            # PyTorch's native MHA can sometimes return [Batch, Num_Heads, Tokens, Tokens]
            # If we don't catch this, the subsequent math will sum across the wrong axis!
            if weights.ndim == 4:
                print("DEBUG: 4D Tensor detected. Averaging across attention heads...")
                weights = weights.mean(dim=1)

            attn_matrix = weights[0].cpu().numpy() # [64, 64]
            
            # 2. Sum along the query axis to see how much attention each token *received*
            token_importance = attn_matrix.sum(axis=0)
            
            if np.isnan(token_importance).any():
                print("WARNING: NaN values detected in attention matrix!")
            if token_importance.max() == 0:
                print("WARNING: Token importance is completely flat (zeros).")


            # 3. Normalize between 0 and 1
            token_importance = token_importance - token_importance.min()
            token_importance = token_importance / (token_importance.max() + 1e-8)
            
            # 4. Interpolate from 64 tokens back to 1024 pixels
            token_tensor = torch.tensor(token_importance, dtype=torch.float32).view(1, 1, -1)
            attn_map = torch.nn.functional.interpolate(
                token_tensor, size=len(wavelengths), mode='linear', align_corners=False
            ).squeeze().numpy()
            
            max_idx = attn_map.argmax()
            max_wave = wavelengths[max_idx]
            print(f"DEBUG: Maximum attention localized at exactly: {max_wave:.1f} Å")


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
    run_signed_gradcam_subplot_analysis()
