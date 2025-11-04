"""
Visualization tool for Frequency Domain Separator.

This script visualizes:
1. Original features
2. Low-frequency components
3. High-frequency components
4. Fused features
5. Frequency spectrum analysis
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path (parent of tools directory)
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics.nn.modules.domain_modules import FrequencyDomainSeparator


def visualize_frequency_separation(
    channels=64,
    img_size=40,
    save_dir="visualizations",
    kernel_size=3,
):
    """
    Visualize frequency domain separation process.
    
    Args:
        channels (int): Number of channels to test.
        img_size (int): Spatial size of feature maps.
        save_dir (str): Directory to save visualizations.
        kernel_size (int): Kernel size for frequency separator.
    """
    print("="*70)
    print("Frequency Domain Feature Visualization")
    print("="*70)
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)
    
    # Create module
    freq_sep = FrequencyDomainSeparator(channels=channels, kernel_size=kernel_size)
    freq_sep.eval()
    
    # Create test input with mixed frequency content
    print(f"\n[1/6] Creating test input ({channels} channels, {img_size}x{img_size})...")
    x = create_mixed_frequency_input(1, channels, img_size)
    
    # Forward pass and extract components
    print("\n[2/6] Extracting frequency components...")
    with torch.no_grad():
        # Get low-frequency component
        low_freq = freq_sep.low_pass(x)
        
        # Get high-frequency component
        smoothed = torch.nn.functional.avg_pool2d(
            x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )
        high_freq_input = x - smoothed
        high_freq = freq_sep.high_pass(high_freq_input)
        
        # Get fusion weights
        weights = torch.nn.functional.softmax(freq_sep.fusion_weight, dim=0)
        
        # Get fused output
        fused = low_freq * weights[0] + high_freq * weights[1]
        refined = freq_sep.refine(fused)
        output = refined + x
    
    print(f"  Low-freq weight: {weights[0].item():.3f}")
    print(f"  High-freq weight: {weights[1].item():.3f}")
    
    # Visualize spatial features
    print("\n[3/6] Visualizing spatial features...")
    visualize_spatial_features(
        x, low_freq, high_freq, fused, output,
        save_path / "spatial_features.png"
    )
    
    # Visualize frequency spectrum
    print("\n[4/6] Computing frequency spectrum...")
    visualize_frequency_spectrum(
        x, low_freq, high_freq, fused, output,
        save_path / "frequency_spectrum.png"
    )
    
    # Visualize statistics
    print("\n[5/6] Analyzing statistics...")
    visualize_statistics(
        x, low_freq, high_freq, fused, output,
        save_path / "statistics.png"
    )
    
    # Visualize channel-wise energy distribution
    print("\n[6/6] Visualizing energy distribution...")
    visualize_energy_distribution(
        x, low_freq, high_freq, output,
        save_path / "energy_distribution.png"
    )
    
    print(f"\n{'='*70}")
    print(f"✅ Visualization complete!")
    print(f"📁 Saved to: {save_path.absolute()}")
    print(f"{'='*70}\n")
    
    return save_path


def create_mixed_frequency_input(batch_size, channels, size):
    """
    Create input with mixed frequency content.
    
    Combines:
    - Low-frequency: smooth gradients
    - High-frequency: sharp edges and textures
    """
    # Low-frequency base (smooth)
    low_freq_base = torch.randn(batch_size, channels, size // 4, size // 4)
    low_freq_base = torch.nn.functional.interpolate(
        low_freq_base, size=(size, size), mode='bilinear', align_corners=False
    )
    
    # High-frequency details (sharp)
    high_freq_base = torch.randn(batch_size, channels, size, size) * 0.5
    
    # Combine
    mixed = low_freq_base + high_freq_base
    
    return mixed


def visualize_spatial_features(x, low_freq, high_freq, fused, output, save_path):
    """Visualize spatial feature maps."""
    
    # Select first channel for visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Spatial Feature Visualization (Channel 0)', fontsize=16, fontweight='bold')
    
    # Original
    im1 = axes[0, 0].imshow(x[0, 0].cpu().numpy(), cmap='viridis')
    axes[0, 0].set_title('Original Features')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Low-frequency
    im2 = axes[0, 1].imshow(low_freq[0, 0].cpu().numpy(), cmap='viridis')
    axes[0, 1].set_title('Low-Frequency (Global)')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # High-frequency
    im3 = axes[0, 2].imshow(high_freq[0, 0].cpu().numpy(), cmap='viridis')
    axes[0, 2].set_title('High-Frequency (Details)')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Fused
    im4 = axes[1, 0].imshow(fused[0, 0].cpu().numpy(), cmap='viridis')
    axes[1, 0].set_title('Fused Features')
    axes[1, 0].axis('off')
    plt.colorbar(im4, ax=axes[1, 0])
    
    # Output (with residual)
    im5 = axes[1, 1].imshow(output[0, 0].cpu().numpy(), cmap='viridis')
    axes[1, 1].set_title('Final Output (Fused + Residual)')
    axes[1, 1].axis('off')
    plt.colorbar(im5, ax=axes[1, 1])
    
    # Residual contribution
    residual = output[0, 0] - fused[0, 0]
    im6 = axes[1, 2].imshow(residual.cpu().numpy(), cmap='RdBu')
    axes[1, 2].set_title('Residual Contribution')
    axes[1, 2].axis('off')
    plt.colorbar(im6, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved spatial visualization: {save_path}")


def visualize_frequency_spectrum(x, low_freq, high_freq, fused, output, save_path):
    """Visualize frequency spectrum using FFT."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Frequency Spectrum Analysis (FFT)', fontsize=16, fontweight='bold')
    
    def compute_spectrum(tensor):
        """Compute 2D FFT magnitude spectrum."""
        # Average over batch and channels
        feat = tensor[0].mean(0).cpu().numpy()
        
        # Compute FFT
        fft = np.fft.fft2(feat)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        # Log scale for better visualization
        magnitude_db = 20 * np.log10(magnitude + 1e-8)
        
        return magnitude_db
    
    # Compute spectra
    spectra = {
        'Original': compute_spectrum(x),
        'Low-Frequency': compute_spectrum(low_freq),
        'High-Frequency': compute_spectrum(high_freq),
        'Fused': compute_spectrum(fused),
        'Output': compute_spectrum(output),
    }
    
    # Plot spectra
    positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    titles = list(spectra.keys())
    
    for (i, j), title in zip(positions, titles):
        im = axes[i, j].imshow(spectra[title], cmap='hot', aspect='auto')
        axes[i, j].set_title(f'{title}\n(Brighter = Higher Energy)')
        axes[i, j].axis('off')
        plt.colorbar(im, ax=axes[i, j], label='Magnitude (dB)')
    
    # Add radial profile comparison
    axes[1, 2].set_title('Radial Frequency Profile')
    for name, spectrum in spectra.items():
        profile = compute_radial_profile(spectrum)
        axes[1, 2].plot(profile, label=name, linewidth=2, alpha=0.7)
    axes[1, 2].set_xlabel('Frequency (radial distance from center)')
    axes[1, 2].set_ylabel('Average Magnitude (dB)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved frequency spectrum: {save_path}")


def compute_radial_profile(spectrum):
    """Compute radial average of 2D spectrum."""
    h, w = spectrum.shape
    center_h, center_w = h // 2, w // 2
    
    # Create coordinate grids
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center_w)**2 + (y - center_h)**2)
    r = r.astype(int)
    
    # Compute radial average
    tbin = np.bincount(r.ravel(), spectrum.ravel())
    nr = np.bincount(r.ravel())
    radial_prof = tbin / (nr + 1e-8)
    
    return radial_prof


def visualize_statistics(x, low_freq, high_freq, fused, output, save_path):
    """Visualize statistical properties."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Statistical Analysis', fontsize=16, fontweight='bold')
    
    # Collect data
    data = {
        'Original': x,
        'Low-Freq': low_freq,
        'High-Freq': high_freq,
        'Fused': fused,
        'Output': output,
    }
    
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    # 1. Mean and Std
    means = [d.mean().item() for d in data.values()]
    stds = [d.std().item() for d in data.values()]
    
    x_pos = np.arange(len(data))
    axes[0, 0].bar(x_pos, means, color=colors, alpha=0.6, label='Mean')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(data.keys(), rotation=45, ha='right')
    axes[0, 0].set_ylabel('Mean Value')
    axes[0, 0].set_title('Mean Values')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].bar(x_pos, stds, color=colors, alpha=0.6, label='Std')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(data.keys(), rotation=45, ha='right')
    axes[0, 1].set_ylabel('Standard Deviation')
    axes[0, 1].set_title('Standard Deviations')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 2. Distribution histograms
    for name, tensor, color in zip(data.keys(), data.values(), colors):
        values = tensor.flatten().cpu().numpy()
        axes[1, 0].hist(values, bins=50, alpha=0.5, label=name, color=color, density=True)
    axes[1, 0].set_xlabel('Value')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Value Distributions')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 3. Variance across channels
    variances = []
    for name, tensor in data.items():
        # Variance per channel
        var_per_channel = tensor[0].var(dim=(1, 2)).cpu().numpy()
        variances.append(var_per_channel)
    
    # Plot box plot
    axes[1, 1].boxplot(variances, labels=data.keys())
    axes[1, 1].set_xticklabels(data.keys(), rotation=45, ha='right')
    axes[1, 1].set_ylabel('Variance')
    axes[1, 1].set_title('Channel-wise Variance Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved statistics: {save_path}")


def visualize_energy_distribution(x, low_freq, high_freq, output, save_path):
    """Visualize energy distribution across channels."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Energy Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Compute energy (L2 norm) per channel
    def compute_channel_energy(tensor):
        # Energy = sum of squared values
        energy = (tensor[0] ** 2).sum(dim=(1, 2)).cpu().numpy()
        return energy
    
    energies = {
        'Original': compute_channel_energy(x),
        'Low-Frequency': compute_channel_energy(low_freq),
        'High-Frequency': compute_channel_energy(high_freq),
        'Output': compute_channel_energy(output),
    }
    
    # 1. Energy per channel
    for name, energy in energies.items():
        axes[0, 0].plot(energy, label=name, alpha=0.7, linewidth=2)
    axes[0, 0].set_xlabel('Channel Index')
    axes[0, 0].set_ylabel('Energy')
    axes[0, 0].set_title('Energy per Channel')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Total energy comparison
    total_energies = [e.sum() for e in energies.values()]
    axes[0, 1].bar(energies.keys(), total_energies, color=['blue', 'green', 'red', 'orange'], alpha=0.6)
    axes[0, 1].set_ylabel('Total Energy')
    axes[0, 1].set_title('Total Energy Comparison')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Energy ratio (low vs high)
    low_energy = energies['Low-Frequency']
    high_energy = energies['High-Frequency']
    ratio = low_energy / (high_energy + 1e-8)
    
    axes[1, 0].plot(ratio, color='purple', linewidth=2)
    axes[1, 0].axhline(y=1.0, color='red', linestyle='--', label='Equal energy')
    axes[1, 0].set_xlabel('Channel Index')
    axes[1, 0].set_ylabel('Low / High Energy Ratio')
    axes[1, 0].set_title('Low-to-High Frequency Energy Ratio')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Cumulative energy
    for name, energy in energies.items():
        sorted_energy = np.sort(energy)[::-1]
        cumulative = np.cumsum(sorted_energy) / sorted_energy.sum()
        axes[1, 1].plot(cumulative, label=name, alpha=0.7, linewidth=2)
    axes[1, 1].set_xlabel('Number of Channels (sorted by energy)')
    axes[1, 1].set_ylabel('Cumulative Energy Ratio')
    axes[1, 1].set_title('Cumulative Energy Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved energy distribution: {save_path}")


def compare_kernel_sizes(save_dir="visualizations"):
    """Compare different kernel sizes for frequency separation."""
    
    print("\n" + "="*70)
    print("Comparing Different Kernel Sizes")
    print("="*70)
    
    save_path = Path(save_dir)
    channels = 64
    img_size = 40
    
    # Test different kernel sizes
    kernel_sizes = [3, 5, 7]
    
    # Create test input
    x = create_mixed_frequency_input(1, channels, img_size)
    
    fig, axes = plt.subplots(len(kernel_sizes), 4, figsize=(16, 4*len(kernel_sizes)))
    fig.suptitle('Kernel Size Comparison', fontsize=16, fontweight='bold')
    
    for idx, k in enumerate(kernel_sizes):
        print(f"\n  Testing kernel size: {k}")
        
        freq_sep = FrequencyDomainSeparator(channels=channels, kernel_size=k)
        freq_sep.eval()
        
        with torch.no_grad():
            low_freq = freq_sep.low_pass(x)
            smoothed = torch.nn.functional.avg_pool2d(x, k, 1, k//2)
            high_freq = freq_sep.high_pass(x - smoothed)
            output = freq_sep(x)
        
        # Plot
        row = axes[idx] if len(kernel_sizes) > 1 else axes
        
        row[0].imshow(x[0, 0].cpu().numpy(), cmap='viridis')
        row[0].set_title(f'Original (k={k})')
        row[0].axis('off')
        
        row[1].imshow(low_freq[0, 0].cpu().numpy(), cmap='viridis')
        row[1].set_title(f'Low-Freq (k={k})')
        row[1].axis('off')
        
        row[2].imshow(high_freq[0, 0].cpu().numpy(), cmap='viridis')
        row[2].set_title(f'High-Freq (k={k})')
        row[2].axis('off')
        
        row[3].imshow(output[0, 0].cpu().numpy(), cmap='viridis')
        row[3].set_title(f'Output (k={k})')
        row[3].axis('off')
    
    plt.tight_layout()
    save_file = save_path / "kernel_size_comparison.png"
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n  ✓ Saved comparison: {save_file}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Frequency Domain Feature Visualization Tool")
    print("="*70 + "\n")
    
    # Main visualization
    save_dir = visualize_frequency_separation(
        channels=64,
        img_size=40,
        kernel_size=3,
    )
    
    # Compare kernel sizes
    compare_kernel_sizes(save_dir=save_dir)
    
    print("\n" + "="*70)
    print("✅ All visualizations complete!")
    print(f"📁 Check the '{save_dir}' directory for all plots")
    print("="*70 + "\n")
    
    print("Generated files:")
    for file in sorted(Path(save_dir).glob("*.png")):
        print(f"  - {file.name}")

