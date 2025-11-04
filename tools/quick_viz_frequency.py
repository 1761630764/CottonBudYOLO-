"""
Quick visualization script for FrequencyDomainSeparator.

Usage:
    python quick_viz_frequency.py
    
This will generate visualizations in the 'visualizations' folder.
"""

import torch
import sys
from pathlib import Path

# Add project root to path (parent of tools directory)
sys.path.insert(0, str(Path(__file__).parent.parent))

print("\n" + "="*70)
print("Quick Frequency Domain Visualization")
print("="*70 + "\n")

try:
    # Check if matplotlib is available
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    
    from ultralytics.nn.modules.domain_modules import FrequencyDomainSeparator
    
    # Parameters
    channels = 64
    img_size = 40
    kernel_size = 3
    
    print(f"Configuration:")
    print(f"  Channels: {channels}")
    print(f"  Image size: {img_size}x{img_size}")
    print(f"  Kernel size: {kernel_size}\n")
    
    # Create module
    print("[1/5] Creating FrequencyDomainSeparator...")
    freq_sep = FrequencyDomainSeparator(channels=channels, kernel_size=kernel_size)
    freq_sep.eval()
    
    params = sum(p.numel() for p in freq_sep.parameters())
    print(f"  ✓ Module created with {params:,} parameters")
    
    # Create test input
    print("\n[2/5] Creating test input...")
    # Mix low and high frequency components
    low = torch.randn(1, channels, img_size // 4, img_size // 4)
    low = torch.nn.functional.interpolate(low, (img_size, img_size), mode='bilinear')
    high = torch.randn(1, channels, img_size, img_size) * 0.3
    x = low + high
    print(f"  ✓ Input shape: {x.shape}")
    
    # Forward pass
    print("\n[3/5] Processing through module...")
    with torch.no_grad():
        low_freq = freq_sep.low_pass(x)
        
        smoothed = torch.nn.functional.avg_pool2d(x, kernel_size, 1, kernel_size // 2)
        high_freq = freq_sep.high_pass(x - smoothed)
        
        output = freq_sep(x)
        
        weights = torch.nn.functional.softmax(freq_sep.fusion_weight, dim=0)
    
    print(f"  ✓ Low-freq weight: {weights[0].item():.3f}")
    print(f"  ✓ High-freq weight: {weights[1].item():.3f}")
    
    # Create visualization
    print("\n[4/5] Creating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle('Frequency Domain Separation (Channel 0)', fontsize=14, fontweight='bold')
    
    # Original
    im0 = axes[0, 0].imshow(x[0, 0].cpu().numpy(), cmap='viridis')
    axes[0, 0].set_title('Original Features')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
    
    # Low-frequency
    im1 = axes[0, 1].imshow(low_freq[0, 0].cpu().numpy(), cmap='viridis')
    axes[0, 1].set_title(f'Low-Frequency (w={weights[0]:.2f})')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    # High-frequency
    im2 = axes[1, 0].imshow(high_freq[0, 0].cpu().numpy(), cmap='viridis')
    axes[1, 0].set_title(f'High-Frequency (w={weights[1]:.2f})')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)
    
    # Output
    im3 = axes[1, 1].imshow(output[0, 0].cpu().numpy(), cmap='viridis')
    axes[1, 1].set_title('Final Output')
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)
    
    plt.tight_layout()
    
    # Save
    save_dir = Path("visualizations")
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / "quick_frequency_viz.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {save_path}")
    
    # Statistics
    print("\n[5/5] Statistics:")
    print(f"  Original  - mean: {x.mean():.4f}, std: {x.std():.4f}")
    print(f"  Low-freq  - mean: {low_freq.mean():.4f}, std: {low_freq.std():.4f}")
    print(f"  High-freq - mean: {high_freq.mean():.4f}, std: {high_freq.std():.4f}")
    print(f"  Output    - mean: {output.mean():.4f}, std: {output.std():.4f}")
    
    # Energy analysis
    orig_energy = (x ** 2).sum().item()
    low_energy = (low_freq ** 2).sum().item()
    high_energy = (high_freq ** 2).sum().item()
    output_energy = (output ** 2).sum().item()
    
    print(f"\n  Energy:")
    print(f"  Original:  {orig_energy:,.0f}")
    print(f"  Low-freq:  {low_energy:,.0f} ({low_energy/orig_energy*100:.1f}%)")
    print(f"  High-freq: {high_energy:,.0f} ({high_energy/orig_energy*100:.1f}%)")
    print(f"  Output:    {output_energy:,.0f} ({output_energy/orig_energy*100:.1f}%)")
    
    print("\n" + "="*70)
    print("✅ Visualization complete!")
    print(f"📁 Check: {save_path.absolute()}")
    print("="*70 + "\n")
    
    print("💡 Tip: For more detailed visualization, run:")
    print("   python visualize_frequency_features.py")
    print("\n   For interactive visualization, run:")
    print("   python interactive_frequency_viz.py\n")

except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("\nPlease install required packages:")
    print("  pip install matplotlib numpy")
    sys.exit(1)

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

