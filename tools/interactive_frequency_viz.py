"""
Interactive visualization tool for Frequency Domain Separator.

Provides an interactive interface to:
1. Adjust kernel size dynamically
2. Adjust reduction ratio
3. Visualize real-time changes
4. Compare different configurations
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from pathlib import Path
import sys

# Add project root to path (parent of tools directory)
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics.nn.modules.domain_modules import FrequencyDomainSeparator


class InteractiveFrequencyVisualizer:
    """Interactive visualizer for frequency domain separation."""
    
    def __init__(self, channels=64, img_size=40):
        """
        Initialize interactive visualizer.
        
        Args:
            channels (int): Number of feature channels.
            img_size (int): Spatial size of feature maps.
        """
        self.channels = channels
        self.img_size = img_size
        
        # Create test input
        self.x = self.create_test_input()
        
        # Initial parameters
        self.kernel_size = 3
        self.reduction = 4
        
        # Setup plot
        self.setup_plot()
        
    def create_test_input(self):
        """Create test input with mixed frequency content."""
        # Low-frequency base
        low_base = torch.randn(1, self.channels, self.img_size // 4, self.img_size // 4)
        low_base = torch.nn.functional.interpolate(
            low_base, size=(self.img_size, self.img_size),
            mode='bilinear', align_corners=False
        )
        
        # High-frequency details
        high_base = torch.randn(1, self.channels, self.img_size, self.img_size) * 0.3
        
        # Combine
        return low_base + high_base
    
    def setup_plot(self):
        """Setup interactive plot with sliders."""
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('Interactive Frequency Domain Visualization', 
                         fontsize=16, fontweight='bold')
        
        # Create subplots
        gs = self.fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3, 
                                   left=0.1, right=0.95, bottom=0.15, top=0.92)
        
        # Feature maps (top 2 rows)
        self.ax_orig = self.fig.add_subplot(gs[0, 0])
        self.ax_low = self.fig.add_subplot(gs[0, 1])
        self.ax_high = self.fig.add_subplot(gs[0, 2])
        self.ax_output = self.fig.add_subplot(gs[0, 3])
        
        self.ax_spectrum_orig = self.fig.add_subplot(gs[1, 0])
        self.ax_spectrum_low = self.fig.add_subplot(gs[1, 1])
        self.ax_spectrum_high = self.fig.add_subplot(gs[1, 2])
        self.ax_spectrum_output = self.fig.add_subplot(gs[1, 3])
        
        # Statistics (bottom row)
        self.ax_energy = self.fig.add_subplot(gs[2, 0:2])
        self.ax_stats = self.fig.add_subplot(gs[2, 2:4])
        
        # Add sliders
        ax_kernel = plt.axes([0.15, 0.08, 0.3, 0.03])
        ax_reduction = plt.axes([0.15, 0.04, 0.3, 0.03])
        
        self.slider_kernel = Slider(
            ax_kernel, 'Kernel Size', 3, 9, valinit=3, valstep=2,
            color='steelblue'
        )
        self.slider_reduction = Slider(
            ax_reduction, 'Reduction', 2, 16, valinit=4, valstep=2,
            color='forestgreen'
        )
        
        # Add update button
        ax_button = plt.axes([0.55, 0.04, 0.1, 0.075])
        self.button_update = Button(ax_button, 'Update', color='lightgray')
        
        # Add reset button
        ax_reset = plt.axes([0.67, 0.04, 0.1, 0.075])
        self.button_reset = Button(ax_reset, 'Reset', color='lightcoral')
        
        # Connect events
        self.button_update.on_clicked(self.update)
        self.button_reset.on_clicked(self.reset)
        
        # Initial plot
        self.update(None)
    
    def update(self, event):
        """Update visualization with new parameters."""
        # Get parameters
        self.kernel_size = int(self.slider_kernel.val)
        self.reduction = int(self.slider_reduction.val)
        
        print(f"\nUpdating: kernel_size={self.kernel_size}, reduction={self.reduction}")
        
        # Create module
        freq_sep = FrequencyDomainSeparator(
            channels=self.channels,
            kernel_size=self.kernel_size,
            reduction=self.reduction
        )
        freq_sep.eval()
        
        # Forward pass
        with torch.no_grad():
            low_freq = freq_sep.low_pass(self.x)
            
            smoothed = torch.nn.functional.avg_pool2d(
                self.x, self.kernel_size, 1, self.kernel_size // 2
            )
            high_freq_input = self.x - smoothed
            high_freq = freq_sep.high_pass(high_freq_input)
            
            output = freq_sep(self.x)
            
            weights = torch.nn.functional.softmax(freq_sep.fusion_weight, dim=0)
        
        # Count parameters
        params = sum(p.numel() for p in freq_sep.parameters())
        
        # Update plots
        self.plot_features(low_freq, high_freq, output, weights, params)
        
        plt.draw()
    
    def plot_features(self, low_freq, high_freq, output, weights, params):
        """Plot feature maps and analysis."""
        # Clear axes
        for ax in [self.ax_orig, self.ax_low, self.ax_high, self.ax_output,
                   self.ax_spectrum_orig, self.ax_spectrum_low, 
                   self.ax_spectrum_high, self.ax_spectrum_output,
                   self.ax_energy, self.ax_stats]:
            ax.clear()
        
        # Select channel 0 for visualization
        ch = 0
        
        # Plot spatial features
        self.ax_orig.imshow(self.x[0, ch].cpu().numpy(), cmap='viridis')
        self.ax_orig.set_title('Original')
        self.ax_orig.axis('off')
        
        self.ax_low.imshow(low_freq[0, ch].cpu().numpy(), cmap='viridis')
        self.ax_low.set_title(f'Low-Freq\n(weight={weights[0]:.3f})')
        self.ax_low.axis('off')
        
        self.ax_high.imshow(high_freq[0, ch].cpu().numpy(), cmap='viridis')
        self.ax_high.set_title(f'High-Freq\n(weight={weights[1]:.3f})')
        self.ax_high.axis('off')
        
        self.ax_output.imshow(output[0, ch].cpu().numpy(), cmap='viridis')
        self.ax_output.set_title('Output')
        self.ax_output.axis('off')
        
        # Plot frequency spectra
        self.plot_spectrum(self.ax_spectrum_orig, self.x[0, ch], 'Original')
        self.plot_spectrum(self.ax_spectrum_low, low_freq[0, ch], 'Low-Freq')
        self.plot_spectrum(self.ax_spectrum_high, high_freq[0, ch], 'High-Freq')
        self.plot_spectrum(self.ax_spectrum_output, output[0, ch], 'Output')
        
        # Plot energy distribution
        energies = {
            'Original': (self.x[0] ** 2).sum(dim=(1, 2)).cpu().numpy(),
            'Low-Freq': (low_freq[0] ** 2).sum(dim=(1, 2)).cpu().numpy(),
            'High-Freq': (high_freq[0] ** 2).sum(dim=(1, 2)).cpu().numpy(),
            'Output': (output[0] ** 2).sum(dim=(1, 2)).cpu().numpy(),
        }
        
        for name, energy in energies.items():
            self.ax_energy.plot(energy[:20], label=name, alpha=0.7, linewidth=2)
        self.ax_energy.set_xlabel('Channel Index (first 20)')
        self.ax_energy.set_ylabel('Energy')
        self.ax_energy.set_title('Channel Energy Distribution')
        self.ax_energy.legend()
        self.ax_energy.grid(True, alpha=0.3)
        
        # Plot statistics
        stats_text = [
            f"Kernel Size: {self.kernel_size}",
            f"Reduction: {self.reduction}",
            f"Parameters: {params:,}",
            f"",
            f"Low-freq weight: {weights[0]:.4f}",
            f"High-freq weight: {weights[1]:.4f}",
            f"",
            f"Original mean: {self.x.mean():.4f}",
            f"Original std: {self.x.std():.4f}",
            f"",
            f"Low-freq std: {low_freq.std():.4f}",
            f"High-freq std: {high_freq.std():.4f}",
            f"Output std: {output.std():.4f}",
        ]
        
        self.ax_stats.text(0.1, 0.95, '\n'.join(stats_text),
                          transform=self.ax_stats.transAxes,
                          fontsize=10, verticalalignment='top',
                          fontfamily='monospace',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        self.ax_stats.axis('off')
    
    def plot_spectrum(self, ax, tensor, title):
        """Plot frequency spectrum."""
        # Compute FFT
        feat = tensor.cpu().numpy()
        fft = np.fft.fft2(feat)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        magnitude_db = 20 * np.log10(magnitude + 1e-8)
        
        im = ax.imshow(magnitude_db, cmap='hot', aspect='auto')
        ax.set_title(f'{title}\nSpectrum')
        ax.axis('off')
    
    def reset(self, event):
        """Reset to initial parameters."""
        self.slider_kernel.reset()
        self.slider_reduction.reset()
        self.update(None)
    
    def show(self):
        """Show interactive plot."""
        plt.show()


def main():
    """Main function."""
    print("\n" + "="*70)
    print("Interactive Frequency Domain Visualizer")
    print("="*70)
    print("\nInstructions:")
    print("  - Use sliders to adjust kernel size and reduction ratio")
    print("  - Click 'Update' button to refresh visualization")
    print("  - Click 'Reset' button to restore default values")
    print("  - Close window to exit")
    print("="*70 + "\n")
    
    # Create visualizer
    viz = InteractiveFrequencyVisualizer(channels=64, img_size=40)
    viz.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nVisualization closed by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

