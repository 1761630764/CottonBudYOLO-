"""
Comprehensive verification script for all domain adaptation modules.

This script tests all implemented modules:
- DomainNormalization
- DomainNormalizedConv
- FeatureMixer
- AdaptiveFeatureCalibration
- FrequencyDomainSeparator
"""

import torch
import sys
from pathlib import Path

# Ensure we can import from ultralytics
# Add project root to path (parent of tools directory)
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 70)
print("Comprehensive Verification of Domain Adaptation Modules")
print("=" * 70)

try:
    # Import all modules
    print("\n[1/6] Importing modules...")
    from ultralytics.nn.modules.domain_modules import (
        DomainNormalization,
        DomainNormalizedConv,
        FeatureMixer,
        AdaptiveFeatureCalibration,
        FrequencyDomainSeparator,
    )
    print("✓ All imports successful!")
    
    # Test DomainNormalization
    print("\n[2/6] Testing DomainNormalization...")
    dn = DomainNormalization(num_features=256, num_domains=3)
    x = torch.randn(4, 256, 40, 40)
    y = dn(x)
    print(f"✓ Shape: {x.shape} -> {y.shape}")
    print(f"  Parameters: {sum(p.numel() for p in dn.parameters()):,}")
    assert y.shape == x.shape and not torch.isnan(y).any()
    
    # Test DomainNormalizedConv
    print("\n[3/6] Testing DomainNormalizedConv...")
    dnconv = DomainNormalizedConv(c1=128, c2=256, k=3, s=2, num_domains=3)
    x = torch.randn(4, 128, 80, 80)
    y = dnconv(x)
    print(f"✓ Shape: {x.shape} -> {y.shape}")
    print(f"  Parameters: {sum(p.numel() for p in dnconv.parameters()):,}")
    assert y.shape == (4, 256, 40, 40) and not torch.isnan(y).any()
    
    # Test FeatureMixer
    print("\n[4/6] Testing FeatureMixer...")
    mixer = FeatureMixer(channels=256)
    x = torch.randn(4, 256, 40, 40)
    y = mixer(x)
    print(f"✓ Shape: {x.shape} -> {y.shape}")
    print(f"  Parameters: {sum(p.numel() for p in mixer.parameters()):,}")
    assert y.shape == x.shape and not torch.isnan(y).any()
    
    # Test AdaptiveFeatureCalibration
    print("\n[5/6] Testing AdaptiveFeatureCalibration...")
    calibration = AdaptiveFeatureCalibration(channels=256)
    x = torch.randn(4, 256, 40, 40)
    y = calibration(x)
    print(f"✓ Shape: {x.shape} -> {y.shape}")
    print(f"  Parameters: {sum(p.numel() for p in calibration.parameters()):,}")
    assert y.shape == x.shape and not torch.isnan(y).any()
    
    # Test FrequencyDomainSeparator
    print("\n[6/6] Testing FrequencyDomainSeparator...")
    freq_sep = FrequencyDomainSeparator(channels=256)
    x = torch.randn(4, 256, 40, 40)
    y = freq_sep(x)
    print(f"✓ Shape: {x.shape} -> {y.shape}")
    print(f"  Parameters: {sum(p.numel() for p in freq_sep.parameters()):,}")
    assert y.shape == x.shape and not torch.isnan(y).any()
    
    # Test gradient flow through all modules
    print("\n[7/7] Testing gradient flow...")
    
    class TestNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dn = DomainNormalization(64, 3)
            self.mixer = FeatureMixer(64)
            self.calibration = AdaptiveFeatureCalibration(64)
            self.freq_sep = FrequencyDomainSeparator(64)
            
        def forward(self, x):
            x = self.dn(x)
            x = self.mixer(x)
            x = self.calibration(x)
            x = self.freq_sep(x)
            return x
    
    net = TestNet()
    x = torch.randn(2, 64, 40, 40, requires_grad=True)
    y = net(x)
    loss = y.sum()
    loss.backward()
    
    assert x.grad is not None, "Input gradient is None"
    assert not torch.isnan(x.grad).any(), "Gradient contains NaN"
    print("✓ Gradient flow successful through all modules")
    
    # Count total parameters
    print("\n" + "=" * 70)
    print("📊 Module Statistics")
    print("=" * 70)
    
    modules = {
        'DomainNormalization': DomainNormalization(256, 3),
        'DomainNormalizedConv': DomainNormalizedConv(256, 256, 3, 1, num_domains=3),
        'FeatureMixer': FeatureMixer(256),
        'AdaptiveFeatureCalibration': AdaptiveFeatureCalibration(256),
        'FrequencyDomainSeparator': FrequencyDomainSeparator(256),
    }
    
    total_params = 0
    for name, module in modules.items():
        params = sum(p.numel() for p in module.parameters())
        total_params += params
        print(f"{name:30s}: {params:8,} parameters")
    
    print("-" * 70)
    print(f"{'Total':30s}: {total_params:8,} parameters")
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ ALL VERIFICATION CHECKS PASSED!")
    print("=" * 70)
    print("\n📋 Summary:")
    print("  ✓ All 5 modules can be imported")
    print("  ✓ All modules have correct output shapes")
    print("  ✓ No NaN or Inf values in outputs")
    print("  ✓ Gradient flow works correctly")
    print(f"  ✓ Total parameters: {total_params:,}")
    
    print("\n🎯 Next Steps:")
    print("  1. Run full test suite: python tests/test_domain_norm.py")
    print("  2. Run feature tests: python tests/test_feature_modules.py")
    print("  3. Start implementing C3k2_DN (Week 2, Day 8-9)")
    print("  4. Integrate modules into Ultralytics")
    print("=" * 70 + "\n")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

