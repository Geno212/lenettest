#!/usr/bin/env python3
"""
Create a TinyAlexNet model that can be synthesized on 16GB RAM.

The original AlexNet has massive FC layers (37M+ params) that cause OOM.
This version uses Global Average Pooling instead of FC layers.

Architecture:
- Conv layers similar to AlexNet (but smaller)
- Global Average Pooling instead of Flatten + huge FC
- Small FC output layer

Total params: ~500K instead of 60M
"""

import torch
import torch.nn as nn


class TinyAlexNet(nn.Module):
    """
    A smaller AlexNet-like model suitable for HLS synthesis on 16GB RAM.
    
    Changes from original AlexNet:
    - Reduced filter counts (64->32, 192->64, etc.)
    - Global Average Pooling instead of 3 massive FC layers
    - Single small FC layer for classification
    
    Input: 224x224x3 (like AlexNet)
    Output: 10 classes (configurable)
    """
    
    def __init__(self, num_classes=10):
        super(TinyAlexNet, self).__init__()
        
        # Convolutional layers (similar structure to AlexNet but smaller)
        self.features = nn.Sequential(
            # Conv1: 224x224x3 -> 55x55x32
            nn.Conv2d(3, 32, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # -> 27x27x32
            
            # Conv2: 27x27x32 -> 27x27x64
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # -> 13x13x64
            
            # Conv3: 13x13x64 -> 13x13x128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv4: 13x13x128 -> 13x13x128
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv5: 13x13x128 -> 13x13x64
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # -> 6x6x64
        )
        
        # Global Average Pooling - reduces 6x6x64 to 1x1x64
        # This eliminates the need for massive FC layers!
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Single small classifier layer: 64 -> num_classes
        # Instead of AlexNet's 9216->4096->4096->1000 (54M params!)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    # Create model
    model = TinyAlexNet(num_classes=10)
    model.eval()
    
    # Count parameters
    num_params = count_parameters(model)
    print(f"TinyAlexNet created!")
    print(f"  Parameters: {num_params:,} (~{num_params/1e6:.2f}M)")
    print(f"  Compare to AlexNet: ~61M parameters")
    print(f"  Reduction: {61e6/num_params:.0f}x smaller")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"\n  Input shape:  {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    
    # Save as TorchScript for Catapult
    scripted = torch.jit.script(model)
    save_path = "tiny_alexnet.pt"
    scripted.save(save_path)
    print(f"\n  Saved TorchScript model: {save_path}")
    
    # Also save state dict for flexibility
    torch.save(model.state_dict(), "tiny_alexnet_weights.pth")
    print(f"  Saved weights: tiny_alexnet_weights.pth")
    
    print("\n" + "="*50)
    print("To synthesize with Catapult:")
    print("="*50)
    print(f"""
$MGC_HOME/bin/python3 catapult_keras_flow.py tiny_alexnet.pt \\
    --input-shape 1 3 224 224 \\
    -n TinyAlexNet \\
    -o ~/catapult_tiny_alexnet \\
    --reuse-factor 64 \\
    --clock-period 10 \\
    --strategy Resource \\
    --precision "ac_fixed<8,4>"
""")


if __name__ == "__main__":
    main()
