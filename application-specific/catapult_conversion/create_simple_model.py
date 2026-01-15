#!/usr/bin/env python3
"""
Create a simple CNN model for testing Catapult AI NN flow.
This model is small enough to synthesize quickly and avoids edge cases.
"""

import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """
    Simple CNN: 
    - Input: 1x1x28x28 (like MNIST)
    - Conv 3x3 -> ReLU -> MaxPool
    - Conv 3x3 -> ReLU -> MaxPool  
    - Flatten -> Dense -> Output (10 classes)
    """
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Conv1: 1x28x28 -> 8x28x28 -> 8x14x14
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Conv2: 8x14x14 -> 16x14x14 -> 16x7x7
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def main():
    # Create and save the model
    model = SimpleCNN(num_classes=10)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Simple CNN created with {total_params:,} parameters")
    print(f"  (Compare to AlexNet's 57M+ parameters)")
    
    # Test forward pass
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Save as TorchScript (better for deployment)
    scripted = torch.jit.script(model)
    scripted.save("simple_cnn.pt")
    print(f"\nSaved: simple_cnn.pt")
    
    # Also save with torch.save for compatibility
    torch.save(model, "simple_cnn_full.pt")
    print(f"Saved: simple_cnn_full.pt")
    
    print("\nTo convert with Catapult:")
    print("  $MGC_HOME/bin/python3 catapult_keras_flow.py simple_cnn.pt \\")
    print("    --input-shape 1 1 28 28 --output ~/catapult_output --name SimpleCNN")


if __name__ == "__main__":
    main()
