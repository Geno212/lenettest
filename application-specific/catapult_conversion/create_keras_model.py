#!/usr/bin/env python3
"""
Create a Keras model directly for Catapult AI NN.
This follows the EXACT pattern from Catapult's notebook.txt tutorial.

The key is using channels-last (NHWC) format which Catapult's hls4ml supports
with io_stream mode.

Usage on EC2:
    $MGC_HOME/bin/python3 create_keras_model.py
    
Then run conversion:
    $MGC_HOME/bin/python3 catapult_direct_keras.py
"""

import os
import sys

# Ensure we use Catapult's Python environment
MGC_HOME = os.environ.get('MGC_HOME', '/data/tools/catapult/Mgc_home')
os.environ['MGC_HOME'] = MGC_HOME

def create_simple_cnn():
    """
    Create a simple CNN in Keras with CHANNELS-LAST format (NHWC).
    This is the format Catapult's hls4ml expects.
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    # Use channels_last (NHWC) - this is TensorFlow's default
    tf.keras.backend.set_image_data_format('channels_last')
    
    # Input shape: (height, width, channels) - NHWC format
    # For MNIST-like: 28x28x1
    input_shape = (28, 28, 1)
    
    model = keras.Sequential([
        # Input layer
        keras.Input(shape=input_shape, name='input1'),
        
        # Conv1: 28x28x1 -> 28x28x8 (with padding='same')
        layers.Conv2D(8, kernel_size=3, strides=1, padding='same', 
                      use_bias=True, name='conv2d1'),
        layers.Activation('relu', name='relu1'),
        
        # Pool1: 28x28x8 -> 14x14x8
        layers.MaxPooling2D(pool_size=2, strides=2, name='maxpool1'),
        
        # Conv2: 14x14x8 -> 14x14x16 (with padding='same')
        layers.Conv2D(16, kernel_size=3, strides=1, padding='same',
                      use_bias=True, name='conv2d2'),
        layers.Activation('relu', name='relu2'),
        
        # Pool2: 14x14x16 -> 7x7x16
        layers.MaxPooling2D(pool_size=2, strides=2, name='maxpool2'),
        
        # Flatten: 7x7x16 = 784
        layers.Flatten(name='flatten1'),
        
        # Dense: 784 -> 64
        layers.Dense(64, use_bias=True, name='dense1'),
        layers.Activation('relu', name='relu3'),
        
        # Output: 64 -> 10
        layers.Dense(10, use_bias=True, name='dense2'),
        layers.Activation('softmax', name='softmax1'),
    ], name='SimpleCNN')
    
    return model


def create_minimal_cnn():
    """
    Create a minimal CNN matching Catapult's notebook example EXACTLY.
    Based on notebook.txt "Conv2D, Batchnorm and Dense Model 3"
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    tf.keras.backend.set_image_data_format('channels_last')
    
    # Match notebook: 14x14x1 input (downsampled MNIST)
    input_shape = (14, 14, 1)
    
    model = keras.Sequential([
        keras.Input(shape=input_shape, name='input1'),
        
        # Single conv with stride 3: 14x14x1 -> 4x4x5
        layers.Conv2D(5, kernel_size=5, strides=3, padding='valid',
                      use_bias=True, name='conv2d1'),
        layers.BatchNormalization(name='batchnorm1'),
        layers.Activation('relu', name='relu1'),
        
        # Flatten: 4x4x5 = 80
        layers.Flatten(name='flatten1'),
        
        # Dense: 80 -> 10
        layers.Dense(10, use_bias=True, name='dense1'),
        layers.Activation('softmax', name='softmax1'),
    ], name='MinimalCNN')
    
    return model


def main():
    import tensorflow as tf
    import numpy as np
    
    print("="*60)
    print("Creating Keras Models for Catapult AI NN")
    print("="*60)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Image data format: {tf.keras.backend.image_data_format()}")
    print()
    
    # Create the simple model
    print("Creating SimpleCNN (28x28x1 input)...")
    simple_model = create_simple_cnn()
    simple_model.summary()
    
    # Test forward pass
    dummy_input = np.random.randn(1, 28, 28, 1).astype(np.float32)
    output = simple_model(dummy_input)
    print(f"\nTest: input {dummy_input.shape} -> output {output.shape}")
    
    # Save model
    simple_model.save('simple_cnn_keras.h5')
    print("\nSaved: simple_cnn_keras.h5")
    
    # Also save as JSON + weights (like Catapult expects)
    with open('simple_cnn_keras.json', 'w') as f:
        f.write(simple_model.to_json())
    simple_model.save_weights('simple_cnn_keras_weights.h5')
    print("Saved: simple_cnn_keras.json + simple_cnn_keras_weights.h5")
    
    print()
    print("-"*60)
    print()
    
    # Create the minimal model (matches notebook exactly)
    print("Creating MinimalCNN (14x14x1 input - matches Catapult notebook)...")
    minimal_model = create_minimal_cnn()
    minimal_model.summary()
    
    # Test forward pass
    dummy_input = np.random.randn(1, 14, 14, 1).astype(np.float32)
    output = minimal_model(dummy_input)
    print(f"\nTest: input {dummy_input.shape} -> output {output.shape}")
    
    # Save model
    minimal_model.save('minimal_cnn_keras.h5')
    print("\nSaved: minimal_cnn_keras.h5")
    
    with open('minimal_cnn_keras.json', 'w') as f:
        f.write(minimal_model.to_json())
    minimal_model.save_weights('minimal_cnn_keras_weights.h5')
    print("Saved: minimal_cnn_keras.json + minimal_cnn_keras_weights.h5")
    
    print()
    print("="*60)
    print("Next: Run catapult_direct_keras.py to convert to HLS")
    print("="*60)


if __name__ == "__main__":
    main()
