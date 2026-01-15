# Catapult AI NN Conversion Tools

This directory contains tools to convert trained PyTorch models to RTL using Siemens Catapult AI NN.

## ⚡ One-Command Flow (Recommended)

The simplest way - runs entire flow from Windows, automatically calls EC2:

```powershell
# Convert PyTorch model to RTL with a single command
python catapult_nn_flow.py ..\weights\Defense.pt --input-shape 1 3 224 224

# Or with a simple model
python catapult_nn_flow.py ..\outputs\pretrained\model.pt
```

This automatically:
1. ✅ Uploads your `.pt` model to EC2
2. ✅ Runs Catapult AI NN synthesis (uses integrated hls4ml)
3. ✅ Downloads generated RTL back to Windows

## Overview

The workflow is:
1. **Local**: Prepare PyTorch model and configuration files
2. **Transfer**: Copy files to EC2 where Catapult is installed
3. **EC2**: Run Catapult AI NN synthesis to generate RTL

## Quick Start

### Option 1: Automated Remote Flow (NEW)

```powershell
# Run everything with one command - automatically calls EC2
python catapult_nn_flow.py model.pt --input-shape 1 1 28 28 -o ./output
```

### Option 2: Simple Project Creation

```powershell
# Create a Catapult project from your trained model
python create_catapult_project.py weights/Defense.pt -o ./my_project -s 1,3,224,224

# Transfer to EC2
scp -r ./my_project ubuntu@10.10.8.216:~/catapult_projects/

# SSH to EC2 and run synthesis
ssh ubuntu@10.10.8.216
cd ~/catapult_projects/my_project
./setup.sh
python synthesize.py
```

### Option 3: Full Conversion Pipeline

```powershell
# Full conversion with all options
python convert_pytorch_to_hls.py `
    --model weights/Defense.pt `
    --output ./catapult_output `
    --input-shape 1,3,224,224 `
    --precision "ac_fixed<16,6>" `
    --reuse-factor 4 `
    --io-type io_stream `
    --strategy Latency

# Follow the generated instructions to transfer and run on EC2
```

## Files

| File | Description |
|------|-------------|
| `catapult_nn_flow.py` | **One-command flow** - runs everything on EC2 automatically |
| `create_catapult_project.py` | Simple project creator |
| `convert_pytorch_to_hls.py` | Full conversion with more options |
| `run_catapult_synthesis.sh` | EC2 helper script |
| `example_usage.py` | Examples for your NN Generator models |

| File | Description |
|------|-------------|
| `create_catapult_project.py` | Simple project creator (recommended) |
| `convert_pytorch_to_hls.py` | Full conversion pipeline with ONNX/Keras |
| `example_usage.py` | Example usage with your project's models |

## Configuration Options

### Precision
Fixed-point representation: `ac_fixed<TOTAL_BITS, INTEGER_BITS>`

| Precision | Description | Accuracy | Area |
|-----------|-------------|----------|------|
| `ac_fixed<16,6>` | 16-bit, 6 integer | High | Medium |
| `ac_fixed<8,3>` | 8-bit, 3 integer | Medium | Low |
| `ac_fixed<32,16>` | 32-bit, 16 integer | Very High | High |

### Reuse Factor
Controls parallelism vs resource trade-off:

| Value | Description |
|-------|-------------|
| 1 | Fully parallel (fastest, most resources) |
| 2-4 | Balanced |
| 8+ | Resource-efficient (slower) |

### IO Type
| Type | Description |
|------|-------------|
| `io_stream` | Pipelined streaming (recommended for CNNs) |
| `io_parallel` | All inputs/outputs in parallel |

### Strategy
| Strategy | Description |
|----------|-------------|
| `Latency` | Optimize for speed |
| `Resource` | Optimize for area |

## EC2 Setup

The EC2 instance (10.10.8.216) has Catapult installed at:
- **Catapult**: `/data/tools/catapult/Mgc_home/bin/catapult`
- **hls4ml**: `/data/tools/catapult/Mgc_home/shared/pkgs/ccs_hls4ml/`
- **License**: `MGLS_LICENSE_FILE=29000@10.9.8.8`

### First-time Setup on EC2

```bash
# Create Python virtual environment (one-time)
sh /data/tools/catapult/Mgc_home/shared/pkgs/ccs_hls4ml/create_env.sh
/data/tools/catapult/Mgc_home/bin/python3 $HOME/ccs_venv

# Add to ~/.bashrc
echo 'export MGLS_LICENSE_FILE=29000@10.9.8.8' >> ~/.bashrc
echo 'export MGC_HOME=/data/tools/catapult/Mgc_home' >> ~/.bashrc
```

## Example Models

The project has these trained models that can be converted:

| Model | Path | Input Shape |
|-------|------|-------------|
| Defense | `weights/Defense.pt` | Check model |
| YOLOX | `weights/yolox-small.pth` | (1, 3, 640, 640) |

## Output

After synthesis, you'll get:
- **Verilog/VHDL RTL**: `project/Catapult/project.v1/concat_rtl.v`
- **Synthesis report**: `project/Catapult/project.v1/nnet_layer_results.txt`
- **Area/timing estimates**: Per-layer breakdown

## Troubleshooting

### License Error
```bash
export MGLS_LICENSE_FILE=29000@10.9.8.8
```

### Import Error
```bash
source $HOME/ccs_venv/bin/activate
```

### Model Loading Error
Ensure the .pt file contains a complete model (not just state_dict).
If using state_dict, you need to provide the model class.

### Unsupported Layer
Not all PyTorch layers are supported. Check the hls4ml documentation for supported operations.
