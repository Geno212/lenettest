# Running Catapult AI NN on EC2

Since your Windows machine can't SSH directly to the EC2 instance (it's on a private VPC),
you need to run the Catapult flow **from inside the EC2** via your DCV session.

## Step 1: Copy Files to EC2

In your **DCV session** (the remote desktop), open a terminal and run:

```bash
# Create a working directory
mkdir -p ~/catapult_nn
cd ~/catapult_nn

# Option A: If you have the files on a shared drive or can paste
# Just copy catapult_local.py to this directory

# Option B: Download from your git repo
git clone https://github.com/KarimBassel/Application-Specific-Deep-Learning-Accelerator-Designer.git
cd Application-Specific-Deep-Learning-Accelerator-Designer/catapult_conversion
```

## Step 2: Copy Your Model

Copy your trained `.pt` model to the EC2. Options:

1. **Via DCV file transfer** - drag and drop
2. **Via S3** - upload from Windows, download on EC2
3. **Via git** - if models are in the repo

```bash
# If using S3
aws s3 cp s3://your-bucket/model.pt ./

# Or if the model is in the repo
ls ../weights/
```

## Step 3: Run the Conversion

```bash
# Check installation first
python3 catapult_local.py --check

# Convert a model (example with MNIST-like input)
python3 catapult_local.py model.pt --input-shape 1 1 28 28

# Convert with custom settings
python3 catapult_local.py model.pt \
    --input-shape 1 3 224 224 \
    --precision "ac_fixed<16,6>" \
    --reuse-factor 4 \
    --clock 10.0 \
    -o ./my_output

# Just convert without running Catapult (to inspect files first)
python3 catapult_local.py model.pt --convert-only
```

## Step 4: View Results

```bash
# Check output files
ls -la ./catapult_output/

# View generated RTL
cat ./catapult_output/hls_*/Catapult/*.v

# Check synthesis reports
cat ./catapult_output/hls_*/timing_report.txt
cat ./catapult_output/hls_*/area_report.txt
```

## Quick Reference

| Option | Description |
|--------|-------------|
| `--input-shape B C H W` | Input tensor shape (batch, channels, height, width) |
| `--precision` | Fixed-point precision, e.g., `ac_fixed<16,6>` |
| `--reuse-factor` | Resource reuse (1=fastest, higher=smaller) |
| `--clock` | Clock period in ns |
| `--io-type` | `io_stream` or `io_parallel` |
| `--strategy` | `Latency` or `Resource` |
| `--convert-only` | Don't run Catapult synthesis |

## Troubleshooting

### hls4ml not found
```bash
# Install hls4ml
pip3 install hls4ml

# Or use Catapult's bundled version
export PYTHONPATH=/data/tools/catapult/Mgc_home/shared/pkgs/ccs_hls4ml:$PYTHONPATH
```

### License error
```bash
# Set license server
export MGLS_LICENSE_FILE=29000@10.9.8.8
```

### Model conversion fails
- Check if your model uses supported layers
- Try with `--convert-only` to see the generated files
- Check hls4ml documentation for supported operations
