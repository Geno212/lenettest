# Catapult HLS PyTorch-to-RTL Flow - Summary

## Date: December 5, 2025

## What We Accomplished

### 1. Complete PyTorch to RTL Conversion Flow
Successfully created `catapult_keras_flow.py` that:
- Converts PyTorch models → ONNX → Keras → HLS C++ → RTL (Verilog/VHDL)
- Handles NCHW (PyTorch) to NHWC (Keras) format conversion for Catapult compatibility
- Uses Catapult's bundled hls4ml at `$MGC_HOME/shared/pkgs/ccs_hls4ml`

### 2. Key Issues Solved

#### Memory Issue (OOM at 15GB+)
- **Problem**: Catapult synthesis crashed with out-of-memory on 16GB instance
- **Solution**: Changed `reuse_factor` from 1 to 64
- **Result**: Memory reduced from 15GB+ to ~2.4GB peak

#### Timing/Scheduling Issue
- **Problem**: "Feedback path too long" error at 5ns clock period
- **Solution**: 
  - Increased clock period from 5ns to 10ns (100MHz)
  - Added `CLOCK_OVERHEAD 0` directive
- **Result**: Synthesis completed successfully

#### TCL Patching Bug
- **Problem**: Regex incorrectly inserted directive inside CLOCKS brackets
- **Solution**: Fixed regex pattern to insert after CLOCKS line

### 3. Successful Synthesis
- **Model**: SimpleCNN (~52K parameters)
- **Synthesis Time**: 3 hours 9 minutes
- **Peak Memory**: ~3.7GB
- **Output**: Verilog and VHDL RTL files generated

### 4. QuestaSim RTL Verification - PASSED ✅
- Ran RTL vs C++ co-simulation
- **Result**: Simulation PASSED @ 6234806 ns
- **Comparison count**: 10
- **Error count**: 0

### 5. QuestaSim Integration Added
New command-line options in `catapult_keras_flow.py`:
- `--run-questasim`: Run RTL verification after synthesis
- `--questa-timeout`: Set timeout (default 1 hour)
- `--questasim-only`: Only run verification on existing project

## Scripts Created

### catapult_keras_flow.py
Main conversion script with full flow:
```bash
$MGC_HOME/bin/python3 catapult_keras_flow.py model.pt \
    --input-shape 1 1 28 28 \
    -n SimpleCNN \
    --reuse-factor 64 \
    --clock-period 10 \
    --run-questasim
```

### create_simple_model.py
Creates test SimpleCNN model:
```bash
python create_simple_model.py
# Creates simple_cnn.pt
```

## Environment Setup (EC2)

```bash
export MGC_HOME=/data/tools/catapult/Mgc_home
export MTI_HOME=/data/tools/questa/questasim
export MODEL_TECH=$MTI_HOME/linux_x86_64
export LM_LICENSE_FILE=29000@10.9.8.8
export SALT_LICENSE_SERVER=29000@10.9.8.8
export PATH=$MTI_HOME/linux_x86_64:$MGC_HOME/bin:$PATH
```

## Remote Execution (Pending)

### SSM Setup Required
To trigger Catapult jobs remotely via AWS SSM, need IAM permissions added to `vsiDeployment` user:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "ssm:SendCommand",
                "ssm:GetCommandInvocation",
                "ssm:DescribeInstanceInformation",
                "ssm:ListCommands",
                "ssm:ListCommandInvocations"
            ],
            "Resource": "*"
        }
    ]
}
```

Once added, can trigger remotely:
```powershell
aws ssm send-command `
    --instance-ids "i-0962af7f77f3cb484" `
    --document-name "AWS-RunShellScript" `
    --parameters 'commands=["cd /home/ubuntu && $MGC_HOME/bin/python3 catapult_keras_flow.py model.pt --input-shape 1 1 28 28"]'
```

## Known Issues & Solutions

### AlexNet / Large Models - RAM Exceeded
**Problem**: `go analyze` crashes with large models like AlexNet due to RAM limits

**Solutions** (see next section):
1. Increase `reuse_factor` (e.g., 128, 256, or higher)
2. Use `Resource` strategy instead of `Latency`
3. Split model into smaller sub-networks
4. Use larger EC2 instance (r5.xlarge = 32GB, r5.2xlarge = 64GB)
5. Layer-by-layer synthesis

## Files Location

- **Scripts**: `catapult_conversion/`
- **S3 Bucket**: `s3://catapult-nn-models-transfer/scripts/`
- **EC2 Instance**: `vsi-training-3` (i-0962af7f77f3cb484)

## Commit
```
f4b93ba - Add Catapult HLS PyTorch-to-RTL conversion with QuestaSim verification
```
