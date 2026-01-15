# Configuration MCP Server Implementation

## Overview

This document describes the implementation of the Configuration MCP Server, which provides tools for configuring neural network model parameters, optimizers, loss functions, and schedulers through the Model Context Protocol (MCP).

## Implementation Pattern

Following the MCP Server README guidelines, the config CLI commands have been converted to MCP tools with the following changes:

### 1. Function Signature Changes

**Before (CLI):**
```python
@app.command("show")
def show_config():
    arch_manager = arch_manager.current_architecture
    # ... console.print() outputs
```

**After (MCP):**
```python
@arch_mcp.tool()
async def show_config(ctx: Context) -> Dict[str, Any]:
    arch_manager = ctx.get_state("arch_manager")
    # ... return structured dictionary
```

### 2. Output Format Conversion

**Before (CLI):**
```python
console.print(f"[green]✅ Model parameters updated![/green]")
console.print(Panel(...))
```

**After (MCP):**
```python
return {
    "status": "success",
    "message": "Model parameters updated successfully",
    "details": {"updates": updates}
}
```

### 3. Error Handling

**Before (CLI):**
```python
if optimizer_type not in arch_manager.optimizers:
    console.print(f"[red]❌ Optimizer '{optimizer_type}' not available[/red]")
    return
```

**After (MCP):**
```python
if optimizer_type not in arch_manager.optimizers:
    return {
        "status": "error",
        "message": f"Optimizer '{optimizer_type}' not available",
        "details": {
            "optimizer_type": optimizer_type,
            "available_optimizers": list(arch_manager.optimizers.keys())[:10]
        }
    }
```

## Available Tools

### Configuration Viewing

#### `show_config(ctx: Context)`
Shows complete current configuration including:
- Model parameters (height, width, channels, epochs, batch size, device, dataset)
- Complex parameters (data workers, eval interval, warmup epochs, etc.)
- Optimizer configuration
- Loss function configuration
- Scheduler configuration
- Pretrained model configuration

**Returns:**
```python
{
    "status": "success",
    "message": "Configuration retrieved successfully",
    "details": {
        "model_params": {...},
        "complex_params": {...},
        "optimizer": {...},
        "loss_function": {...},
        "scheduler": {...},
        "pretrained": {...}
    }
}
```

### Model Parameters

#### `set_model_params(ctx, height, width, channels, epochs, batch_size, device, dataset, dataset_path)`
Set basic model parameters.

**Example:**
```python
result = await set_model_params(
    ctx=ctx,
    height=224,
    width=224,
    channels=3,
    epochs=10,
    batch_size=32,
    device="cuda",
    dataset="CIFAR10",
    dataset_path="/data/cifar10"
)
```

**Returns:**
```python
{
    "status": "success",
    "message": "Model parameters updated successfully",
    "details": {
        "updates": {
            "height": 224,
            "width": 224,
            "channels": 3,
            "epochs": 10,
            "batch_size": 32,
            "device": "cuda",
            "dataset": "CIFAR10",
            "dataset_path": "/data/cifar10"
        }
    }
}
```

#### `set_complex_params(ctx, data_workers, eval_interval, warmup_epochs, scheduler, num_classes, pretrained_weights)`
Set complex model parameters for advanced configurations.

**Example:**
```python
result = await set_complex_params(
    ctx=ctx,
    data_workers=4,
    eval_interval=5,
    warmup_epochs=5,
    num_classes=10
)
```

### Optimizer Configuration

#### `set_optimizer(ctx, optimizer_type, params)`
Configure the optimizer with parameters.

**Example:**
```python
result = await set_optimizer(
    ctx=ctx,
    optimizer_type="Adam",
    params=["lr=0.001", "betas=[0.9,0.999]", "weight_decay=0.0001"]
)
```

**Returns:**
```python
{
    "status": "success",
    "message": "Optimizer configured successfully",
    "details": {
        "optimizer_type": "Adam",
        "parameters": {
            "lr": 0.001,
            "betas": [0.9, 0.999],
            "weight_decay": 0.0001,
            "eps": 1e-08,
            "amsgrad": false
        }
    }
}
```

#### `list_optimizers(ctx, limit)`
List available optimizer types.

**Returns:**
```python
{
    "status": "success",
    "message": "Found 15 optimizer(s)",
    "details": {
        "total_count": 15,
        "returned_count": 15,
        "optimizers": ["Adam", "SGD", "AdamW", "RMSprop", ...],
        "note": "Showing all optimizers"
    }
}
```

### Loss Function Configuration

#### `set_loss_function(ctx, loss_type, params)`
Configure the loss function with parameters.

**Example:**
```python
result = await set_loss_function(
    ctx=ctx,
    loss_type="CrossEntropyLoss",
    params=["ignore_index=-100", "reduction=mean", "label_smoothing=0.1"]
)
```

**Returns:**
```python
{
    "status": "success",
    "message": "Loss function configured successfully",
    "details": {
        "loss_type": "CrossEntropyLoss",
        "parameters": {
            "ignore_index": -100,
            "reduction": "mean",
            "label_smoothing": 0.1
        }
    }
}
```

#### `list_loss_functions(ctx, limit)`
List available loss function types.

### Scheduler Configuration

#### `set_scheduler(ctx, scheduler_type, params)`
Configure the learning rate scheduler.

**Example:**
```python
result = await set_scheduler(
    ctx=ctx,
    scheduler_type="StepLR",
    params=["step_size=30", "gamma=0.1"]
)
```

To disable scheduler:
```python
result = await set_scheduler(ctx=ctx, scheduler_type="None")
```

**Returns:**
```python
{
    "status": "success",
    "message": "Scheduler configured successfully",
    "details": {
        "scheduler_type": "StepLR",
        "parameters": {
            "step_size": 30,
            "gamma": 0.1
        }
    }
}
```

#### `list_schedulers(ctx, limit)`
List available scheduler types.

### Additional Tools

#### `list_complex_models(ctx)`
List all available complex pretrained models.

#### `list_datasets(ctx)`
List available dataset types.

## Parameter Validation

All configuration tools include comprehensive parameter validation:

1. **Type Checking**: Ensures parameters match expected types (int, float, bool, list)
2. **Value Validation**: Checks if parameters are valid for the selected component
3. **Required Parameters**: Validates all required parameters are provided
4. **Default Values**: Automatically fills in default values for optional parameters

### Example Error Response

```python
{
    "status": "error",
    "message": "Parameter 'lr' expects float value, got: abc",
    "details": {
        "parameter": "lr",
        "value": "abc",
        "expected_type": "float"
    }
}
```

## Integration

The config server is automatically imported and registered in `main.py`:

```python
# Import config server tools to register them with the MCP server
from . import config_server
```

All tools are registered with the `arch_mcp` FastMCP application and are available through the MCP protocol.

## Usage Examples

### Complete Workflow Example

```python
# 1. Set model parameters
await set_model_params(
    ctx=ctx,
    height=224,
    width=224,
    channels=3,
    epochs=50,
    batch_size=64,
    device="cuda",
    dataset="ImageNet",
    dataset_path="/data/imagenet"
)

# 2. Configure optimizer
await set_optimizer(
    ctx=ctx,
    optimizer_type="SGD",
    params=["lr=0.1", "momentum=0.9", "weight_decay=0.0001"]
)

# 3. Configure loss function
await set_loss_function(
    ctx=ctx,
    loss_type="CrossEntropyLoss",
    params=["label_smoothing=0.1"]
)

# 4. Configure scheduler
await set_scheduler(
    ctx=ctx,
    scheduler_type="CosineAnnealingLR",
    params=["T_max=50", "eta_min=0"]
)

# 5. View complete configuration
config = await show_config(ctx=ctx)
print(config)
```

## Testing

To test the config server tools:

```bash
# Run the MCP server
python -m src.cli.mcp_server.main

# Use MCP client to call tools
# Example using fastmcp client:
from fastmcp import FastMCP

client = FastMCP(url="http://localhost:8000")
result = await client.call_tool("show_config", {})
print(result)
```

## Benefits of MCP Implementation

1. **Stateful Sessions**: Configuration persists across multiple tool calls
2. **Structured Data**: Easy to parse and process programmatically
3. **Comprehensive Errors**: Detailed error information for debugging
4. **Type Safety**: Strong type hints for better IDE support
5. **AI Integration**: Seamless integration with AI assistants like Claude
6. **Programmatic Access**: Can be used from any MCP-compatible client

## Migration Notes

All CLI config commands have been successfully migrated to MCP tools:
- ✅ `show` → `show_config`
- ✅ `model-params` → `set_model_params`
- ✅ `complex-params` → `set_complex_params`
- ✅ `optimizer` → `set_optimizer`
- ✅ `loss` → `set_loss_function`
- ✅ `scheduler` → `set_scheduler`
- ✅ `list-complex-models` → `list_complex_models`
- ✅ Added: `list_optimizers`
- ✅ Added: `list_loss_functions`
- ✅ Added: `list_schedulers`
- ✅ Added: `list_datasets`

The original CLI commands remain functional and unchanged.
