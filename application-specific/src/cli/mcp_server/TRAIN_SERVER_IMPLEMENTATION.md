# Train MCP Server Implementation

## Overview
The train MCP server provides tools for training neural network models through the MCP (Model Context Protocol) interface. It supports three types of training:
1. **Manual Training** - Custom architecture models
2. **Pretrained Training** - Transfer learning with pretrained models
3. **YOLOX Training** - Object detection models

## Implementation Pattern
The train_server.py follows the same standardized pattern as other MCP servers (config_server, generate_server, etc.):
- Uses FastMCP decorators (@arch_mcp.tool())
- Returns structured dictionaries with status, message, and details
- Integrates with the session-based architecture manager
- Provides comprehensive error handling and validation

## Tools Implemented

### 1. `train_manual_model`
Trains a custom architecture neural network model.

**Parameters:**
- `config_file` (Optional[Path]): Path to manual training JSON config. If None, uses current architecture
- `output_dir` (Path): Output directory (default: current_dir/outputs/manual)
- `verbose` (bool): Enable verbose output

**Returns:**
```json
{
  "status": "success|error|warning",
  "message": "Human-readable message",
  "details": {
    "output_dir": "path/to/output",
    "log_dir": "path/to/tensorboard/logs",
    "training_script": "path/to/generated/script",
    "config_file": "config_file_path_or_current_architecture",
    "progress": 100
  }
}
```

**Features:**
- Generates training code using cookiecutter templates
- Sets up training environment identical to GUI workflow
- Dynamically imports and executes generated training script
- Provides progress tracking via callback
- Creates TensorBoard logs for visualization

### 2. `train_pretrained_model`
Trains a model using transfer learning with a pretrained base model.

**Parameters:**
- `config_file` (Optional[Path]): Path to pretrained training JSON config. If None, uses current architecture
- `output_dir` (Path): Output directory (default: current_dir/outputs/pretrained)
- `verbose` (bool): Enable verbose output

**Returns:**
```json
{
  "status": "success|error|warning",
  "message": "Human-readable message",
  "details": {
    "output_dir": "path/to/output",
    "log_dir": "path/to/tensorboard/logs",
    "training_script": "path/to/generated/script",
    "pretrained_model": "model_name",
    "config_file": "config_file_path_or_current_architecture",
    "progress": 100
  }
}
```

**Features:**
- Uses transfer learning templates
- Fine-tunes pretrained models on custom datasets
- Automatically updates paths based on selected model
- Supports all available pretrained models (ResNet, VGG, EfficientNet, etc.)

### 3. `train_yolox_model`
Trains a YOLOX object detection model.

**Parameters:**
- `config_file` (Optional[Path]): Path to YOLOX training JSON config. If None, uses current architecture
- `output_dir` (Path): Output directory (default: current_dir/outputs/yolox)
- `verbose` (bool): Enable verbose output

**Returns:**
```json
{
  "status": "success|error|warning",
  "message": "Human-readable message",
  "details": {
    "output_dir": "path/to/output",
    "log_dir": "path/to/tensorboard/logs",
    "training_script": "path/to/generated/script",
    "yolox_model": "yolox_variant_name",
    "config_file": "config_file_path_or_current_architecture",
    "progress": 100
  }
}
```

**Features:**
- Specialized for YOLOX object detection
- Handles YOLOX-specific configuration
- Uses pretrained YOLOX variants (Nano, Tiny, S, M, L, X)

### 4. `get_training_status`
Retrieves the status of the most recent training run.

**Parameters:**
- `log_dir` (Optional[Path]): Path to TensorBoard logs. If None, uses default location

**Returns:**
```json
{
  "status": "success|info",
  "message": "Human-readable message",
  "details": {
    "log_dir": "path/to/logs",
    "has_logs": true,
    "tensorboard_command": "tensorboard --logdir=path/to/logs"
  }
}
```

**Features:**
- Checks for existing training logs
- Provides TensorBoard command for visualization
- Validates log directory existence

## Helper Functions

### `load_and_validate_config(config_file: Path) -> Dict[str, Any]`
Loads and validates JSON configuration files.

**Raises:**
- `FileNotFoundError`: If config file doesn't exist
- `json.JSONDecodeError`: If config has invalid JSON

### `setup_training_environment(config: dict, output_dir: Path, training_type: str) -> tuple`
Sets up the training environment exactly like the GUI.

**Returns:**
- `(cookiecutter, template_dir, jinja_json_path, cookie_json_path)`

**Features:**
- Initializes SystemPaths and Cookiecutter
- Creates output directories
- Sets log directory in config
- Updates paths based on training type

## Integration

The train_server is registered in `main.py`:

```python
# Import server tools to register them with the MCP server
from . import config_server
from . import generate_server
from . import project_server
from . import arch_server
from . import train_server
```

## Usage Example

### Via MCP Client:

```python
# Train a manual model
result = await client.call_tool("train_manual_model", {
    "config_file": "path/to/config.json",
    "output_dir": "outputs/my_training",
    "verbose": True
})

# Train with pretrained model
result = await client.call_tool("train_pretrained_model", {
    "config_file": "path/to/pretrained_config.json",
    "output_dir": "outputs/transfer_learning"
})

# Train YOLOX model
result = await client.call_tool("train_yolox_model", {
    "config_file": "path/to/yolox_config.json",
    "output_dir": "outputs/yolox_detection"
})

# Check training status
status = await client.call_tool("get_training_status", {})
```

### Using Current Architecture (No Config File):

```python
# Use current architecture in session
result = await client.call_tool("train_manual_model", {
    "output_dir": "outputs/current_arch"
})
```

## Workflow

1. **Configuration Loading**: Load JSON config or use current architecture
2. **Environment Setup**: Create output dirs, initialize cookiecutter and paths
3. **Code Generation**: Generate training scripts via cookiecutter templates
4. **Training Execution**: Dynamically import and run generated training script
5. **Progress Tracking**: Monitor training via callbacks
6. **Cleanup**: Remove temporary imports from sys.path

## Error Handling

The implementation provides comprehensive error handling:
- File not found errors
- JSON parsing errors
- Import errors
- Training execution errors
- Template generation errors

All errors return structured responses with:
- Clear error messages
- Relevant context in details
- Appropriate status codes

## Logging

Uses Python's logging module:
```python
logger = logging.getLogger(__name__)
logger.info("Training progress: 50%")
```

## Dependencies

- `fastmcp`: MCP framework
- `src.utils.Cookiecutter`: Template rendering
- `src.paths.SystemPaths`: Path management
- `json`: Config file parsing
- `importlib`: Dynamic module loading
- `logging`: Progress and error logging

## Comparison with CLI Implementation

The MCP server implementation maintains feature parity with the CLI commands:

| Feature | CLI | MCP Server |
|---------|-----|------------|
| Manual training | ✅ | ✅ |
| Pretrained training | ✅ | ✅ |
| YOLOX training | ✅ | ✅ |
| Progress tracking | ✅ | ✅ |
| TensorBoard logs | ✅ | ✅ |
| Config validation | ✅ | ✅ |
| Error handling | ✅ | ✅ |
| Verbose output | ✅ | ✅ |

## Future Enhancements

Potential improvements:
- Real-time progress streaming
- Training cancellation support
- Multi-GPU training configuration
- Checkpoint management tools
- Training resume functionality
- Hyperparameter tuning tools
