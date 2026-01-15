# Catapult Conversion Module
# Tools for converting PyTorch models to Catapult AI NN format

from .create_catapult_project import (
    create_pytorch_catapult_project,
    extract_model_architecture,
    convert_layer_to_hls4ml
)

from .convert_pytorch_to_hls import (
    load_pytorch_model,
    convert_to_onnx,
    configure_hls4ml,
    generate_hls_project
)

__all__ = [
    'create_pytorch_catapult_project',
    'extract_model_architecture',
    'convert_layer_to_hls4ml',
    'load_pytorch_model',
    'convert_to_onnx',
    'configure_hls4ml',
    'generate_hls_project'
]
