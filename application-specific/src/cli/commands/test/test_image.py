from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
import torch
import typer
from YOLOX.yolox.exp import get_exp

from src.cli.utils.console import console, error_console
from src.cli.utils.test import (
    detect_and_visualize,
    get_HGD_model,
    validate_and_parse_class_names,
)

app = typer.Typer(help="Test a model with a single image.")


@app.command("image")
def test_image(
    image_path: Annotated[
        Path,
        typer.Argument(
            help="Path to image file to test",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    model_path: Annotated[
        Path,
        typer.Argument(
            help="Path to model weights .pth file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    hgd_ckpt_path: Annotated[
        Path,
        typer.Argument(
            help="Path to HGD checkpoint .pt file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    class_names_path: Annotated[
        Path,
        typer.Argument(
            help="Path to file containing class names",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output directory",
            file_okay=False,
            dir_okay=True,
            writable=True,
        ),
    ] = Path("./test_results"),
) -> None:
    """Test a model with a single image."""
    try:
        class_names_list = validate_and_parse_class_names(class_names_path)
    except ValueError as e:
        error_console.print(f"[red]Error: {e}[/red]")
        return
    print(class_names_list)

    num_classes = len(class_names_list)

    # Load YOLOX model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp = get_exp(None, "yolox-s")
    exp.num_classes = num_classes
    model = exp.get_model()
    model.eval()

    console.print("[bold blue]Loading YOLOX model weights...[/bold blue]")
    ckpt = torch.load(model_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    console.print("[bold green]YOLOX model loaded successfully![/bold green]")

    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        error_console.print(f"[red]❌ Image {image_path} not found![/red]")
        return

    detect_and_visualize(
        model,
        img,
        class_names_list,
        "Perturbed Image",
        num_classes,
        device,
        output_dir,
    )

    # Load HGD model
    console.print("[bold blue]Loading HGD model...[/bold blue]")
    hgd_model = get_HGD_model(device, hgd_ckpt_path)
    hgd_model.eval()

    # Preprocess for HGD
    img_tensor = (
        torch.from_numpy(img.astype(np.float32))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
        .float()
    )

    with torch.no_grad():
        noise = hgd_model(img_tensor)
        denoised_tensor = img_tensor - noise
        denoised_tensor = torch.clamp(denoised_tensor, 0.0, 255.0)

    denoised_tensor = denoised_tensor.squeeze(0).cpu().byte().permute(1, 2, 0).numpy()

    # Detect on denoised image
    detect_and_visualize(
        model,
        denoised_tensor,
        class_names_list,
        "Denoised Image",
        num_classes,
        device,
        output_dir,
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()
