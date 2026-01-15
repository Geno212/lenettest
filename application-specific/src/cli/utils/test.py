import os
from pathlib import Path

import cv2
import numpy as np
import torch
from rich.console import Console
from YOLOX.yolox.utils import postprocess

from src.utils.DefenseLayers.high_level_guided_denoiser import HGD

console = Console()


def validate_and_parse_class_names(class_names_path: Path) -> list[str]:
    """Validate and parse class names from a file."""
    try:
        with class_names_path.open("r") as f:
            content = f.read()
            # Split by commas and strip surrounding quotes and whitespace
            class_names = [
                name.strip().strip("\"'") for name in content.split(",") if name.strip()
            ]
        if not class_names:
            raise ValueError("Class names file is empty.")
        if any(" " in name for name in class_names):
            raise ValueError("Class names must not contain spaces.")
        return class_names
    except Exception as e:
        raise ValueError(f"Failed to parse class names: {e}")


def preprocess(img, input_size):
    h, w = img.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    nh, nw = int(h * scale), int(w * scale)
    image_resized = cv2.resize(img, (nw, nh))
    image_padded = np.full((input_size[0], input_size[1], 3), 114, dtype=np.uint8)
    image_padded[:nh, :nw] = image_resized
    image_padded = image_padded.astype(np.float32)
    image_padded = image_padded.transpose(2, 0, 1)
    return torch.from_numpy(image_padded).unsqueeze(0), scale


def visualize(img, bboxes, scores, cls_ids, class_names, conf_threshold=0.0001):
    h, w = img.shape[:2]
    for bbox, score, cls_id in zip(bboxes, scores, cls_ids, strict=False):
        if score < conf_threshold:
            continue
        x1, y1, x2, y2 = bbox.astype(int)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)
        color = (0, 255, 0)
        label = f"{class_names[cls_id]}: {score:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            label,
            (x1, y1 - 10 if y1 - 10 > 10 else y1 + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    return img


def detect_and_visualize(
    model,
    image,
    class_names,
    tag,
    num_classes,
    device: torch.device,
    output_dir: Path = Path(),
    show_image: bool = True,
):
    width, height = image.shape[1], image.shape[0]
    img_input, scale = preprocess(image, (width, height))
    img_input = img_input.float().to(device)

    with torch.no_grad():
        outputs = model(img_input)
        outputs = postprocess(
            outputs,
            num_classes=num_classes,
            conf_thre=0.4,
            nms_thre=0.3,
        )
    result_img = image.copy()
    if outputs[0] is not None:
        output = outputs[0].cpu().numpy()
        bboxes = output[:, 0:4] / scale
        scores = output[:, 4] * output[:, 5]
        cls_ids = output[:, 6].astype(np.int32)

        valid_idx = np.all(np.isfinite(bboxes), axis=1) & np.all(bboxes >= 0, axis=1)
        bboxes = bboxes[valid_idx]
        scores = scores[valid_idx]
        cls_ids = cls_ids[valid_idx]

        bboxes = bboxes.astype(np.int32)
        console.print(
            f"[bold green]{tag} - Valid detections: {len(bboxes)}[/bold green]",
        )

        result_img = visualize(
            image.copy(),
            bboxes,
            scores,
            cls_ids,
            class_names,
            conf_threshold=0.00001,
        )
    else:
        console.print(f"[bold yellow]{tag} - No detections found.[/bold yellow]")

    # Save the output image in the same directory as the Python file
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{tag.lower()}_output.jpg"
    cv2.imwrite(str(output_file), result_img)
    if show_image:
        cv2.imshow(f"{tag} Detection", result_img)
    return result_img


def get_HGD_model(
    device,
    checkpoint_name=Path("best_ckpt.pt"),
    width=1.0,
    growth_rate=32,
    bn_size=4,
) -> HGD:
    dir_relative_path = os.path.relpath(os.path.dirname(__file__), os.getcwd())
    # Get the path of the model and the experiment script
    model_path = os.path.join(dir_relative_path, "..", checkpoint_name)
    model = HGD(width=width, growth_rate=growth_rate, bn_size=bn_size)
    model.load_state_dict(torch.load(model_path, device)["model_dict"])
    return model.to(device)
