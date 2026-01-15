"""Full Model Test Script.

Running inference using specified image, weights, and checkpoint paths
"""

from pathlib import Path

# Make repository root (the folder containing `src/`) discoverable when
# running this script directly so `from src...` works without an
# editable install or PYTHONPATH manipulation.
import sys

_me = Path(__file__).resolve()
for _p in (_me, *_me.parents):
    # if repository root contains `src/` add it so `from src...` works
    if (_p / "src").exists():
        sys.path.insert(0, str(_p))
        break

# Add YOLOX package path if found (folder containing `yolox/` lives in `YOLOX/`)
for _p in (_me, *_me.parents):
    candidate = _p / "YOLOX"
    if (candidate / "yolox").exists():
        sys.path.insert(0, str(candidate))
        break

from src.cli.commands.test.test_image import test_image


def main() -> None:
    test_image(
        Path(
            r"C:/Users/Ahmed hosam/Desktop/grad/Application-Specific-Deep-Learning-Accelerator-Designer/src/Test_Images/7144_4.jpg",
        ),
        Path(r"C:/Users/Ahmed hosam/Desktop/grad/Application-Specific-Deep-Learning-Accelerator-Designer/src/Test_Images/best_ckpt.pth"),
        Path(r"C:/Users/Ahmed hosam/Desktop/grad/Application-Specific-Deep-Learning-Accelerator-Designer/weights/Defense.pt"),
        Path(r"C:/Users/Ahmed hosam/Desktop/grad/Application-Specific-Deep-Learning-Accelerator-Designer/src/Test_Images/config.txt"),
    )


if __name__ == "__main__":
    main()
