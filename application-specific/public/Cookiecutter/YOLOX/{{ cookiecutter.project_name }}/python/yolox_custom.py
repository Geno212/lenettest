"""Custom YOLOX Experiment Class
This class extends the YOLOX experiment class to customize dataset paths
"""

import sys
import torch
import torch.optim
from torch import nn
from YOLOX.yolox.exp import Exp as MyExp
from YOLOX.yolox.core import Trainer

class EarlyStopTrainer(Trainer):
    """
    Custom Trainer that implements early stopping based on Target Accuracy.
    Overrides evaluate_and_save_model to check results immediately after validation.
    """
    def evaluate_and_save_model(self):
        # 1. Run the standard YOLOX evaluation first.
        # This updates self.best_ap and saves the model checkpoints.
        super().evaluate_and_save_model()

        # 2. Get the target accuracy from the experiment config
        target = getattr(self.exp, "target_accuracy", None)

        # 3. Check if target exists and is valid
        if target is None or str(target) == 'None':
            return

        # 4. Check current Best AP (Average Precision)
        # self.best_ap is a float (0.0 - 1.0)
        # target is a float (0.0 - 1.0)
        current_ap = float(self.best_ap)
        target_val = float(target)

        if current_ap >= target_val:
            print(f"\n[EarlyStop] 🎯 Target Accuracy ({target_val}) reached! Current Best: {current_ap:.4f}")
            print("[EarlyStop] Stopping training process gracefully...")
            
            # 5. Exit the process with Success code (0).
            sys.exit(0)


class Exp(MyExp):
    """Custom Experiment class for YOLOX."""

    def __init__(self):
        print("Initializing custom Experiment...")  # Debugging point
        super(Exp, self).__init__()

        # ---------------- Dataset Settings ---------------- #
        self.data_dir = r"{{cookiecutter.misc_params.dataset_path}}"  # Dataset root
        self.train_ann = "train.json"
        self.val_ann = "val.json"
        self.train_img_dir = "images/train"
        self.val_img_dir = "images/val"

        print(
            f"Dataset settings: data_dir={self.data_dir}, train_ann={self.train_ann}, val_ann={self.val_ann}",
        )  # Debugging point

        # ---------------- Model Settings ---------------- #
        self.num_classes = {{cookiecutter.complex_misc_params.num_classes}}  # Change this to match your dataset classes
        self.depth = {{cookiecutter.pretrained.depth}}  # Model depth (YOLOX-S)
        self.width = {{cookiecutter.pretrained.width}}  # Model width (YOLOX-S)

        print(
            f"Model settings: num_classes={self.num_classes}, depth={self.depth}, width={self.width}",
        )  # Debugging point

        # ---------------- Training Settings ---------------- #
        MAX_EPOCHS = 15
        _epochs = {{ cookiecutter.misc_params.num_epochs if cookiecutter.misc_params.num_epochs is defined else 'None' }}
        self.max_epoch = _epochs if _epochs is not None else MAX_EPOCHS  # Total training epochs
        self.target_accuracy = {{ cookiecutter.misc_params.target_accuracy if cookiecutter.misc_params.target_accuracy is defined else 'None' }}
        self.data_num_workers = {{cookiecutter.complex_misc_params.data_num_workers}}  # Number of CPU workers
        self.eval_interval = {{cookiecutter.complex_misc_params.eval_interval}}  # Run validation every 5 epochs
        self.warmup_epochs = {{cookiecutter.complex_misc_params.warmup_epochs}}
          # Warmup phase

        print(
            f"Training settings: max_epoch={self.max_epoch}, data_num_workers={self.data_num_workers}, "
            f"eval_interval={self.eval_interval}, warmup_epochs={self.warmup_epochs}",
        )  # Debugging point

        # ---------------- Scheduler & Learning Rate ---------------- #
        self.momentum = 0.9  # default momentum
        self.weight_decay = 5e-4  # default weight decay
        self.scheduler = r"{{cookiecutter.complex_misc_params.scheduler}}"
        print(
            f"Optimizer settings: basic_lr_per_img={self.basic_lr_per_img}, momentum={self.momentum}, "
            f"weight_decay={self.weight_decay}",
        )  # Debugging point

        # ---------------- Advanced Settings ---------------- #
        self.ema = True  # Enable EMA (Exponential Moving Average)
        self.input_size = (
            {{cookiecutter.misc_params.width}},
            {{cookiecutter.misc_params.height}},
        )
        self.test_size = (
            {{cookiecutter.misc_params.width}},
            {{cookiecutter.misc_params.height}},
        )

        # self.optimizer = "adams"
        print(
            f"Advanced settings: no_aug_epochs={self.no_aug_epochs}, ema={self.ema}, "
            f"input_size={self.input_size}, test_size={self.test_size}",
        )  # Debugging point

    def get_trainer(self, args):
        """Override to use our Custom EarlyStopTrainer"""
        # This tells YOLOX to use the class defined at the top of this file
        trainer = EarlyStopTrainer(self, args)
        return trainer

    def get_optimizer(self, batch_size):
        """Get the optimizer for training."""
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)  # apply decay

            optimizer_type = "{{cookiecutter.optimizer.type}}".lower()

            if optimizer_type == "sgd":
                optimizer = torch.optim.SGD(
                    pg0,
                    lr=lr,
                    momentum=self.momentum,
                    nesterov=True,
                    weight_decay=self.weight_decay,
                )
            elif optimizer_type == "adam":
                optimizer = torch.optim.Adam(pg0, lr=lr, weight_decay=self.weight_decay)
            elif optimizer_type == "adamw":
                optimizer = torch.optim.AdamW(
                    pg0,
                    lr=lr,
                    weight_decay=self.weight_decay,
                )
            elif optimizer_type == "rmsprop":
                optimizer = torch.optim.RMSprop(
                    pg0,
                    lr=lr,
                    momentum=self.momentum,
                    weight_decay=self.weight_decay,
                )
            elif optimizer_type == "adagrad": # 
                optimizer = torch.optim.Adagrad(
                    pg0, 
                    lr=lr, 
                    weight_decay=self.weight_decay
                )
            else:
                raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay},
            )  # add pg1 with weight_decay
            optimizer.add_param_group({"params": pg2})
            self.optimizer = optimizer

        return self.optimizer