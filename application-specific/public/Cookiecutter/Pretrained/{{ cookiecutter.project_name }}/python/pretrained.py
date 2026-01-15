# Import the necessary packages
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms, datasets
from torchvision.transforms import v2
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import *
from torch.utils.tensorboard import SummaryWriter
import torchvision
import datetime
import os
import json
import glob
from PIL import Image

# Set model output path
basedir = os.path.dirname(__file__)
model_output = os.path.normpath(os.path.join(basedir, "../SystemC/Pt/model.pt"))

{% if cookiecutter.misc_params.dataset.value == "CustomDataset" %}
class CustomDataset(Dataset):
    def __init__(self, root, transform=None, download=None, train=None):
        self.root = root
        self.transform = transform
        self.data = self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root, self.data[idx]['filename'])
        image = Image.open(img_name).convert('RGB')
        label = int(self.data[idx]['label'])

        if self.transform:
            image = self.transform(image)

        return image, label

    def load_data(self):
        json_file = glob.glob(os.path.join(self.root, "*.json"))[0]
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data
{% endif %}

def train(callback, logdir):
    # Initialization
    unique_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    base_log_dir = logdir
    log_dir = os.path.join(base_log_dir, unique_name)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir, comment="Pretrained_Model")
    # JSON metrics file (updated each epoch and finally)
    metrics_file = os.path.join(log_dir, "training_metrics.json")
    metrics = {"epochs": []}
    MAX_EPOCHS = 15
    TARGET_ACCURACY = {{ cookiecutter.misc_params.target_accuracy if cookiecutter.misc_params.target_accuracy is defined else 'None' }}
    if TARGET_ACCURACY is None:
        target_acc = None
    else:
        target_acc = float(TARGET_ACCURACY)

    HEIGHT = {{cookiecutter.misc_params.height}}
    WIDTH = {{cookiecutter.misc_params.width}}
    BATCH_SIZE = {{cookiecutter.misc_params.batch_size}}
    EPOCHS = {{ cookiecutter.misc_params.num_epochs if cookiecutter.misc_params.num_epochs is defined else 'None' }}
    if EPOCHS is None:
        EPOCHS = MAX_EPOCHS
    CHANNELS = {{cookiecutter.misc_params.channels}}
    device = torch.device("{{cookiecutter.misc_params.device.value}}")
    name2 = "{{cookiecutter.pretrained.value}}".lower()
    
    # Enforce minimum input size for pretrained models
    MIN_SIZE = 224
    if HEIGHT < MIN_SIZE or WIDTH < MIN_SIZE:
        import warnings
        warnings.warn(
            f"Pretrained models require minimum input size {MIN_SIZE}x{MIN_SIZE}. "
            f"Provided: {HEIGHT}x{WIDTH}. Auto-correcting to {MIN_SIZE}x{MIN_SIZE}.",
            UserWarning
        )
        HEIGHT = max(HEIGHT, MIN_SIZE)
        WIDTH = max(WIDTH, MIN_SIZE)

    if name2.lower().startswith("yolo"):
        from YOLOX.yolox.exp import get_exp
        current_file_path = os.path.abspath(__file__)
        weights_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))), 'weights')
        file_name = f'{name2}.pth'
        file_path = os.path.join(weights_dir, file_name)
        print(file_path)

        # Load YOLOX experiment
        exp = get_exp(None, 'yolox_s')
        model = exp.get_model()
        model.eval()

        # Load weights
        checkpoint = torch.load(file_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])

    else:
        model = models.{{cookiecutter.pretrained.value}}(weights='DEFAULT')

    for param in model.parameters():
        param.requires_grad = False

    transform = transforms.Compose(
        [
            v2.Resize((HEIGHT, WIDTH)),
            v2.Grayscale(num_output_channels=3),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ]
    )

    {% if cookiecutter.misc_params.dataset.value == "CustomDataset" %}
    train_dataset = CustomDataset(root=r"{{cookiecutter.misc_params.dataset_path}}", train=True, download=True, transform=transform)
    {% else %}
    train_dataset = datasets.{{cookiecutter.misc_params.dataset.value}}(root=r"{{cookiecutter.misc_params.dataset_path}}", train=True, download=True, transform=transform)
    {% endif %}
    test_dataset = datasets.{{cookiecutter.misc_params.dataset.value}}(root=r"{{cookiecutter.misc_params.dataset_path}}", train=False, download=True, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    loss_fn = nn.{{cookiecutter.loss_func.type}}(
        {% for key, value in cookiecutter.loss_func.params|dictsort %}
        {% if value is string and not (value.startswith("'") or value.startswith('"') or value[0] in '-0123456789') -%}{{key}}='{{value}}',
        {% else -%}{{key}}={{value}},
        {% endif %}
        {% endfor %}
    )

    class_names = train_dataset.classes

    try:
        model.aux.logits = False
    except:
        pass

    if name2.startswith("yolo"):
        (name, head) = list(model.named_modules())[-6]
        if isinstance(head, nn.Sequential):
            (i, inner_head) = list(head.named_children())[-1]
            head[i] = nn.Conv2d(
                in_channels=inner_head.in_channels,
                #out_channels=len(class_names) + 4 + 1,
                out_channels=15,
                kernel_size=1,
                stride=1
            )
        else:
            model.__dict__['_modules'][name] = nn.Conv2d(
                in_channels=head.in_channels,
                #               out_channels=len(class_names) + 4 + 1,
                out_channels=15,
                kernel_size=1,
                stride=1
            )
    else:
        (name, layer) = list(model.named_children())[-1]
        if type(layer) == type(nn.Sequential()):
            (i, j) = list(layer.named_children())[-1]
            model.__dict__['_modules'][name].__dict__['_modules'][i] = nn.Linear(
                j.in_features, len(class_names), device=device)
        else:
            model.__dict__['_modules'][name] = nn.Linear(
                layer.in_features, len(class_names), device=device
            )

    model = model.to(device)

    optimizer = optim.{{cookiecutter.optimizer.type}}(
        [{"params": model.parameters(), "initial_lr": {{cookiecutter.optimizer.params.lr}}}]
        {% for key, value in cookiecutter.optimizer.params|dictsort %}
        {% if value is sequence and value is not string -%}, {{key}}=({{value | join(', ')}})
        {% elif value is string and not (value.startswith("'") or value.startswith('"') or value[0] in '-0123456789') -%}, {{key}}='{{value}}'
        {% else -%}, {{key}}={{value}}
        {% endif %}
        {% endfor %}
    )

    train_size = len(train_dataset)
    for param in model.parameters():
        param.requires_grad = True

    {% if cookiecutter.scheduler.type != "None" %}
    scheduler = {{cookiecutter.scheduler.type}}(optimizer,
        {% for key, value in cookiecutter.scheduler.params|dictsort %}
        {% if value is sequence and value is not string -%}
        {{key}}=({{value | join(', ')}}),
        {% elif value is string and not (value.startswith("'") or value.startswith('"') or value[0] in '-0123456789') -%}{{key}}='{{value}}',
        {% else -%}{{key}}={{value}},
        {% endif %}
        {% endfor %}
    )
    {% endif %}

    for e in range(0, EPOCHS):
        model.train()
        totalTrainLoss = 0
        trainCorrect = 0
        data_iter = iter(train_dataloader)
        images, labels = next(data_iter)
        grid = torchvision.utils.make_grid(images)
        writer.add_image("Input Images", grid, e)

        for i, (x, y) in enumerate(train_dataloader):
            (x, y) = (x.to(device), y.to(device))
            pred = model(x)
            loss = loss_fn(pred, y)
            writer.add_scalar("Loss/train", loss.item(),
                            e * len(train_dataloader) + i)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            totalTrainLoss += loss.item()
            trainCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()
            progress = ((e * len(train_dataloader) + i) /
                        (EPOCHS * len(train_dataloader))) * 100
            callback(progress)

        writer.add_scalar("Train/Accuracy", trainCorrect / len(train_dataloader.dataset), e)
        writer.add_scalar("Train/Loss", totalTrainLoss, e)

        # Append epoch metrics to JSON and flush to disk (atomic replace)
        try:
            epoch_entry = {
                "epoch": int(e + 1),
                "train_loss": float(totalTrainLoss),
                "train_accuracy": float(trainCorrect / len(train_dataloader.dataset))
            }
            metrics["epochs"].append(epoch_entry)
            tmp_path = metrics_file + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            os.replace(tmp_path, metrics_file)
        except Exception:
            # Best-effort logging; do not fail training on metrics write errors
            pass

        {% if cookiecutter.scheduler.type != "None" %}
        scheduler.step()
        {% endif %}

        model.eval()
        print(f"Epoch {e+1}, Train Accuracy: {trainCorrect / len(train_dataloader.dataset)}")

        # Early stop if target accuracy reached
        if target_acc is not None and (trainCorrect / len(train_dataloader.dataset)) >= target_acc:
            break

    dummy_input = torch.randn(3, 3, HEIGHT, WIDTH).to(device)
    writer.add_graph(model, dummy_input)

    with torch.no_grad():
        model.eval()
        preds = []
        testCorrect = 0

        for x, y in test_dataloader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            preds.extend(pred.argmax(axis=1).cpu().numpy())
            testCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()

    trainAccuracy = trainCorrect / len(train_dataloader.dataset)
    testAccuracy = testCorrect / len(test_dataloader.dataset)

    print("Train Accuracy:", trainAccuracy)
    print("Test Accuracy:", testAccuracy)

    torch.save(model, model_output)
    # Final metrics
    try:
        metrics["final_train_accuracy"] = float(trainAccuracy)
        metrics["final_test_accuracy"] = float(testAccuracy)
        tmp_path = metrics_file + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        os.replace(tmp_path, metrics_file)
    except Exception:
        pass

    writer.close()

if __name__ == "__main__":
    train()
