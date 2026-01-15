
# import the necessary packages
import torch
from torch import nn, optim
from torch.utils.data import DataLoader,Dataset
from torchvision.datasets import mnist
from torchvision import models, transforms
from torch.optim import lr_scheduler
from torch.utils.tensorboard.writer import SummaryWriter
import torchvision
from torchvision.transforms import v2
from torchvision import datasets, transforms

from python.model import CNN
import datetime
import os
from PIL import Image
import json
import glob

basedir = os.path.dirname(__file__)
model_output = os.path.normpath(
    os.path.join(basedir, '../SystemC/Pt/model.pt'))
test_output = os.path.normpath(os.path.join(basedir, '../test.txt'))

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
        json_file = glob.glob(os.path.join(self.root,"*.json"))[0]
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data
{%- endif %}
def DatasetFactory(name, path, transform, is_train=True):
    if name == "CustomDataset":
        return CustomDataset(root=path, transform=transform)

    elif name == "GTSRB":
        return datasets.GTSRB(root=path, split="train" if is_train else "test", download=True, transform=transform)

    elif name == "MNIST":
        return datasets.MNIST(root=path, train=is_train, download=True, transform=transform)

    elif name == "CIFAR10":
        return datasets.CIFAR10(root=path, train=is_train, download=True, transform=transform)

    # Add more datasets as needed
    else:
        raise ValueError(f"Dataset '{name}' is not supported.")

def train(callback, logdir):
    unique_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    base_log_dir = logdir
    log_dir = os.path.join(base_log_dir, unique_name)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)
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
    CHANNELS = {{cookiecutter.misc_params.channels}}
    BATCH_SIZE = {{cookiecutter.misc_params.batch_size}}
    EPOCHS = {{ cookiecutter.misc_params.num_epochs if cookiecutter.misc_params.num_epochs is defined else 'None' }}
    if EPOCHS is None:
        EPOCHS = MAX_EPOCHS
    TRAIN_SPLIT = 0.75
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.1
    device = torch.device("{{cookiecutter.misc_params.device.value}}")

    model = CNN()
    model = model.to(device)

    transform = v2.Compose(
        [v2.Resize((HEIGHT, WIDTH)), v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    {% if cookiecutter.misc_params.dataset.value == "CustomDataset" %}
    train_dataset = {{cookiecutter.misc_params.dataset.value}}(root=r"{{cookiecutter.misc_params.dataset_path}}",train=True, download=True, transform=transform)
    {% else %}
    # train_dataset = datasets.{{cookiecutter.misc_params.dataset.value}}(root=r"{{cookiecutter.misc_params.dataset_path}}",
    #                                                   train=True, download=True, transform=transform)
    train_dataset = DatasetFactory(
        name="{{cookiecutter.misc_params.dataset.value}}",
        path=r"{{cookiecutter.misc_params.dataset_path}}",
        transform=transform,
        is_train=True
    )
    {% endif %}
    {% if cookiecutter.misc_params.dataset.value == "CustomDataset" %}
    {%- endif %}
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    loss_fn = nn.{{ cookiecutter.loss_func.type }}(
    {%- for key, value in cookiecutter.loss_func.params|dictsort %}
        {%- if key == "device" -%}
            {{ key }}=device{{ "," if not loop.last }}
        {%- else -%}
            {%- if value is sequence and value is not string -%}
                {{ key }}=({{ value | join(', ') }}){{ "," if not loop.last }}
            {%- else -%}
                {%- if value is string -%}
                    {%- if value.startswith("'") or value.startswith('"') or value[0] in '-0123456789' -%}
                        {{ key }}={{ value }}{{ "," if not loop.last }}
                    {%- else -%}
                        {{ key }}='{{ value }}'{{ "," if not loop.last }}
                    {%- endif -%}
                {%- else -%}
                    {{ key }}={{ value }}{{ "," if not loop.last }}
                {%- endif -%}
            {%- endif -%}
        {%- endif -%}
    {%- endfor %}
    )

    optimizer = optim.{{ cookiecutter.optimizer.type }}(
        model.parameters()
        {%- if cookiecutter.optimizer.params -%}, {% endif -%}
    {%- for key, value in cookiecutter.optimizer.params|dictsort %}
        {%- if value is sequence and value is not string -%}
            {{ key }}=({{ value | join(', ') }}){{ "," if not loop.last }}
        {%- else -%}
            {%- if key == "device" -%}
                {{ key }}=device{{ "," if not loop.last }}
            {%- else -%}
                {%- if value is string -%}
                    {%- if value.startswith("'") or value.startswith('"') or value[0] in '-0123456789' -%}
                        {{ key }}={{ value }}{{ "," if not loop.last }}
                    {%- else -%}
                        {{ key }}='{{ value }}'{{ "," if not loop.last }}
                    {%- endif -%}
                {%- else -%}
                    {{ key }}={{ value }}{{ "," if not loop.last }}
                {%- endif -%}
            {%- endif -%}
        {%- endif -%}
    {%- endfor %}
    )

    train_size = len(train_dataset)
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
            writer.add_scalar("Loss/train",loss.item(),
                              e * len(train_dataloader) + i)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            totalTrainLoss += loss.item()
            trainCorrect += (pred.argmax(1) == y).type(
                torch.float).sum().item()

            progress = ((e*train_size + i) / (EPOCHS*train_size)) * 100
            callback(progress)
        # Normalized accuracy and scalar logging
        train_acc = trainCorrect / len(train_dataloader.dataset)
        writer.add_scalar("Train/Accuracy", train_acc, e)
        writer.add_scalar("Train/Loss", float(totalTrainLoss), e)

        # Append epoch metrics to JSON
        try:
            epoch_entry = {
                "epoch": int(e + 1),
                "train_loss": float(totalTrainLoss),
                "train_accuracy": float(train_acc)
            }
            metrics["epochs"].append(epoch_entry)
            tmp_path = metrics_file + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            os.replace(tmp_path, metrics_file)
        except Exception:
            pass
        model.eval()
        print(
            f"Epoch {e+1}, Train Accuracy: {trainCorrect / len(train_dataloader.dataset)}")

        # Early stop if target accuracy reached
        if target_acc is not None and train_acc >= target_acc:
            break
    dummy_input = torch.randn(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH).to(
        device)  # Example input tensor
    writer.add_graph(model, dummy_input)
    torch.save(model, model_output)
    # Final metrics - test split is not computed here; keep final_train_accuracy
    try:
        metrics["final_train_accuracy"] = float(trainCorrect / len(train_dataloader.dataset))
        tmp_path = metrics_file + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        os.replace(tmp_path, metrics_file)
    except Exception:
        pass

    writer.close()


if __name__ == '__main__':
    train()
