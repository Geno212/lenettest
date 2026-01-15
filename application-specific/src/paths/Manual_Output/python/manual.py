
# import the necessary packages
import torch
from torch import nn, optim
from torch.utils.data import DataLoader,Dataset
from torchvision.datasets import mnist
from torchvision import models, transforms
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
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

    writer = SummaryWriter(log_dir=log_dir)
    HEIGHT = 1
    WIDTH = 1
    CHANNELS = 3
    BATCH_SIZE = 2
    EPOCHS = 3
    TRAIN_SPLIT = 0.75
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.1
    device = torch.device("cuda:0")

    model = CNN()
    model = model.to(device)

    transform = v2.Compose(
        [v2.Resize((HEIGHT, WIDTH)), v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    
    # train_dataset = datasets.MNIST(root=r"C:/Users/acer/Downloads/MNISTds",
    #                                                   train=True, download=True, transform=transform)
    train_dataset = DatasetFactory(
        name="MNIST",
        path=r"C:/Users/acer/Downloads/MNISTds",
        transform=transform,
        is_train=True
    )
    
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    loss_fn = nn.AdaptiveLogSoftmaxWithLoss(cutoffs=2,device=device,div_value=4.0,dtype='torch.float32',head_bias=False,in_features=1,n_classes=1
    )

    optimizer = optim.ASGD(
        model.parameters(), alpha=0.75,capturable=False,differentiable=False,foreach=False,lambd=0.0001,lr=0.01,maximize=False,t0=100000.0,weight_decay=1
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

            totalTrainLoss += loss
            trainCorrect += (pred.argmax(1) == y).type(
                torch.float).sum().item()

            progress = ((e*train_size + i) / (EPOCHS*train_size)) * 100
            callback(progress)
        writer.add_scalar("Train/Accuracy", trainCorrect, e)
        writer.add_scalar("Train/Loss", totalTrainLoss, e)
        model.eval()
        print(
            f"Epoch {e+1}, Train Accuracy: {trainCorrect / len(train_dataloader.dataset)}")
    dummy_input = torch.randn(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH).to(
        device)  # Example input tensor
    writer.add_graph(model, dummy_input)
    scripted = torch.jit.script(model)
    try:
        torch.jit.save(scripted, model_output)
    except e:
        print(e)
    writer.close()


if __name__ == '__main__':
    train()
