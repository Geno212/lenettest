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



def train(callback, logdir):
    # Initialization
    unique_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    base_log_dir = logdir
    log_dir = os.path.join(base_log_dir, unique_name)

    writer = SummaryWriter(log_dir=log_dir, comment="Pretrained_Model")
    HEIGHT = 63
    WIDTH = 63
    BATCH_SIZE = 2
    EPOCHS = 3
    CHANNELS = 3
    device = torch.device("cuda:0")
    name2 = "alexnet".lower()

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
        model = models.alexnet(weights='DEFAULT')

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

    
    train_dataset = datasets.MNIST(root=r"C:/Users/acer/Downloads/MNISTds", train=True, download=True, transform=transform)
    
    test_dataset = datasets.MNIST(root=r"C:/Users/acer/Downloads/MNISTds", train=False, download=True, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    loss_fn = nn.AdaptiveLogSoftmaxWithLoss(
        
        cutoffs=2,
        
        
        device='In miscelleneous params',
        
        
        div_value=4.0,
        
        
        dtype='torch.float32',
        
        
        head_bias=False,
        
        
        in_features=1,
        
        
        n_classes=1,
        
        
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

    optimizer = optim.ASGD(
        [{"params": model.parameters(), "initial_lr": 0.01}]
        
        , alpha=0.75
        
        
        , capturable=False
        
        
        , differentiable=False
        
        
        , foreach=False
        
        
        , lambd=0.0001
        
        
        , lr=0.01
        
        
        , maximize=False
        
        
        , t0=100000.0
        
        
        , weight_decay=1
        
        
    )

    train_size = len(train_dataset)
    for param in model.parameters():
        param.requires_grad = True

    
    scheduler = CosineAnnealingLR(optimizer,
        
        eta_min=1,
        
        
        last_epoch=1,
        
        
    )
    

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

        
        scheduler.step()
        

        model.eval()
        print(f"Epoch {e+1}, Train Accuracy: {trainCorrect / len(train_dataloader.dataset)}")

    dummy_input = torch.randn(3, CHANNELS, HEIGHT, WIDTH).to(device)
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

    scripted = torch.jit.script(model)
    try:
        torch.jit.save(scripted, model_output)
    except Exception as e:
        print(e)
    writer.close()

if __name__ == "__main__":
    train()
