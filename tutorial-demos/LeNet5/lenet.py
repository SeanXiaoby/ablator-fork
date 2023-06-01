from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import torch
import torchvision
import torchvision.models as models
from PIL import Image
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from ablator import ModelConfig, ModelWrapper, RunConfig, configclass, Literal


# Dataset wrapper for retrieving MNIST dataset and preprocessing data
class MnistWrapper(torchvision.datasets.MNIST):
    def __getitem__(self, index: int) -> Dict[Any, Any]:
        x, y = super().__getitem__(index)
        return {
            "x": x,
            "labels": y,
            # "custom_input": transforms.ToTensor()(Image.fromarray(self.data[index])),
        }


@configclass
class LeConfig(ModelConfig):
    # Configurable attributes
    name: Literal[
        "lenet5",
    ]
    weights: str = "IMAGENET1K_V1"
    progress: bool = False


# Model configuration class, defining hyperparameters of the model
@configclass
class LeRunConfig(RunConfig):
    model_config: LeConfig


# Function to load CIFAR-10 dataset, returns a DataLoader instance
def load_mnist(config: LeRunConfig, flag: str = "train") -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize((227, 227)),  # Resize images to fit the input size of AlexNet
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))]  # Normalize the image data
    )
    root = "/tmp/mnist"
    dataset = MnistWrapper(
        root=root,
        train=flag == "train",
        transform=transform,
        target_transform=None,
        download=not Path(root).exists(),
    )
    dataloader: DataLoader = DataLoader(
        dataset,
        batch_size=config.train_config.batch_size,
        shuffle=True,
    )
    return dataloader


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
# Custom Model class implementing the model architecture.
class MyModel(nn.Module):
    def __init__(self, config: LeConfig) -> None:
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

        model_dict = {
            "lenet5": AlexNet(),
        }

        self.model = model_dict[config.name]

    def forward(self, x, labels, custom_input=None):
        # custom_input is for demo purposes only, defined in the dataset wrapper
        out: torch.Tensor = self.model(x)
        loss = None
        if labels is not None:
            loss = self.loss(out, labels)

        out = out.argmax(dim=-1)
        return {"y_pred": out, "y_true": labels}, loss

# Custom accuracy function
def my_accuracy(y_true, y_pred):
    return accuracy_score(y_true.flatten(), y_pred.flatten())


# Custom Model Wrapper, extending ModelWrapper class from Ablator
class MyModelWrapper(ModelWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def make_dataloader_train(self, run_config: LeRunConfig):  # type: ignore
        # return load_cifar10(run_config, flag="val")
        return load_mnist(run_config, flag="train")

    def make_dataloader_val(self, run_config: LeRunConfig):  # type: ignore
        return load_mnist(run_config, flag="val")

    def evaluation_functions(self) -> Dict[str, Callable]:
        return {"accuracy_score": my_accuracy}

    def custom_evaluation(
        self, model: nn.Module, dataloader: Iterable
    ) -> Optional[Dict[str, Any]]:
        b = next(iter(dataloader))
        img = torchvision.utils.make_grid(b["custom_input"])
        self.logger.update({"train_image": img})
        return super().custom_evaluation(model, dataloader)
