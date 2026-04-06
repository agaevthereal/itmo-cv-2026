import torch
import torch.nn as nn
from torchvision import models


class CarColorClassifier(nn.Module):
    def __init__(self, backbone, num_classes=10, freeze_backbone=True):
        super(CarColorClassifier, self).__init__()

        if not backbone:
            raise ValueError("Invalid model_type. pass some model")

        self.backbone = backbone

        # 2. Замораживаем веса "хребта" (backbone), если нужно
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 3. Заменяем последний полносвязный слой (fc) под наше кол-во цветов
        # У ResNet18 in_features = 512
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.backbone(x)


class CarColorClassifierCustom(nn.Module):
    def __init__(self, num_classes=10):
        super(CarColorClassifierCustom, self).__init__()
        
        # Сверточный блок 1: вытаскиваем базовые цвета и границы
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # 224 -> 112
        )
        
        # Сверточный блок 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # 112 -> 56
        )
        
        # Сверточный блок 3
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # 56 -> 28
        )
        
        # Полносвязные слои (Классификатор)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fc(x)

        return x
