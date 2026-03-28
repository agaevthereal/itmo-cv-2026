from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader
import os


def get_dataloaders(data_dir, batch_size=32, input_size=224):
    weights = ResNet18_Weights.IMAGENET1K_V1
    preprocess_params = weights.transforms()
    
    mean = preprocess_params.mean
    std = preprocess_params.std

    # Трансформации для тренировки: аугментация для борьбы с переобучением
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size),   # Случайный кроп и ресайз
        transforms.RandomHorizontalFlip(),          # Случайный поворот по горизонтали
        transforms.ToTensor(),
        # Нормализация (стандартная для ImageNet моделей)
        transforms.Normalize(mean=mean, std=std)
    ])

    # Трансформации для валидации: только ресайз и кроп центра
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Загрузка датасетов из папок
    train_set = datasets.ImageFolder(os.path.join(data_dir, 'train'), train_transform)
    val_set = datasets.ImageFolder(os.path.join(data_dir, 'val'), val_transform)

    # Создание Dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
