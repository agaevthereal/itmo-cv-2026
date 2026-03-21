import torch
import torch.nn as nn

class RegularizedNet(nn.Module):
    def __init__(self, activation_type='relu', use_dropout=True):
        super(RegularizedNet, self).__init__()
        self.flatten = nn.Flatten()
        self.use_dropout = use_dropout
        
        # Выбор функции активации
        self.activation = nn.ReLU() if activation_type == 'relu' else nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)
        
        # Слои
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.activation(self.fc1(x))
        if self.use_dropout: x = self.dropout(x)
        
        x = self.activation(self.fc2(x))
        if self.use_dropout: x = self.dropout(x)
        
        return self.fc3(x)

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        # Первый сверточный блок
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Второй сверточный блок
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = nn.ReLU()(self.fc1(x))
        return self.fc2(x)
        
    def get_activations(self, x):
        """Метод для 4-го пункта: возвращает активации после первого слоя"""
        with torch.no_grad():
            activations = self.relu1(self.conv1(x))
        return activations
