import torch
import time
import copy
from tqdm import tqdm
from sklearn.metrics import f1_score


class ModelTrainer:
    def __init__(self, model, device, criterion, optimizer, scheduler=None):
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}

    def train(self, train_loader, val_loader, num_epochs=10):
        since = time.time()
        best_f1 = 0.0

        for epoch in range(num_epochs):
            train_loss, train_f1 = self._train_epoch(train_loader)
            val_loss, val_f1 = self._validate_epoch(val_loader)

            if self.scheduler:
                self.scheduler.step()

            self.history['train_loss'].append(train_loss)
            self.history['train_f1'].append(train_f1)
            self.history['val_loss'].append(val_loss)
            self.history['val_f1'].append(val_f1)

            print(f'Epoch {epoch+1}/{num_epochs} | '
                  f'Train Loss: {train_loss:.4f} F1: {train_f1:.4f} | '
                  f'Val Loss: {val_loss:.4f} F1: {val_f1:.4f}')

            # Сохраняем лучшую модель по F1
            if val_f1 > best_f1:
                best_f1 = val_f1
                self.best_model_wts = copy.deepcopy(self.model.state_dict())

        print(f'Training complete in {time.time() - since:.0f}s. Best Val F1: {best_f1:.4f}')
        self.model.load_state_dict(self.best_model_wts)

        return self.history

    def _train_epoch(self, loader):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(enumerate(loader), total=len(loader), desc="    Train", leave=False)
        for _, (inputs, labels) in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        epoch_loss = running_loss / len(loader.dataset)
        epoch_f1 = f1_score(all_labels, all_preds, average='macro')

        return epoch_loss, epoch_f1

    @torch.no_grad()
    def _validate_epoch(self, loader):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(enumerate(loader), total=len(loader), desc="    Val", leave=False)
        for _, (inputs, labels) in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        epoch_loss = running_loss / len(loader.dataset)
        epoch_f1 = f1_score(all_labels, all_preds, average='macro')

        return epoch_loss, epoch_f1
