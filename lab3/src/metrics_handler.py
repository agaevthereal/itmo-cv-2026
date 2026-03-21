import matplotlib.pyplot as plt


class MetricsHandler:
    @staticmethod
    def plot_comparison(histories, titles):
        """Рисует графики Loss и Accuracy для списка историй обучения"""
        epochs = range(1, len(histories[0]['train_loss']) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        for hist, title in zip(histories, titles):
            ax1.plot(epochs, hist['val_loss'], label=f'Val Loss: {title}', marker='o')
            ax2.plot(epochs, hist['val_acc'], label=f'Val Acc: {title}', marker='o')
            
        ax1.set_title('Validation Loss Comparison')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.set_title('Validation Accuracy Comparison')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def show_activations(activations, num_cols=4):
        """Отрисовывает карты признаков (Feature Maps) CNN"""
        # activations имеет форму (1, Channels, Height, Width)
        act = activations.squeeze(0).cpu().numpy()
        num_filters = act.shape[0]
        num_rows = num_filters // num_cols + (num_filters % num_cols > 0)
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*3, num_rows*3))
        for i, ax in enumerate(axes.flat):
            if i < num_filters:
                ax.imshow(act[i], cmap='viridis')
                ax.set_title(f'Filter {i+1}')
            ax.axis('off')
        plt.tight_layout()
        plt.show()
