import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import predictio_dataset
from sklearn.metrics import roc_auc_score, auc, roc_curve
from model.phznet import PHZNet
import matplotlib.pyplot as plt

from config import folder, cohort

# Set seed for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Determine device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

fig, ax = plt.subplots()
clr_dict = {"EDC": "blue", "Tab_Transformer": "violet", "MLP": "coral", "PHZNet": "dodgerblue", "MLPP": "hotpink"}

def process_data(data, device):
    img, label = data
    img = img.view(img.size(0), -1).to(device)
    label = label.to(device)
    return img, label

def calculate_metrics(pred, label, loss):
    pred_label = torch.max(pred, 1)[1]
    correct = (pred_label == label).sum().item()
    tp = ((label == 1) & (pred_label == 1)).sum().item()
    fn = ((label == 1) & (pred_label == 0)).sum().item()
    fp = ((label == 0) & (pred_label == 1)).sum().item()
    precision = tp / (tp + fp) if tp + fp != 0 else 0
    recall = tp / (tp + fn) if tp + fn != 0 else 0
    f1_score = (2 * recall * precision) / (precision + recall) if precision + recall != 0 else 0
    return {'loss': loss, 'accuracy': correct / label.size(0), 'precision': precision, 'recall': recall, 'f1_score': f1_score}

def train_model(model, optimizer, criterion, batch_size, dataset, num_epochs=100):
    lengths = [int(len(dataset)*0.8+0.5), int(len(dataset)*0.2+0.5)]   
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, lengths)   
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize best loss and accuracy
    best_loss = float('inf')
    best_acc = 0

    for epoch in range(num_epochs):
        model.train()
        train_metrics = {'loss': 0, 'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0}
        for data in train_loader:
            img, label = process_data(data, device)
            out = model(img)
            loss = criterion(out, label)
            train_metrics = {k: train_metrics[k] + v for k, v in calculate_metrics(out, label, loss).items()}

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Compute mean of metrics
        train_metrics = {k: v / len(train_loader) for k, v in train_metrics.items()}

        if epoch % 50 == 0:
            print(f"Epoch {epoch}/{num_epochs-1}")
            print('-' * 10)
            print(f"[ Train | {epoch:03d}/{num_epochs:03d} ] loss = {train_metrics['loss']:.5f}, "
                  f"acc = {train_metrics['accuracy']:.5f}, "
                  f"precision = {train_metrics['precision']:.5f}, "
                  f"recall = {train_metrics['recall']:.5f}, "
                  f"F1_score = {train_metrics['f1_score']:.5f}")

        model.eval()
        with torch.no_grad():
            val_metrics = {'loss': 0, 'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0}
            for data in test_loader:
                img, label = process_data(data, device)
                out = model(img)
                loss = criterion(out, label)
                val_metrics = {k: val_metrics[k] + v for k, v in calculate_metrics(out, label, loss).items()}

            val_metrics = {k: v / len(test_loader) for k, v in val_metrics.items()}

        if epoch % 50 == 0:
            print(f"[ Val   | {epoch:03d}/{num_epochs:03d} ] loss = {val_metrics['loss']:.5f}, "
                  f"acc = {val_metrics['accuracy']:.5f}, "
                  f"precision = {val_metrics['precision']:.5f}, "
                  f"recall = {val_metrics['recall']:.5f}, "
                  f"F1_score = {val_metrics['f1_score']:.5f}")

        if best_acc < val_metrics['accuracy']:
            # torch.save(model.state_dict(), f'./model/models/model_{model.__class__.__name__}_{optimizer.param_groups[0]["lr"]}_{batch_size}.pth')
            best_acc = val_metrics['accuracy']
            print(f"[ best   | {epoch:03d}/{num_epochs:03d} ] loss = {val_metrics['loss']:.5f}, "
                  f"acc = {val_metrics['accuracy']:.5f}, "
                  f"precision = {val_metrics['precision']:.5f}, "
                  f"recall = {val_metrics['recall']:.5f}, "
                  f"F1_score = {val_metrics['f1_score']:.5f}")


def run_experiment(models, learning_rates, batch_sizes, num_epochs, device, criterion, dataset, folder):
    print(f'-------------{folder}-----------')
    for lr in learning_rates:
        for bs in batch_sizes:
            print(f'learning_rate: {lr}, batch_size: {bs}')
            fig, ax = plt.subplots()
            
            for model_info in models:
                print(f'-------------------------{model_info["name"]}------------------------')
                model = model_info["model"]
                model = model.to(device)
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

                train_model(model, optimizer, criterion, bs, dataset, num_epochs)
                print('\n\n')

            print('\n\n\n\n')

            fig.legend()
            fig.savefig(f"./plots/roc_{lr}_{bs}.png", bbox_inches="tight")


if __name__ == '__main__':
    batch_size = 1
    learning_rate = 0.01
    num_epochs = 500
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = nn.CrossEntropyLoss()

    models = [
        {"name": "phznet", "model": PHZNet(in_features=16093, out_features=2)},
        # {"name": "edc", "model": EDC()},
        # {"name": "MLPP", "model": MLPP()},
        # {"name": "Tab_Transformer", "model": Tab_Transformer()},
        # {"name": "MLP", "model": MLP(n_features=dataset.n_features)}
    ]

    learning_rates = [0.0001, 0.0005, 0.001, 0.002]
    batch_sizes = [32, 64, 128]

    folder = '3/augmented'
    cohort = 'augmented'
    dataset = predictio_dataset(folder=folder, cohort=cohort)
    run_experiment(models, learning_rates, batch_sizes, num_epochs, device, criterion, dataset, folder)
