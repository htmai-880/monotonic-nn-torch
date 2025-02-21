from pathlib import Path
from tqdm import tqdm
import numpy as np
import argparse

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import functional as F

from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam


from monotonic_nn.layer import MonoLinear

def target_fn(x):
    if isinstance(x, np.ndarray):
        # return  0.5 * x ** 3 + x**2 +0.3 * np.sin(5*x)
        return np.log(x)
        # return np.exp(-x**2)
    # return 0.5 * x ** 3 + x**2 + 0.3 * torch.sin(5*x)
    return np.log(x)
    # return torch.exp(-x**2)

def make_dataset(n_samples, std=1, x_min=-20, x_max = 20):
    x = np.random.uniform(x_min, x_max, n_samples)
    x = np.sort(x)
    y = target_fn(x) + np.random.normal(0, std, n_samples)
    return x, y

def eval(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, y in loader:
            y_pred = model(X.to(device))
            loss = criterion(y_pred, y.to(device))
            total_loss += loss.item()
    return total_loss / len(loader)

def main(args):
    n_samples = args.n_samples
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    hidden_dim = args.hidden_dim
    lr = args.lr
    decay = args.decay
    std = args.std
    x_min = args.x_min
    x_max = args.x_max

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare model

    model = nn.Sequential(
        MonoLinear(1, hidden_dim, act="ELU", monotonicity_indicator=1, is_convex=args.convex, is_concave=args.concave),
        *([MonoLinear(hidden_dim, hidden_dim, act="ELU")] * (args.n_layers-1)), 
        MonoLinear(hidden_dim, 1, act=None)
    )
    model.to(device)

    # Prepare optimizer

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=decay)
    criterion = nn.MSELoss()

    # Prepare data

    n_train = round(0.8 * n_samples)
    n_test = n_samples - n_train

    X_train, y_train = make_dataset(n_train, std, x_min, x_max)
    X_test, y_test = make_dataset(n_test, std, x_min, x_max)

    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    eval_every = 10
    with tqdm(np.arange(n_epochs)) as pbar:
        for epoch in pbar:
            model.train()
            train_loss = 0
            for x, y in train_loader:
                optimizer.zero_grad()
                y_pred = model(x.to(device))
                loss = criterion(y_pred, y.to(device))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            if (epoch + 1) % eval_every == 0:
                test_loss = eval(model, test_loader, criterion, device)            
                pbar.set_postfix(train_loss=train_loss, test_loss=test_loss)
                pbar.update(1)

    # Plot results

    X_true = np.linspace(x_min, x_max + (x_max - x_min) * 0.3, 1000)
    y_true = target_fn(X_true)
    plt.figure()
    plt.plot(X_true, y_true, label="True")


    # Get all test predictions
    X_ = []
    y_preds = []
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            X_.append(x)
            y_preds.append(model(x.to(device)).cpu().detach())
        X_ = torch.cat(X_).numpy()
        y_preds = torch.cat(y_preds).numpy()

    plt.scatter(X_, y_preds, label="Predictions", color="red", alpha=0.25, s=2)

    X_inductive = np.linspace(x_max, x_max + (x_max - x_min) * 0.3, 1000)
    with torch.no_grad():
        x = torch.tensor(X_inductive, dtype=torch.float32).unsqueeze(-1)
        y_pred = model(x.to(device)).cpu().detach().numpy()
    
    plt.scatter(X_inductive, y_pred, label="Out of distribution", color="green", alpha=0.25, s=2)

    plt.xlabel("x")
    plt.ylabel("y = f(x)")
    plt.legend()
    plt.grid()
    plt.title("Approximation of the target function")
    # Save figure
    plt.savefig("toy_example_prediction.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=3000)
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--decay", type=float, default=0.0)
    parser.add_argument("--std", type=float, default=0.05)
    parser.add_argument("--x_min", type=float, default=0.5)
    parser.add_argument("--x_max", type=float, default=20)
    parser.add_argument("--convex", action="store_true")
    parser.add_argument("--concave", action="store_true")
    args = parser.parse_args()
    main(args)








