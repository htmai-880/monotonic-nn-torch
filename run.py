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


from src.layer import MonoLinear
from src.data import get_train_n_test_data

from sklearn.metrics import classification_report, f1_score, recall_score, precision_score



def make_dataset(name):
    train, test = get_train_n_test_data(name)
    print(train.head())
    y_train = train["ground_truth"].values
    X_train = train.drop(columns=["ground_truth"])
    y_test = test["ground_truth"].values
    X_test = test.drop(columns=["ground_truth"])
    return X_train, y_train, X_test, y_test

def eval(model, X, y, device, print_report=False):
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.Tensor(X, device=device))
        y_pred = y_pred.cpu().numpy()
        y_pred = (y_pred > 0.5).astype(int)
    if print_report:
        print(classification_report(y, y_pred))
    return {
        "f1": f1_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "precision": precision_score(y, y_pred)
    }

def main(args):
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    hidden_dim = args.hidden_dim
    lr = args.lr
    decay = args.decay

    X_train, y_train, X_test, y_test = make_dataset("heart")
    monotonicity_indicator_dict = {
        "age": 0,
        "sex": 0,
        "cp": 0,
        "trestbps": 1,
        "chol": 1,
        "fbs": 0,
        "restecg": 0,
        "thalach": 0,
        "exang": 0,
        "oldpeak": 0,
        "slope": 0,
        "ca": 0,
        "thal": 0,
    }
    # Get features in order
    features = X_train.columns
    monotonicity_indicator = [monotonicity_indicator_dict[feature] for feature in features]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare model
    input_dim = len(features)

    model = nn.Sequential(
        MonoLinear(input_dim, hidden_dim, act="ELU", monotonicity_indicator=monotonicity_indicator),
        *([MonoLinear(hidden_dim, hidden_dim, act="ELU")] * (args.n_layers - 1)), 
        MonoLinear(hidden_dim, 1, act=None)
    )
    model.to(device)

    # Prepare optimizer

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=decay)
    criterion = nn.BCEWithLogitsLoss()

    # Prepare data
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()


    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    eval_every = 10
    with tqdm(np.arange(n_epochs)) as pbar:
        for epoch in pbar:
            model.train()
            train_loss = 0
            for x, y in train_loader:
                optimizer.zero_grad()
                y_pred = model(x.to(device)).squeeze(-1)
                loss = criterion(y_pred, y.to(device))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            if (epoch + 1) % eval_every == 0:
                test_metrics = eval(model, X_test, y_test, device, False)
                pbar.set_postfix(train_loss=train_loss, f1=test_metrics["f1"], recall=test_metrics["recall"], precision=test_metrics["precision"])
                pbar.update(1)

    # Plot results
    test_metrics = eval(model, X_test, y_test, device, True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=3000)
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--decay", type=float, default=0.0)
    args = parser.parse_args()
    main(args)








