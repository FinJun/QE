import sys
sys.path.insert(0, '../src')

import argparse
import torch
import torch.nn as nn
import numpy as np
import json
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Tuple, List
import os
from datetime import datetime

from grud import GRUD
from gru_baselines import GRUMean, GRUForward, GRUSimple


class PhysioNetDataset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]


def load_physionet_data(file_path: str, task: str) -> Dict[str, Any]:
    data = np.load(file_path, allow_pickle=True)
    full_data = data['data']

    task_map = {
        'mortality': 'labels_mortality',
        'los': 'labels_los',
        'cardiac': 'labels_cardiac',
        'surgery': 'labels_surgery'
    }

    labels = data[task_map[task]]
    x_mean = data['x_mean']
    n_features = int(data['n_features'])
    n_steps = int(data['n_steps'])
    n_samples = int(data['n_samples'])

    return {
        'data': full_data,
        'labels': labels,
        'x_mean': x_mean,
        'n_features': n_features,
        'n_steps': n_steps,
        'n_samples': n_samples
    }


def create_model(
    model_name: str,
    input_size: int,
    hidden_size: int,
    x_mean: torch.Tensor,
    num_layers: int,
    recurrent_dropout: float = 0.3,
    output_dropout: float = 0.5
) -> nn.Module:
    if model_name == 'GRU-Mean':
        return GRUMean(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=1,
            x_mean=x_mean,
            num_layers=num_layers,
            recurrent_dropout=recurrent_dropout,
            output_dropout=output_dropout
        )
    elif model_name == 'GRU-Forward':
        return GRUForward(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=1,
            x_mean=x_mean,
            num_layers=num_layers,
            recurrent_dropout=recurrent_dropout,
            output_dropout=output_dropout
        )
    elif model_name == 'GRU-Simple':
        return GRUSimple(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=1,
            x_mean=x_mean,
            num_layers=num_layers,
            recurrent_dropout=recurrent_dropout,
            output_dropout=output_dropout
        )
    elif model_name == 'GRU-D':
        return GRUD(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=1,
            x_mean=x_mean,
            num_layers=num_layers,
            recurrent_dropout=recurrent_dropout,
            output_dropout=output_dropout
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


class ModelWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        x = input_tensor[:, 0, :, :].permute(0, 2, 1)
        mask = input_tensor[:, 1, :, :].permute(0, 2, 1)
        delta = input_tensor[:, 2, :, :].permute(0, 2, 1)
        return self.model(x, mask, delta)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_data, batch_labels in train_loader:
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        output = model(batch_data)
        pred = output.squeeze(-1)
        loss = criterion(pred, batch_labels)
        total_loss += loss.item()
        n_batches += 1

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    return total_loss / n_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_preds = []
    all_probs = []
    all_labels = []

    for batch_data, batch_labels in val_loader:
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)

        output = model(batch_data)
        pred = output.squeeze(-1)

        loss = criterion(pred, batch_labels)
        total_loss += loss.item()
        n_batches += 1

        probs = torch.sigmoid(pred).cpu().numpy()
        preds = (probs > 0.5).astype(int)

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(batch_labels.cpu().numpy())

    avg_loss = total_loss / n_batches
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.5

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1': f1,
        'auc': auc
    }


def run_single_experiment(
    model_name: str,
    data_dict: Dict[str, Any],
    hidden_size: int,
    num_layers: int,
    recurrent_dropout: float,
    output_dropout: float,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    n_folds: int,
    seed: int,
    device: torch.device,
    verbose: bool = False
) -> Dict[str, List[float]]:
    data = data_dict['data']
    labels = data_dict['labels']
    x_mean = torch.from_numpy(data_dict['x_mean']).float()
    n_features = data_dict['n_features']

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    results = {'accuracy': [], 'f1': [], 'auc': []}

    for fold, (train_idx, val_idx) in enumerate(skf.split(data, labels)):
        if verbose:
            print(f"    Fold {fold + 1}/{n_folds}", end='', flush=True)

        train_dataset = PhysioNetDataset(data[train_idx], labels[train_idx])
        val_dataset = PhysioNetDataset(data[val_idx], labels[val_idx])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        base_model = create_model(
            model_name, n_features, hidden_size, x_mean,
            num_layers, recurrent_dropout, output_dropout
        )
        model = ModelWrapper(base_model).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10
        )

        best_auc = 0.0
        best_metrics = None
        patience_counter = 0
        patience = 20

        for epoch in range(epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = evaluate(model, val_loader, criterion, device)

            scheduler.step(val_metrics['auc'])

            if val_metrics['auc'] > best_auc:
                best_auc = val_metrics['auc']
                best_metrics = val_metrics.copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        if verbose:
            print(f" - AUC: {best_metrics['auc']:.4f}")

        results['accuracy'].append(best_metrics['accuracy'])
        results['f1'].append(best_metrics['f1'])
        results['auc'].append(best_metrics['auc'])

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default='../data/physionet2012_full.npz')
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--recurrent_dropout', type=float, default=0.3)
    parser.add_argument('--output_dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--output_dir', type=str, default='../results')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    models = ['GRU-Mean', 'GRU-Forward', 'GRU-Simple', 'GRU-D']
    model_hidden_sizes = {
        'GRU-Mean': 64,
        'GRU-Forward': 64,
        'GRU-Simple': 43,
        'GRU-D': 49,
    }

    tasks = ['mortality', 'los', 'cardiac', 'surgery']
    task_names = {
        'mortality': 'Mortality',
        'los': 'LOS < 3 days',
        'cardiac': 'Cardiac',
        'surgery': 'Surgery'
    }

    print("\n" + "=" * 80)
    print("Comparing GRU Models on PhysioNet 2012 (5-fold CV)")
    print("=" * 80)

    all_results = {model: {} for model in models}

    for task in tasks:
        print(f"\n{'=' * 80}")
        print(f"Task: {task_names[task]}")
        print(f"{'=' * 80}")

        data_dict = load_physionet_data(args.data_file, task)
        print(f"Samples: {data_dict['n_samples']}, "
              f"Positive rate: {data_dict['labels'].mean()*100:.1f}%")

        for model_name in models:
            hidden_size = model_hidden_sizes[model_name]
            print(f"\n  {model_name} (hidden={hidden_size}):")

            results = run_single_experiment(
                model_name=model_name,
                data_dict=data_dict,
                hidden_size=hidden_size,
                num_layers=args.num_layers,
                recurrent_dropout=args.recurrent_dropout,
                output_dropout=args.output_dropout,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                n_folds=args.n_folds,
                seed=args.seed,
                device=device,
                verbose=args.verbose
            )

            all_results[model_name][task] = results

            print(f"    Acc: {np.mean(results['accuracy']):.4f} +/- {np.std(results['accuracy']):.4f} | "
                  f"F1: {np.mean(results['f1']):.4f} +/- {np.std(results['f1']):.4f} | "
                  f"AUC: {np.mean(results['auc']):.4f} +/- {np.std(results['auc']):.4f}")

    print("\n" + "=" * 80)
    print("SUMMARY: AUC Scores (mean +/- std)")
    print("=" * 80)

    header = f"{'Model':<15}"
    for task in tasks:
        header += f" {task_names[task]:<18}"
    print(header)
    print("-" * 95)

    for model_name in models:
        row = f"{model_name:<15}"
        for task in tasks:
            results = all_results[model_name][task]
            auc_str = f"{np.mean(results['auc']):.4f}+/-{np.std(results['auc']):.4f}"
            row += f" {auc_str:<18}"
        print(row)

    print("=" * 95)

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    results_json = {
        'timestamp': timestamp,
        'config': {
            'model_hidden_sizes': model_hidden_sizes,
            'num_layers': args.num_layers,
            'recurrent_dropout': args.recurrent_dropout,
            'output_dropout': args.output_dropout,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'n_folds': args.n_folds,
            'seed': args.seed
        },
        'models': models,
        'tasks': tasks,
        'results': {}
    }

    for model_name in models:
        results_json['results'][model_name] = {}
        for task in tasks:
            r = all_results[model_name][task]
            results_json['results'][model_name][task] = {
                'accuracy': {'mean': float(np.mean(r['accuracy'])), 'std': float(np.std(r['accuracy'])), 'folds': r['accuracy']},
                'f1': {'mean': float(np.mean(r['f1'])), 'std': float(np.std(r['f1'])), 'folds': r['f1']},
                'auc': {'mean': float(np.mean(r['auc'])), 'std': float(np.std(r['auc'])), 'folds': r['auc']}
            }

    output_path = os.path.join(args.output_dir, f'physionet_all_models_{timestamp}.json')
    with open(output_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
