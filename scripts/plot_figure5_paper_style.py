import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from grud import GRUDClassifier

PAPER_VAR_ORDER = [
    'Cholesterol', 'TroponinI', 'TroponinT', 'Albumin', 'ALP', 'ALT', 'AST',
    'Bilirubin', 'Lactate', 'SaO2', 'WBC', 'Glucose', 'Na', 'Mg', 'HCO3',
    'BUN', 'Creatinine', 'Platelets', 'K', 'HCT', 'PaO2', 'PaCO2', 'pH',
    'FiO2', 'RespRate', 'GCS', 'Temp', 'Weight', 'Urine', 'MAP', 'DiasABP',
    'SysABP', 'HR'
]

MISSING_RATES = {
    'Cholesterol': 0.9989, 'TroponinI': 0.9984, 'TroponinT': 0.9923,
    'Albumin': 0.9915, 'ALP': 0.9888, 'ALT': 0.9885, 'AST': 0.9885,
    'Bilirubin': 0.9884, 'Lactate': 0.9709, 'SaO2': 0.9705, 'WBC': 0.9532,
    'Glucose': 0.9528, 'Na': 0.9508, 'Mg': 0.9507, 'HCO3': 0.9507,
    'BUN': 0.9496, 'Creatinine': 0.9493, 'Platelets': 0.9489, 'K': 0.9477,
    'HCT': 0.9338, 'PaO2': 0.9158, 'PaCO2': 0.9157, 'pH': 0.9118,
    'FiO2': 0.8830, 'RespRate': 0.8053, 'GCS': 0.7767, 'Temp': 0.6915,
    'Weight': 0.5452, 'Urine': 0.5095, 'MAP': 0.2141, 'DiasABP': 0.2054,
    'SysABP': 0.2052, 'HR': 0.1984
}


def load_model_and_data():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_path = '../data/physionet2012.npz'
    data = np.load(data_path, allow_pickle=True)

    input_data = np.array([np.array(x, dtype=np.float32) for x in data['input']])
    masking = np.array([np.array(x, dtype=np.float32) for x in data['masking']])
    input_data = np.nan_to_num(input_data, nan=0.0)
    masking = np.nan_to_num(masking, nan=0.0)

    var_names = [str(v) for v in data['variables']]

    N, T, D = masking.shape
    input_t = input_data.transpose(0, 2, 1)
    masks_t = masking.transpose(0, 2, 1)
    x_sum = (input_t * masks_t).sum(axis=(0, 2))
    mask_sum = masks_t.sum(axis=(0, 2))
    x_mean = np.where(mask_sum > 0, x_sum / mask_sum, 0.0)
    x_mean = torch.tensor(x_mean, dtype=torch.float32, device=device)

    model = GRUDClassifier(
        input_size=D,
        hidden_size=49,
        x_mean=x_mean,
        num_layers=1,
        recurrent_dropout=0.3,
        output_dropout=0.5
    ).to(device)

    model_path = '../results/grud_physionet_model_v3.pt'
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model, var_names


def extract_decay_params(model):
    grud_cell = model.grud.grud_cell

    W_gamma_x = grud_cell.W_gamma_x.detach().cpu().numpy()
    b_gamma_x = grud_cell.b_gamma_x.detach().cpu().numpy() if grud_cell.b_gamma_x is not None else np.zeros_like(W_gamma_x)
    W_gamma_h = grud_cell.W_gamma_h.weight.detach().cpu().numpy()

    return W_gamma_x, b_gamma_x, W_gamma_h


def compute_decay_curve(W, b, delta_range):
    decay_val = W * delta_range[:, None] + b
    decay_val = np.maximum(0, decay_val)
    gamma = np.exp(-decay_val)
    return gamma


def plot_figure5_paper_style(W_gamma_x, b_gamma_x, W_gamma_h, var_names, save_path):
    var_to_idx = {v: i for i, v in enumerate(var_names)}

    fig = plt.figure(figsize=(14, 11))

    gs_top = gridspec.GridSpec(3, 11, figure=fig,
                                top=0.92, bottom=0.42,
                                left=0.05, right=0.98,
                                hspace=0.35, wspace=0.25)

    delta_range = np.linspace(0, 24, 100)

    fig.text(0.5, 0.96, 'Plots of Input Decay Function for All Features from PhysioNet Dataset',
             ha='center', fontsize=12, fontweight='bold')
    fig.text(0.5, 0.935, r'(a) x-axis, time interval $\delta_t^d$ between 0 and 24 hours; y-axis, value of decay rate $\gamma_{x_t}^d$.',
             ha='center', fontsize=10)

    for plot_idx, var_name in enumerate(PAPER_VAR_ORDER):
        if var_name not in var_to_idx:
            continue

        var_idx = var_to_idx[var_name]
        row = plot_idx // 11
        col = plot_idx % 11

        ax = fig.add_subplot(gs_top[row, col])

        decay_val = W_gamma_x[var_idx] * delta_range + b_gamma_x[var_idx]
        decay_val = np.maximum(0, decay_val)
        gamma = np.exp(-decay_val)

        ax.plot(delta_range, gamma, 'b-', linewidth=1.2)
        ax.set_ylim(-0.05, 1.1)
        ax.set_xlim(0, 24)
        ax.set_title(var_name, fontsize=7, pad=2)

        if row == 2:
            ax.set_xlabel('Hour(s)', fontsize=6)
        else:
            ax.set_xticklabels([])

        if col == 0:
            ax.set_ylabel(r'$\gamma_{x_t}^d$', fontsize=8)
        else:
            ax.set_yticklabels([])

        ax.tick_params(axis='both', labelsize=5, length=2)
        ax.set_xticks([0, 12, 24])
        ax.set_yticks([0, 0.5, 1])

    gs_bottom = gridspec.GridSpec(1, 10, figure=fig,
                                   top=0.32, bottom=0.08,
                                   left=0.05, right=0.98,
                                   wspace=0.35)

    fig.text(0.5, 0.38, 'Histograms of Hidden State Decay Weights for 10 Features from PhysioNet Dataset',
             ha='center', fontsize=12, fontweight='bold')
    fig.text(0.5, 0.355, r'(b) x-axis, value of decay parameters $W_{\gamma_h}$; y-axis, frequency; MR, missing rate.',
             ha='center', fontsize=10)

    sorted_vars = sorted(MISSING_RATES.keys(), key=lambda x: MISSING_RATES[x], reverse=True)
    highest_mr = sorted_vars[:5]
    lowest_mr = sorted_vars[-5:][::-1]

    selected_vars = highest_mr + lowest_mr

    for plot_idx, var_name in enumerate(selected_vars):
        if var_name not in var_to_idx:
            continue

        var_idx = var_to_idx[var_name]
        ax = fig.add_subplot(gs_bottom[0, plot_idx])

        weights = W_gamma_h[:, var_idx]

        ax.hist(weights, bins=15, color='steelblue', edgecolor='black', linewidth=0.5)
        ax.set_yscale('log')
        ax.set_xlim(-0.35, 0.35)
        ax.set_ylim(1, 50)

        mr = MISSING_RATES.get(var_name, 0.5)
        ax.set_title(f'{var_name}\nMR: {mr:.4f}', fontsize=7, pad=2)

        if plot_idx == 0 or plot_idx == 5:
            ax.set_ylabel('Frequency', fontsize=7)

        ax.tick_params(axis='both', labelsize=5, length=2)
        ax.set_xticks([-0.2, 0, 0.2])

    fig.text(0.5, 0.02, r'Weight ($W_{\gamma_h}$) for Hidden State Decay', ha='center', fontsize=10)

    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved figure to {save_path}")


def main():
    print("Loading model and data...")
    model, var_names = load_model_and_data()

    print("Extracting decay parameters...")
    W_gamma_x, b_gamma_x, W_gamma_h = extract_decay_params(model)

    print(f"W_gamma_x range: [{W_gamma_x.min():.4f}, {W_gamma_x.max():.4f}]")
    print(f"b_gamma_x range: [{b_gamma_x.min():.4f}, {b_gamma_x.max():.4f}]")
    print(f"W_gamma_h range: [{W_gamma_h.min():.4f}, {W_gamma_h.max():.4f}]")

    print("\nGenerating Figure 5 in paper style...")
    save_path = '../results/figure5_paper_style.png'
    plot_figure5_paper_style(W_gamma_x, b_gamma_x, W_gamma_h, var_names, save_path)

    print("Done!")


if __name__ == '__main__':
    main()
