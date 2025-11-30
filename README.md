# GRU-D: Gated Recurrent Unit with Trainable Decay

PyTorch implementation of GRU-D from ["Recurrent Neural Networks for Multivariate Time Series with Missing Values"](https://www.nature.com/articles/s41598-018-24271-9) (Che et al., Scientific Reports 2018).

## Overview

GRU-D addresses missing data in time series through two trainable decay mechanisms:

- **Input Decay (γ_x)**: Decays imputed values toward the empirical mean as time since last observation increases
- **Hidden State Decay (γ_h)**: Decays hidden states to capture temporal irregularity

## Project Structure

```
├── src/
│   ├── grud.py              # GRU-D model (GRUDCell, GRUD, GRUDClassifier)
│   └── gru_baselines.py     # Baseline models (GRU-Mean, GRU-Forward, GRU-Simple)
├── scripts/
│   ├── preprocess_physionet.py      # PhysioNet 2012 preprocessing
│   ├── train_physionet_all.py       # Training script (5-fold CV)
│   └── plot_figure5_paper_style.py  # Decay visualization
```

## Models

| Model | Description |
|-------|-------------|
| **GRU-D** | Full model with input and hidden state decay |
| **GRU-Simple** | GRU with concatenated [x, mask, delta] inputs |
| **GRU-Forward** | GRU with forward-filling imputation |
| **GRU-Mean** | GRU with mean imputation |

## Requirements

```
torch>=1.9.0
numpy
pandas
scikit-learn
matplotlib
```

## Usage

### Data Preprocessing

```bash
python scripts/preprocess_physionet.py \
    --data_dir /path/to/physionet2012 \
    --outcomes /path/to/Outcomes-a.txt \
    --output data/physionet2012.npz
```

### Training

```bash
python scripts/train_physionet_all.py \
    --data_file data/physionet2012_full.npz \
    --epochs 100 \
    --batch_size 64 \
    --n_folds 5 \
    --verbose
```

### Model API

```python
from src.grud import GRUD

model = GRUD(
    input_size=37,
    hidden_size=49,
    output_size=1,
    x_mean=x_mean,
    num_layers=1,
    recurrent_dropout=0.3,
    output_dropout=0.5
)

output = model(x, mask, delta)
```

## Tasks

Evaluated on PhysioNet Challenge 2012:

- **Mortality**: In-hospital death prediction
- **LOS < 3 days**: Length of stay classification
- **Cardiac**: Cardiac condition presence
- **Surgery**: Recovery from surgery

## Reference

```bibtex
@article{che2018recurrent,
  title={Recurrent neural networks for multivariate time series with missing values},
  author={Che, Zhengping and Purushotham, Sanjay and Cho, Kyunghyun and Sontag, David and Liu, Yan},
  journal={Scientific reports},
  volume={8},
  number={1},
  pages={6085},
  year={2018},
  publisher={Nature Publishing Group}
}
```

## License

MIT
