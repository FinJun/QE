import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class GRUDCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        x_mean: torch.Tensor,
        bias: bool = True,
        recurrent_dropout: float = 0.3
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_dropout = recurrent_dropout
        self.register_buffer('x_mean', x_mean.clone().detach().float())

        self.W_gamma_x = nn.Parameter(torch.Tensor(input_size))
        self.b_gamma_x = nn.Parameter(torch.Tensor(input_size)) if bias else None
        self.W_gamma_h = nn.Linear(input_size, hidden_size, bias=bias)

        self.W_xz = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_mz = nn.Linear(input_size, hidden_size, bias=bias)

        self.W_xr = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hr = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_mr = nn.Linear(input_size, hidden_size, bias=bias)

        self.W_xh = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_mh = nn.Linear(input_size, hidden_size, bias=bias)

        self.dropout = nn.Dropout(recurrent_dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.uniform_(self.W_gamma_x, 0.001, 0.01)
        if self.b_gamma_x is not None:
            nn.init.zeros_(self.b_gamma_x)
        for name, param in self.named_parameters():
            if 'W_gamma_x' in name or 'b_gamma_x' in name:
                continue
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(
        self,
        x: torch.Tensor,
        m: torch.Tensor,
        delta: torch.Tensor,
        h: torch.Tensor,
        x_last_obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.b_gamma_x is not None:
            gamma_x = torch.exp(-F.relu(delta * self.W_gamma_x + self.b_gamma_x))
        else:
            gamma_x = torch.exp(-F.relu(delta * self.W_gamma_x))

        gamma_h = torch.exp(-F.relu(self.W_gamma_h(delta)))
        x_last_obs_new = torch.where(m > 0, x, x_last_obs)
        x_imputed = gamma_x * x_last_obs_new + (1 - gamma_x) * self.x_mean
        x_hat = m * x + (1 - m) * x_imputed
        h_decayed = gamma_h * h

        if self.training and self.recurrent_dropout > 0:
            h_decayed = self.dropout(h_decayed)

        z = torch.sigmoid(self.W_xz(x_hat) + self.W_hz(h_decayed) + self.W_mz(m))
        r = torch.sigmoid(self.W_xr(x_hat) + self.W_hr(h_decayed) + self.W_mr(m))
        h_tilde = torch.tanh(self.W_xh(x_hat) + self.W_hh(r * h_decayed) + self.W_mh(m))
        h_new = (1 - z) * h_decayed + z * h_tilde

        return h_new, x_last_obs_new


class GRUD(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        x_mean: torch.Tensor,
        num_layers: int = 1,
        recurrent_dropout: float = 0.3,
        output_dropout: float = 0.5
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.grud_cell = GRUDCell(
            input_size, hidden_size, x_mean,
            recurrent_dropout=recurrent_dropout
        )

        if num_layers > 1:
            self.gru_layers = nn.GRU(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers - 1,
                batch_first=True,
                dropout=recurrent_dropout if num_layers > 2 else 0.0
            )

        self.output_dropout = nn.Dropout(output_dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        delta: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        device = x.device

        h = torch.zeros(batch_size, self.hidden_size, device=device)
        x_last_obs = self.grud_cell.x_mean.unsqueeze(0).expand(batch_size, -1).clone()

        hidden_states = []
        for t in range(seq_len):
            h, x_last_obs = self.grud_cell(
                x[:, t, :],
                mask[:, t, :],
                delta[:, t, :],
                h,
                x_last_obs
            )
            hidden_states.append(h)

        if self.num_layers > 1:
            hidden_seq = torch.stack(hidden_states, dim=1)
            hidden_seq, _ = self.gru_layers(hidden_seq)
            h = hidden_seq[:, -1, :]

        h = self.output_dropout(h)
        h = self.batch_norm(h)
        output = self.fc_out(h)

        return output


class GRUDClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        x_mean: torch.Tensor,
        num_layers: int = 1,
        recurrent_dropout: float = 0.3,
        output_dropout: float = 0.5
    ):
        super().__init__()
        self.grud = GRUD(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=1,
            x_mean=x_mean,
            num_layers=num_layers,
            recurrent_dropout=recurrent_dropout,
            output_dropout=output_dropout
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        x = input_tensor[:, 0, :, :].permute(0, 2, 1)
        mask = input_tensor[:, 1, :, :].permute(0, 2, 1)
        delta = input_tensor[:, 2, :, :].permute(0, 2, 1)
        return self.grud(x, mask, delta)
