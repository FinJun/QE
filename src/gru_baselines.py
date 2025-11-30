import torch
import torch.nn as nn


class GRUMean(nn.Module):
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
        self.register_buffer('x_mean', x_mean.clone().detach().float())

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout if num_layers > 1 else 0.0
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
        x_imputed = torch.where(mask > 0, x, self.x_mean.unsqueeze(0).unsqueeze(0))
        hidden_seq, _ = self.gru(x_imputed)
        h = hidden_seq[:, -1, :]
        h = self.output_dropout(h)
        h = self.batch_norm(h)
        return self.fc_out(h)


class GRUForward(nn.Module):
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
        self.register_buffer('x_mean', x_mean.clone().detach().float())

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout if num_layers > 1 else 0.0
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
        x_imputed = x.clone()
        last_obs = self.x_mean.unsqueeze(0).expand(batch_size, -1).clone()

        for t in range(seq_len):
            observed = mask[:, t, :] > 0
            last_obs = torch.where(observed, x[:, t, :], last_obs)
            missing = mask[:, t, :] == 0
            x_imputed[:, t, :] = torch.where(missing, last_obs, x[:, t, :])

        hidden_seq, _ = self.gru(x_imputed)
        h = hidden_seq[:, -1, :]
        h = self.output_dropout(h)
        h = self.batch_norm(h)
        return self.fc_out(h)


class GRUSimple(nn.Module):
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
        self.register_buffer('x_mean', x_mean.clone().detach().float())

        self.gru = nn.GRU(
            input_size=input_size * 3,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout if num_layers > 1 else 0.0
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
        x_imputed = x.clone()
        last_obs = self.x_mean.unsqueeze(0).expand(batch_size, -1).clone()

        for t in range(seq_len):
            observed = mask[:, t, :] > 0
            last_obs = torch.where(observed, x[:, t, :], last_obs)
            missing = mask[:, t, :] == 0
            x_imputed[:, t, :] = torch.where(missing, last_obs, x[:, t, :])

        combined = torch.cat([x_imputed, mask, delta], dim=-1)
        hidden_seq, _ = self.gru(combined)
        h = hidden_seq[:, -1, :]
        h = self.output_dropout(h)
        h = self.batch_norm(h)
        return self.fc_out(h)


class GRUMeanClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        x_mean: torch.Tensor,
        num_classes: int = 1,
        num_layers: int = 1,
        recurrent_dropout: float = 0.3,
        output_dropout: float = 0.5
    ):
        super().__init__()
        self.model = GRUMean(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=num_classes,
            x_mean=x_mean,
            num_layers=num_layers,
            recurrent_dropout=recurrent_dropout,
            output_dropout=output_dropout
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        x = input_tensor[:, 0, :, :].permute(0, 2, 1)
        mask = input_tensor[:, 1, :, :].permute(0, 2, 1)
        delta = input_tensor[:, 2, :, :].permute(0, 2, 1)
        return self.model(x, mask, delta)


class GRUForwardClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        x_mean: torch.Tensor,
        num_classes: int = 1,
        num_layers: int = 1,
        recurrent_dropout: float = 0.3,
        output_dropout: float = 0.5
    ):
        super().__init__()
        self.model = GRUForward(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=num_classes,
            x_mean=x_mean,
            num_layers=num_layers,
            recurrent_dropout=recurrent_dropout,
            output_dropout=output_dropout
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        x = input_tensor[:, 0, :, :].permute(0, 2, 1)
        mask = input_tensor[:, 1, :, :].permute(0, 2, 1)
        delta = input_tensor[:, 2, :, :].permute(0, 2, 1)
        return self.model(x, mask, delta)


class GRUSimpleClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        x_mean: torch.Tensor,
        num_classes: int = 1,
        num_layers: int = 1,
        recurrent_dropout: float = 0.3,
        output_dropout: float = 0.5
    ):
        super().__init__()
        self.model = GRUSimple(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=num_classes,
            x_mean=x_mean,
            num_layers=num_layers,
            recurrent_dropout=recurrent_dropout,
            output_dropout=output_dropout
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        x = input_tensor[:, 0, :, :].permute(0, 2, 1)
        mask = input_tensor[:, 1, :, :].permute(0, 2, 1)
        delta = input_tensor[:, 2, :, :].permute(0, 2, 1)
        return self.model(x, mask, delta)
