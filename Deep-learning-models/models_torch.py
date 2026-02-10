# @Author: xie
# @Email:  xie@umd.edu
# @Last modified by:   xie
# @Last modified time:  2026-02-09
# @License: MIT License
# PyTorch version of models.py — same STARModel API for GeoDL.

import numpy as np
import torch
import torch.nn as nn

from config import *
from helper import get_X_branch_id_by_group
from metrics import *

NUM_LAYERS = 8

# Default device
def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_numpy(y_pred):
    """Convert torch tensor to numpy for API compatibility."""
    if isinstance(y_pred, torch.Tensor):
        return y_pred.detach().cpu().numpy()
    return y_pred


def _ensure_indices(y, num_class):
    """If y is one-hot (N, C), return class indices (N,). Else return y as 1D indices.
    train() accepts either one-hot (from load_demo_data with ONEHOT=True) or 1D class indices."""
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == num_class:
        return np.argmax(y, axis=1)
    return y.ravel().astype(np.int64)


class STARModel:
    """Base class for STAR PyTorch models. Same interface as TF STARModel."""
    def __init__(self, ckpt_path, num_class=NUM_CLASS, mode=MODE, name=None,
                 lr=LEARNING_RATE, batch_size=BATCH_SIZE, epoch_train=EPOCH_TRAIN):
        self.num_class = num_class
        self.mode = mode
        self.ckpt_path = ckpt_path.rstrip("/") + "/"
        self.model_name = name or "model"
        self.lr = lr
        self.batch_size = batch_size
        self.epoch_train = epoch_train
        self.device = _device()
        self.model = None
        self.optimizer = None
        self.criterion = None

    def model_compile(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if self.mode == "classification":
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()
        return

    def train(self, X, y, branch_id=None, mode=MODE, train_type="partition"):
        """Train on (X, y). y can be one-hot (N, num_class) or 1D class indices (N,); converted to indices for CrossEntropyLoss."""
        if branch_id is None:
            print("Error: branch_id is required for deep learning versions.")
        init_epoch_number = len(branch_id) * self.epoch_train
        if train_type == "partition":
            self.set_trainable_layers(len(branch_id) + 1)
        self.model.train()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        if self.mode == "classification":
            y_indices = _ensure_indices(y, self.num_class)
        else:
            y_indices = y.astype(np.float32)
            if y_indices.ndim == 1:
                y_indices = y_indices.reshape(-1, 1)
        n = X.shape[0]
        for epoch in range(init_epoch_number, init_epoch_number + self.epoch_train):
            perm = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                idx = perm[start:end]
                x_b = torch.from_numpy(X[idx]).to(self.device)
                if self.mode == "classification":
                    y_b = torch.from_numpy(y_indices[idx]).long().to(self.device)
                else:
                    y_b = torch.from_numpy(y_indices[idx]).to(self.device)
                self.optimizer.zero_grad()
                out = self.model(x_b)  # logits for classification (no softmax in model)
                loss = self.criterion(out, y_b)
                loss.backward()
                self.optimizer.step()
        return

    def predict(self, X, prob=False):
        """Return predicted probs (classification) or values (regression). Matches TF model.predict() API."""
        self.model.eval()
        X = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            x = torch.from_numpy(X).to(self.device)
            out = self.model(x)
            if self.mode == "classification":
                out = torch.softmax(out, dim=-1)
        return _to_numpy(out)

    def predict_test(self, X, y, X_group, s_branch, prf=True, X_branch_id=None, append_acc=False):
        true = 0
        total = 0
        true_class = np.zeros(self.num_class)
        total_class = np.zeros(self.num_class)
        total_pred = np.zeros(self.num_class)
        err_abs = 0
        err_square = 0
        if X_branch_id is None:
            X_branch_id = get_X_branch_id_by_group(X_group, s_branch)
        for branch_id in np.unique(X_branch_id):
            id_list = np.where(X_branch_id == branch_id)
            X_part = X[id_list]
            y_part = y[id_list]
            self.load(branch_id)
            y_pred = self.predict(X_part)
            if self.mode == "classification":
                true_part, total_part = get_overall_accuracy(y_part, y_pred)
                true += true_part
                total += total_part
                true_class_part, total_class_part, total_pred_part = get_class_wise_accuracy(
                    y_part, y_pred, prf=True
                )
                true_class += true_class_part
                total_class += total_class_part
                total_pred += total_pred_part
            elif self.mode == "regression":
                y_part_flat = np.reshape(y_part, (-1,))
                y_pred_flat = np.reshape(y_pred, (-1,))
                err_abs += np.sum(np.abs(y_part_flat - y_pred_flat))
                err_square += np.sum(np.square(y_part_flat - y_pred_flat))
        if self.mode == "regression":
            return err_abs / y.shape[0], err_square / y.shape[0]
        if prf and self.mode == "classification":
            if append_acc:
                prf_result = list(get_prf(true_class, total_class, total_pred))
                prf_result.append(np.sum(true) / np.sum(total))
                return tuple(prf_result)
            return get_prf(true_class, total_class, total_pred)
        return true / total

    def predict_geodl(self, X, X_group, s_branch, X_branch_id=None):
        if X_branch_id is None:
            X_branch_id = get_X_branch_id_by_group(X_group, s_branch)
        y_pred_full = None
        for branch_id in np.unique(X_branch_id):
            id_list = np.where(X_branch_id == branch_id)
            X_part = X[id_list]
            self.load(branch_id)
            y_pred = self.predict(X_part)
            if y_pred_full is None:
                y_pred_full = np.zeros((X.shape[0],) + y_pred.shape[1:], dtype=y_pred.dtype)
            y_pred_full[id_list] = y_pred
        return y_pred_full

    def set_trainable_layers(self, partition_level, sharing_level=1.5):
        # Match TF: treat each top-level module with parameters as a "layer"
        layers_with_params = [m for m in self.model.modules() if len(list(m.parameters())) > 0]
        total_num_layers = len(layers_with_params)
        num_to_train = int(np.ceil(total_num_layers / (sharing_level ** partition_level)))
        num_to_train = max(num_to_train, 2)
        for m in layers_with_params[:-num_to_train]:
            for p in m.parameters():
                p.requires_grad = False
        for m in layers_with_params[-num_to_train:]:
            for p in m.parameters():
                p.requires_grad = True
        return

    def save(self, branch_id):
        path = self.ckpt_path + self.model_name + "_ckpt_" + branch_id + ".pt"
        torch.save(self.model.state_dict(), path)
        return

    def load(self, branch_id):
        path = self.ckpt_path + self.model_name + "_ckpt_" + branch_id + ".pt"
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        return


class DNNmodel(STARModel):
    """Dense net — PyTorch version."""

    def __init__(self, ckpt_path, layer_size=None, num_layers=NUM_LAYERS, num_class=NUM_CLASS,
                 mode=MODE, name=None, lr=LEARNING_RATE, batch_size=BATCH_SIZE, epoch_train=EPOCH_TRAIN):
        if layer_size is None:
            layer_size = INPUT_SIZE
        super().__init__(ckpt_path, num_class=num_class, mode=mode, name=name or "dnn",
                         lr=lr, batch_size=batch_size, epoch_train=epoch_train)
        self.layer_size = layer_size
        self.num_layers = num_layers
        self.model = self._create_net().to(self.device)
        print("check ckpt path: " + self.ckpt_path)

    def _create_net(self):
        layers = []
        in_dim = INPUT_SIZE
        for _ in range(self.num_layers):
            layers.append(nn.Linear(in_dim, self.layer_size))
            layers.append(nn.ReLU(inplace=True))
            in_dim = self.layer_size
        layers.append(nn.Linear(in_dim, self.num_class))
        # no softmax: CrossEntropyLoss expects logits; softmax applied in predict()
        for m in layers:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.4)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return nn.Sequential(*layers)


class UNetmodel(STARModel):
    """UNet — PyTorch version. Input X: (N, H, W, C); internally uses (N, C, H, W)."""

    def __init__(self, ckpt_path, num_class=NUM_CLASS, mode=MODE, name=None,
                 lr=LEARNING_RATE, batch_size=BATCH_SIZE, epoch_train=EPOCH_TRAIN, training=True):
        super().__init__(ckpt_path, num_class=num_class, mode=mode, name=name or "unet",
                         lr=lr, batch_size=batch_size, epoch_train=epoch_train)
        self.model = UNet(INPUT_SIZE, num_class).to(self.device)
        print("check ckpt path: " + self.ckpt_path)

    def train(self, X, y, branch_id=None, mode=MODE, train_type="partition"):
        # X: (N, H, W, C) for compatibility
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 4:
            X = np.transpose(X, (0, 3, 1, 2))  # NHWC -> NCHW
        return STARModel.train(self, X, y, branch_id=branch_id, mode=mode, train_type=train_type)

    def predict(self, X, prob=False):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 4:
            X = np.transpose(X, (0, 3, 1, 2))
        return STARModel.predict(self, X, prob=prob)


class UNet(nn.Module):
    """UNet module: NCHW. Matches TF layout: down 16,32,64; up 128,64,32; then conv 32 -> num_class."""

    def __init__(self, in_channels, num_class):
        super().__init__()
        self.down1 = _conv_block(in_channels, 16, 3)
        self.down2 = _conv_block(16, 32, 3)
        self.down3 = _conv_block(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        # bottleneck 64 -> up (64) + skip 64 -> 128; then 128 -> 32 + 32 -> 64; then 64 -> 16 + 16 -> 32
        self.up1 = _upsample_block(64, 64, 3)   # in_c=64, out_c=64
        self.up2 = _upsample_block(128, 32, 3)  # in_c=128, out_c=32
        self.up3 = _upsample_block(64, 16, 3)   # in_c=64, out_c=16
        self.conv_final = nn.Conv2d(32, num_class, 3, padding=1)

    def forward(self, x):
        # x: (N, C, H, W)
        s1 = self.down1(x)
        x = self.pool(s1)
        s2 = self.down2(x)
        x = self.pool(s2)
        s3 = self.down3(x)
        x = self.pool(s3)
        x = self.up1(x)
        x = torch.cat([x, s3], dim=1)
        x = self.up2(x)
        x = torch.cat([x, s2], dim=1)
        x = self.up3(x)
        x = torch.cat([x, s1], dim=1)
        x = self.conv_final(x)
        # logits (no softmax); CrossEntropyLoss expects logits; softmax in predict()
        return x


def _conv_block(in_c, out_c, size):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, size, padding=size // 2),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, size, padding=size // 2),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


def _upsample_block(in_c, out_c, size):
    """Upsample block: Conv -> Conv -> ConvTranspose. mid = out_c*2 to match TF upsample(filters)."""
    mid = out_c * 2
    return nn.Sequential(
        nn.Conv2d(in_c, mid, size, padding=size // 2),
        nn.BatchNorm2d(mid),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid, mid, size, padding=size // 2),
        nn.BatchNorm2d(mid),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(mid, out_c, size, stride=2, padding=size // 2, output_padding=1),
    )


class LSTMmodel(STARModel):
    """LSTM on sequences (N, n_time, INPUT_SIZE) — PyTorch version."""

    def __init__(self, ckpt_path, n_time=N_TIME, layer_size=None, num_layers=NUM_LAYERS,
                 num_class=NUM_CLASS, mode=MODE, name=None,
                 lr=LEARNING_RATE, batch_size=BATCH_SIZE, epoch_train=EPOCH_TRAIN):
        if layer_size is None:
            layer_size = INPUT_SIZE
        super().__init__(ckpt_path, num_class=num_class, mode=mode, name=name or "lstm",
                         lr=lr, batch_size=batch_size, epoch_train=epoch_train)
        self.n_time = n_time
        self.layer_size = layer_size
        self.num_layers = num_layers
        self.model = _LSTMNet(n_time, INPUT_SIZE, layer_size, num_layers, num_class, mode).to(self.device)
        print("check ckpt path: " + self.ckpt_path)


class _LSTMNet(nn.Module):
    def __init__(self, n_time, input_size, hidden_size, num_layers, num_class, mode):
        super().__init__()
        self.mode = mode
        self.timedist = nn.ModuleList([
            nn.Linear(input_size, hidden_size) for _ in range(num_layers)
        ])
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_class)
        # no softmax: CrossEntropyLoss expects logits; softmax applied in predict()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.4)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (N, T, F)
        for linear in self.timedist:
            x = torch.relu(linear(x))
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x


def save_single(model, path, name="single"):
    if hasattr(model, "save"):
        model.save(name)
        return
    if hasattr(model, "model"):
        torch.save(model.model.state_dict(), path + "/" + name + ".pt")
