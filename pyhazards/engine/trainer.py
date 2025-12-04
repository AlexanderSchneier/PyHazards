from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..datasets.base import DataBundle
from ..metrics import MetricBase
from ..utils.hardware import auto_device
from .distributed import select_strategy


class Trainer:
    """
    Lightweight training abstraction with a familiar API:
    fit -> evaluate -> predict.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device | str] = None,
        metrics: Optional[List[MetricBase]] = None,
        strategy: str = "auto",
        mixed_precision: bool = False,
    ):
        self.model = model
        self.device = torch.device(device) if device else auto_device()
        self.metrics = metrics or []
        self.strategy = select_strategy(strategy)
        self.mixed_precision = mixed_precision
        self.model.to(self.device)

    def fit(
        self,
        data: DataBundle,
        train_split: str = "train",
        val_split: Optional[str] = None,
        max_epochs: int = 1,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> None:
        """
        Minimal fit loop that works for tensor-based splits.
        Extend/replace with custom DataLoaders for complex data.
        """
        if optimizer is None or loss_fn is None:
            raise ValueError("optimizer and loss_fn must be provided.")

        train_split_data = data.get_split(train_split)
        train_loader = self._make_loader(train_split_data.inputs, train_split_data.targets, batch_size, num_workers)
        scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision and self.device.type == "cuda")

        self.model.train()
        for _ in range(max_epochs):
            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    out = self.model(x)
                    loss = loss_fn(out, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        if val_split:
            self.evaluate(data, split=val_split)

    def evaluate(
        self,
        data: DataBundle,
        split: str = "test",
        batch_size: int = 64,
        num_workers: int = 0,
    ) -> Dict[str, float]:
        split_data = data.get_split(split)
        loader = self._make_loader(split_data.inputs, split_data.targets, batch_size, num_workers, shuffle=False)
        self.model.eval()
        for metric in self.metrics:
            metric.reset()
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                preds = self.model(x)
                for metric in self.metrics:
                    metric.update(preds, y)
        results: Dict[str, float] = {}
        for metric in self.metrics:
            results.update(metric.compute())
        return results

    def predict(
        self,
        data: DataBundle,
        split: str = "test",
        batch_size: int = 64,
        num_workers: int = 0,
    ) -> List[torch.Tensor]:
        split_data = data.get_split(split)
        loader = self._make_loader(split_data.inputs, split_data.targets, batch_size, num_workers, shuffle=False)
        self.model.eval()
        outputs: List[torch.Tensor] = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(self.device)
                preds = self.model(x)
                outputs.append(preds.cpu())
        return outputs

    def save_checkpoint(self, path: str) -> None:
        torch.save({"model_state": self.model.state_dict()}, path)

    def _make_loader(
        self, inputs: Any, targets: Any, batch_size: int, num_workers: int, shuffle: bool = True
    ) -> Iterable:
        if isinstance(inputs, torch.Tensor) and isinstance(targets, torch.Tensor):
            dataset = TensorDataset(inputs, targets)
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        raise TypeError("Default Trainer only supports tensor inputs/targets. Provide custom DataLoader instead.")
