from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseModelAdapter(ABC):
    @abstractmethod
    def tokenize_captions(self, captions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def generate_repeated(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_return_sequences: int,
        max_len: int,
        temperature: float,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def score_sequences(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        generated: torch.Tensor,
        use_reference: bool = False,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def decode_scores(self, generated: torch.Tensor) -> list:
        raise NotImplementedError

    @abstractmethod
    def trainable_parameters(self):
        raise NotImplementedError

    @abstractmethod
    def save_adapter(self, output_dir: str) -> None:
        raise NotImplementedError
