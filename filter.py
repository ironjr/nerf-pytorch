from functools import partial
from typing import Callable, Union

import torch
import torch.nn as nn

from mathtool.imresize.core import imresize
from .filter import filter_by_kernel, get_cutoff, get_scale, get_filter


class LowPassFilter(nn.Module):
    def __init__(self, filter_type: str, cutoff_factor: float = 1.0) -> None:
        super(LowPassFilter, self).__init__()
        self.lowpass_fn = get_filter(filter_type)
        self.cutoff_factor = cutoff_factor

    def get_lowpass_fn(
        self,
        scale: torch.Tensor = None,
        cutoff: torch.Tensor = None,
    ) -> Callable:
        cutoff = get_cutoff(scale=scale, cutoff=cutoff)
        cutoff *= self.cutoff_factor
        lowpass_fn = partial(self.lowpass_fn, cutoff=cutoff)
        return lowpass_fn

    def forward(
        self, 
        x: torch.Tensor,
        scale: Union[float, torch.Tensor] = None,
        cutoff: Union[float, torch.Tensor] = None,
        downsample: bool = False,
        get_filter: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Callable]]:
        if (scale is None and cutoff is None) or self.lowpass_fn is None:
            return x

        # Lowpass filter the signal.
        lowpass_fn = partial(self.lowpass_fn, scale=scale, cutoff=cutoff)
        x = lowpass_fn(x)

        # Downsample the signal.
        if downsample:
            scale = get_scale(scale=scale, cutoff=cutoff)
            x = imresize(x, scale=(1.0 / scale))

        if get_filter:
            return x, lowpass_fn
        return x
