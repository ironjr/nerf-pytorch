import copy
from typing import Callable, Optional, Tuple, Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

#  from mathtool.filter.wrapper import LowPassFilter


def mask2cloud(mask: torch.BoolTensor) -> torch.Tensor:
    """Convert a mask to a point cloud.
    """
    arange = lambda k: torch.arange(k, device=mask.device)
    grids = torch.meshgrid(*[arange(d) for d in mask.shape], indexing='ij')
    points = torch.stack([grid.masked_select(mask) for grid in grids], dim=-1)
    return points


def postprocess(blue: torch.Tensor) -> torch.Tensor:
    """Randomize the blue noise independently to the batch indices.

    Noise post-processing to get higher diversity; apply random
    transformations that preserve the blue noise property.

    Arguments:
        blue (torch.Tensor): Blue noise of shape (B, ..., H, W).
    """
    assert len(blue.shape) >= 3, 'Blue noise shape {} should be more than 3D' \
        .format(str(tuple(blue.shape)))

    for i in range(blue.shape[0]):
       # 1) Random horizontal flip.
        if np.random.random() < 0.5:
            blue[i] = torch.flip(blue[i], dims=(-1,))
        # 2) Random vertical flip.
        if np.random.random() < 0.5:
            blue[i] = torch.flip(blue[i], dims=(-2,))
        # 3) Random 90 degrees rotation.
        if np.random.random() < 0.5:
            blue[i] = torch.rot90(blue[i], 1, dims=(-2, -1))
        # 4) Random roll.
        rollamount = tuple(np.random.randint(s) for s in blue.shape[-2:])
        blue[i] = torch.roll(blue[i], shifts=rollamount, dims=(-2, -1))
    return blue


class BlueNoiseLoader(nn.Module):
    def __init__(
        self,
        noise_path: str,
        repeat_channel: bool = True,
        repeat_spatial: bool = True,
        save_noise: bool = False,
    ) -> None:
        super(BlueNoiseLoader, self).__init__()
        self.repeat_channel = repeat_channel
        self.repeat_spatial = repeat_spatial

        noises = torch.load(noise_path).to(dtype=torch.float32)
        self.set_length = noises.shape[0]
        self.noise_shape = noises.shape[-2:]
        self.noise_size = self.noise_shape[0] * self.noise_shape[1]
        self.register_buffer('data', noises, save_noise)

    def forward(
        self,
        x: torch.Tensor = None,
        shape: Tuple[int] = (1, 1, -1, -1),
    ) -> torch.Tensor:
        """Sample a blue Gaussian noise from the database.

        To ensure the blue-ness after channel-wise linear transformation, the
        variation along the channel dimension is minimized. By simply
        replicating through the channel dimension, maximum correlation between
        the channels are achived.

        Returns:
            torch.Tensor: A blue Gaussian noise tensor with size (B, C, H, W).
        """
        if x is not None:
            shape = x.shape

        # Pre-calculate the amount of repetition and spread of blue noise.
        # In order to enforce correlation between channels, a same blue
        # noise is used to perturb every channel simultaneously.
        _shape = copy.deepcopy(shape)
        repamount = [1,] * 4
        n_sample = shape[0]
        if self.repeat_channel:
            repamount[1] = shape[1]
            shape = (shape[0] * shape[1], 1, *shape[2:])
        else:
            n_sample *= _shape[1]
        spreadamount = [1, 1]
        for i in (2, 3):
            if shape[i] != -1:
                rep = int(np.ceil(_shape[i] / self.noise_shape[i - 2]))
                if self.repeat_spatial:
                    repamount[i] = rep
                else:
                    spreadamount[i - 2] = rep
                    n_sample *= rep

        # Sample blue noise.
        i_kwargs = {'device': self.data.device, 'dtype': torch.long}
        idx = torch.randint(0, self.set_length, (n_sample,), **i_kwargs)
        blue = self.data[idx, ...].unsqueeze(1)
        blue = postprocess(blue)

        # Cut the blue noise into the specified shape.
        blue = blue.repeat(*repamount)
        for i in range(2):
            blue = torch.cat(
                blue.chunk(spreadamount[i], dim=0), dim=(i + 2))
        blue = blue[:, :, :_shape[2], :_shape[3]].view(*_shape)
        return blue


class MaskedSparseUpsampler(nn.Module):
    """Blue noise sparse upsampler.
    
    Upsample and mask to implement sparse upsampling bottleneck.
    """
    def __init__(
        self,
        *args,
        scale: int = 2,
        channel: int = 1,
        spatial_dim: int = 2,
        mask_val: Optional[Union[int, float]] = 0.0,
        learned_token: bool = False,
        **kwargs,
    ) -> None:
        super(MaskedSparseUpsampler, self).__init__()
        self.scale = scale
        self.channel = channel
        self.spatial_dim = spatial_dim
        self.learned_token = learned_token

        if isinstance(mask_val, int):
            dtype = torch.long
            assert not learned_token, 'Integral type tensor is not learnable!'
        else:
            dtype = torch.float32
        shape = (1, channel) + (1,) * spatial_dim
        if mask_val is None:
            mask_token = torch.randn(shape, dtype=dtype)
        else:
            mask_token = mask_val * torch.ones(shape, dtype=dtype)
        if learned_token:
            self.mask_token = nn.Parameter(mask_token)
        else:
            self.register_buffer('mask_token', mask_token)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(scale={self.scale}, channel="
            f"{self.channel}, spatial_dim={self.spatial_dim}, learned_token="
            f"{self.learned_token})")

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        target_size: Optional[Tuple[int]] = None,
    ) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        scale_factor = self.scale if target_size is None else None
        x = F.interpolate(x, size=target_size, scale_factor=scale_factor)
        if mask is not None:
            x = x * mask + self.mask_token * mask.logical_not()
        x = x.to(dtype=dtype)
        return x


class JitteredSparseUpsampler(MaskedSparseUpsampler):
    """Jittered sparse upsampler with various output option.
    
    This implements information-preserving sparse upsampling. The default
    method is to use spatial upsampling using the pixelshuffle.
    """
    def __init__(self, *args, outtype: str = 'channel', **kwargs) -> None:
        super(JitteredSparseUpsampler, self).__init__(*args, **kwargs)
        assert outtype in ('last', 'channel_merge', 'channel_separate',
            'pixelshuffle'), \
            'Found the unknown `outtype`: {}.'.format(outttype)
        self.outtype = outtype

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(scale={self.scale}, channel="
            f"{self.channel}, spatial_dim={self.spatial_dim}, learned_token="
            f"{self.learned_token}, outtype={self.outtype})")

    def forward(self, x: torch.Tensor, get_mask: bool = False) -> torch.Tensor:
        """
        Args:
          - x (torch.Tensor): An input tesnor with shape (B, Cin, ...).
        
        Returns:
            torch.Tensor: The upsampled tensor in channel dimension. The output
                shape can vary with the prespecified :outtype: argument.
                if :outtype: == 'last':
                    (B, (:self.scale:**:self.spatial_dim:), ..., Cin).
                elif :outtype: == 'channel_merge':
                    (B, Cin*(:self.scale:**:self.spatial_dim:), ...).
                elif :outtype: == 'channel_separate':
                    (B, Cin, (:self.scale:**:self.spatial_dim:), ...).
                elif :outtype: == 'pixelshuffle':
                    (B, Cin, H*:self.scale:, ...)
            Optional[torch.BoolTensor]: The mask tensor used in the sparse
                upsampling module, with the same shape as the output
                except Cin is replaced with 1.
        """
        assert x.shape[1] == self.channel or self.channel == 1, \
            'The input channel {} must match the layer\'s {} or be 1.' \
            .format(x.shape[1], self.channel)

        repeat = self.scale ** self.spatial_dim
        shape = list(x.shape)
        channel = shape[1]
        shape[1] = 1

        mask = torch.randint(repeat, shape, device=x.device)
        mask = F.one_hot(mask, num_classes=repeat)
        mask_ = mask.expand((-1, channel,) + (-1,) * (len(mask.shape) - 2))
        out = (mask_ * x.unsqueeze(-1)
            + mask_.logical_not() * self.mask_token.unsqueeze(-1))

        if self.outtype == 'last':
            pass
        elif 'channel' in self.outtype: # 'channel_merge', 'channel_separate'
            perm = [0, 1, self.spatial_dim + 2]
            perm += list(range(2, self.spatial_dim + 2))
            out = out.permute(perm)
            if get_mask:
                mask = mask.permute(perm)

            if self.outtype == 'channel_merge':
                shape[1] = channel * repeat
                out = out.reshape(shape)
                if get_mask:
                    mask = mask.unsqueeze(1)
            elif self.outtype == 'channel_separate':
                pass
            else:
                raise ValueError(
                    'Unknown `self.outtype`: {}.'.format(self.outtype))
        elif self.outtype == 'pixelshuffle':
            if self.spatial_dim == 1:
                out = rearrange(out, 'b c d d2 -> b c (d d2)')
                if get_mask:
                    mask = rearrange(mask, 'b () d d2 -> b () (d d2)')
            elif self.spatial_dim == 2:
                s = self.scale
                out = rearrange(out,
                    'b c h w (h2 w2) -> b c (h h2) (w w2)', h2=s)
                if get_mask:
                    mask = rearrange(mask,
                        'b () h w (h2 w2) -> b () (h h2) (w w2)', h2=s)
            elif self.spatial_dim == 3:
                s = self.scale
                out = rearrange(out,
                    'b c h w d (h2 w2 d2) -> b c (h h2) (w w2) (d d2)',
                    h2=s, w2=s)
                if get_mask:
                    mask = rearrange(mask,
                        'b () h w d (h2 w2 d2) -> b () (h h2) (w w2) (d d2)',
                        h2=s, w2=s)
            else:
                raise NotImplementedError(
                    'The case of `self.outtype` = {} and `self.spatial_dim` '
                    '> 3 is not implemented.'
                    .format(self.outtype, self.spatial_dim))
        else:
            raise ValueError(
                'Unknown `self.outtype`: {}.'.format(self.outtype))

        if get_mask:
            return out, mask
        return out


class BlueNoiseUpsampler(MaskedSparseUpsampler):
    def __init__(
        self,
        *args,
        scale: int = 2,
        index_path: str = None,
        mask_path: str = None,
        noise_path: str = None,
        save_blue: bool = False,
        repeat_channel: bool = True,
        repeat_spatial: bool = True,
        noise_type: str = 'mult',
        **kwargs,
    ) -> None:
        super(BlueNoiseUpsampler, self).__init__(scale=scale, **kwargs)
        self.save_blue = save_blue
        self.repeat_channel = repeat_channel
        self.repeat_spatial = repeat_spatial
        self.noise_type = noise_type

        assert index_path or noise_path or mask_path, \
            'Please set up the path for either noise or mask.'
        noise_kwargs = {
            'repeat_channel': repeat_channel,
            'repeat_spatial': repeat_spatial,
            'save_noise': save_blue,
        }
        if index_path is not None:
            self.index = BlueNoiseLoader(index_path, **noise_kwargs)
        if noise_path is not None:
            self.noise = BlueNoiseLoader(noise_path, **noise_kwargs)
        if mask_path is not None:
            self.mask = BlueNoiseLoader(mask_path, **noise_kwargs)
        #  self.lpf = LowPassFilter('gauss')

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(scale={self.scale}, channel="
            f"{self.channel}, spatial_dim={self.spatial_dim}, learned_token="
            f"{self.learned_token}, noise_type={self.noise_type})")

    def forward(
        self,
        x: torch.Tensor,
        get_maturity: bool = False,
        get_raw: bool = False,
        target_size: Optional[Tuple[int]] = None,
        noise_lv: Optional[Union[torch.Tensor, float]] = None,
    ) -> torch.Tensor:
        if target_size is not None or self.scale != 1:
            scale = self.scale if target_size is None else None
            x = F.interpolate(x, size=target_size, scale_factor=scale)
            #  x = self.lpf(x, scale=self.scale)

        index = None
        mask = None
        noise = None

        shape = list(x.shape)
        shape[1] = 1
        if self.noise_type == 'mult':
            mask = self.mask(shape=shape)
            x = x * mask
            if get_maturity:
                maturity = mask
        elif self.noise_type == 'add':
            noise = self.noise(shape=shape)
            x = x + noise
            if noise_lv is not None:
                noise *= noise_lv
            if get_maturity:
                # Assume that the signal is uniform in (-1, 1)--this gives
                # its standard deviation be 1/3.
                maturity = (3.0 * noise.pow(2) + 1.0).rsqrt()
        elif self.noise_type == 'affine':
            mask = self.mask(shape=shape)
            noise = self.noise(shape=shape)
            if noise_lv is not None:
                noise *= noise_lv
            x = x * mask + noise
            if get_maturity:
                maturity = mask * (3.0 * noise.pow(2) + 1.0).rsqrt()
        elif self.noise_type == 'mask':
            index = self.index(shape=shape)
            # Use `noise_lv` argument to control the mask probability.
            index = index / self.index.noise_size
            prob = self.scale ** -2 if noise_lv is None else noise_lv
            mask = index >= (1.0 - prob)
            x = x * mask
            if get_maturity:
                maturity = mask
        else:
            raise ValueError(
                'Invalid `noise_type` {}.'.format(self.noise_type))

        ret = [x]
        if get_maturity:
            ret.append(maturity)
        if get_raw:
            if index is not None:
                ret.append(index)
            if mask is not None:
                ret.append(mask)
            if noise is not None:
                ret.append(noise)
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)


"""
class FilteredNoiseUpsampler(MaskedSparseUpsampler):
    def __init__(
        self,
        *args,
        scale: int = 2,
        noise_lv: float = 1.0,
        noise_type: str = 'mult',
        **kwargs,
    ) -> None:
        super(FilteredNoiseUpsampler, self).__init__(scale=scale, **kwargs)
        self.noise_lv = noise_lv
        self.noise_type = noise_type

        self.lpf = LowPassFilter('gauss')

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(scale={self.scale}, channel="
            f"{self.channel}, spatial_dim={self.spatial_dim}, learned_token="
            f"{self.learned_token}, noise_type={self.noise_type}, "
            f"noise_lv={self.noise_lv})")

    def get_mask(
        self,
        x: torch.Tensor,
        repeat_channel: bool = False,
    ) -> torch.Tensor:
        mask = torch.rand_like(x[:, (0,), ...] if repeat_channel else x)
        mask = mask - self.lpf(mask, scale=self.scale)
        return mask

    def get_noise(
        self,
        x: torch.Tensor,
        repeat_channel: bool = False,
        noise_lv: Optional[Union[torch.Tensor, float]] = None,
    ) -> torch.Tensor:
        if noise_lv is None:
            noise_lv = self.noise_lv
        noise = torch.randn_like(x[:, (0,), ...] if repeat_channel else x)
        noise = noise - self.lpf(noise, scale=self.scale)
        return noise * noise_lv

    def forward(
        self,
        x: torch.Tensor,
        get_maturity: bool = False,
        get_raw: bool = False,
        repeat_channel: bool = False,
        target_size: Optional[Tuple[int]] = None,
        noise_lv: Optional[Union[torch.Tensor, float]] = None,
    ) -> Tuple[torch.Tensor]:
        if target_size is not None or self.scale != 1:
            scale = self.scale if target_size is None else None
            x = F.interpolate(x, size=target_size, scale_factor=scale)
            x = self.lpf(x, scale=self.scale)
        if noise_lv is None:
            noise_lv = self.noise_lv

        mask = None
        noise = None
        if self.noise_type == 'mult':
            mask = self.get_mask(x, repeat_channel)
            x = x * mask
            if get_maturity:
                maturity = mask
        elif self.noise_type == 'add':
            noise = self.get_noise(x, repeat_channel, noise_lv)
            x = x + noise
            if get_maturity:
                # Assume that the signal is uniform in (-1, 1)--this gives
                # its standard deviation be 1/3.
                maturity = (3.0 * noise.pow(2) + 1.0).rsqrt()
        elif self.noise_type == 'affine':
            mask = self.get_mask(x, repeat_channel)
            noise = self.get_noise(x, repeat_channel, noise_lv)
            x = x * mask + noise
            if get_maturity:
                maturity = mask * (3.0 * noise.pow(2) + 1.0).rsqrt()
        else:
            raise ValueError(
                'Invalid `noise_type` {}.'.format(self.noise_type))

        ret = [x]
        if get_maturity:
            ret.append(maturity)
        if get_raw:
            if mask is not None:
                ret.append(mask)
            if noise is not None:
                ret.append(noise)
        if len(ret) == 1:
            return ret[0]
        else:
            return ret
"""
