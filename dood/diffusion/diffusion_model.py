from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import numpy as np


class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    def __init__(self, num_timesteps):
        self._weights = np.ones([num_timesteps])

    def weights(self):
        return self._weights


class FeatureNormalization(nn.Module):

    def __init__(self, ft_size, train_min_max=None, queue_size=1024, scale_per_axis=False, scale_to_unit_gauss=False, use_max_std=False, half_scale=False, scale_scaling_factor=None) -> None:
        super().__init__()
        self.queue_size = queue_size
        self.ft_size = ft_size
        self.scale_per_axis = scale_per_axis
        self.scale_to_unit_gauss = scale_to_unit_gauss
        self.use_max_std = use_max_std
        self.half_scale = half_scale
        self.scale_scaling_factor = scale_scaling_factor
        
        self.register_buffer('mins', torch.full((queue_size, ft_size), np.inf))
        self.register_buffer('maxs', torch.full((queue_size, ft_size), -np.inf))
        self.register_buffer('means', torch.full((queue_size, ft_size), np.nan))
        self.register_buffer('stds', torch.full((queue_size, ft_size), np.nan))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        
        self.register_buffer('min', torch.zeros(ft_size))
        self.register_buffer('max', torch.zeros(ft_size))
        self.register_buffer('mean', torch.zeros(ft_size))
        self.register_buffer('std', torch.ones(ft_size))
        self.register_buffer('shift', torch.zeros(ft_size))
        self.register_buffer('scale', torch.ones(ft_size))
        
        assert train_min_max is None, "train_min_max is not supported since introducing the mean and std buffers"

    def _update_running_stats(self, features):
        features = features.detach()
        assert len(features.shape)==2
        new_mins = features.min(0).values
        new_maxs = features.max(0).values
        new_means = features.mean(0)
        new_stds = features.std(0)
        ptr = int(self.queue_ptr)
        self.mins[ptr,:] = new_mins
        self.maxs[ptr,:] = new_maxs
        self.means[ptr,:] = new_means
        self.stds[ptr,:] = new_stds
        self.queue_ptr[0] = (ptr+1) % self.queue_size
        self._update_min_max_mean_std_shift_scale()
        
    def _update_min_max_mean_std_shift_scale(self):
        min, max = self._compute_min_and_max()
        self.min[:] = min
        self.max[:] = max
        
        if self.scale_to_unit_gauss:
            mean, std = self._compute_mean_std()
            self.mean[:] = mean
            self.std[:] = std
        
        shift, scale = self._compute_shift_and_scale()
        self.shift[:] = shift
        self.scale[:] = scale
    
    def _compute_min_and_max(self):
        min = self.mins.min(0).values
        max = self.maxs.max(0).values
        assert torch.any(torch.isfinite(min)) and torch.any(torch.isfinite(min))  # at least one batch was processed -> values are finite
        return min, max
    
    def _compute_mean_std(self):
        valid_means = self.means[torch.isfinite(self.means).all(-1)]
        valid_stds = self.stds[torch.isfinite(self.stds).all(-1)]
        assert len(valid_means) > 0 and len(valid_stds) > 0  # at least one batch was processed
        mean = valid_means.mean(0)
        if self.use_max_std:
            std = valid_stds.max(0).values
        else:
            std = valid_stds.mean(0)
        return mean, std

    def _compute_shift_and_scale(self):
        if not self.scale_to_unit_gauss:
            shift = (self.min + self.max) / 2.
            scale = (self.max - self.min) / 2.
        else:
            shift = self.mean
            scale = self.std
              
        if not self.scale_per_axis:
            scale = scale.max(-1, keepdims=True).values
            
        if self.half_scale:
            scale = scale * 2.
            
        if self.scale_scaling_factor is not None:
            scale = scale / self.scale_scaling_factor
            
        return shift, scale

    def forward(self, x):
        if self.training:
            self._update_running_stats(x)
        shift, scale = self.shift, self.scale
        return (x - shift) / scale
    

class DiffusionModel(nn.Module):
    def __init__(self, diffusion_process, denoiser, train_min_max=None, normalization_queue_size=4096, uniform_timestep_sampler=True, scale_per_axis=False, 
                 scale_to_unit_gauss=False, use_max_std=False, half_scale=False, scale_scaling_factor=None):
        super().__init__()
        
        self.diffusion_process = diffusion_process
        
        self.normalization = FeatureNormalization(denoiser.in_channels, train_min_max=train_min_max, queue_size=normalization_queue_size, scale_per_axis=scale_per_axis,
                                                  scale_to_unit_gauss=scale_to_unit_gauss, use_max_std=use_max_std, half_scale=half_scale, scale_scaling_factor=scale_scaling_factor)

        
        self.denoiser = denoiser
        
        if uniform_timestep_sampler:
            self.timestep_sampler = UniformSampler(diffusion_process.num_timesteps)
        else:
            raise ValueError

    def prior_kl(self, x_start):
        return self.diffusion_process._prior_bpd(x_start)

    def compute_vb(self, x_start, clip_denoised=False, steps=None):
        min_, max_ = self.normalization.min, self.normalization.max
        vb, vb_terms = self.diffusion_process.calc_bpd_loop(self.denoiser, x_start, clip_denoised=clip_denoised, 
                                                            clip_min=min_, clip_max=max_, 
                                                            steps=steps)
        return vb, vb_terms

    def compute_eps_mse(self, x_start, steps=None):
        eps_mse, eps_mse_terms = self.diffusion_process.calc_eps_mse_loop(self.denoiser, x_start, steps=steps)
        return eps_mse, eps_mse_terms
    
    def compute_eps_cos(self, x_start, steps=None):
        eps_cos, eps_cos_terms = self.diffusion_process.calc_eps_cosine_loop(self.denoiser, x_start, steps=steps)
        return eps_cos, eps_cos_terms

    def compute_recon_mse(self, x_start, clip_denoised=False, steps=None):
        min_, max_ = self.normalization.min, self.normalization.max
        recon_mse, recon_mse_terms = self.diffusion_process.calc_recon_mse_loop(self.denoiser, x_start, 
                                                                                clip_denoised=clip_denoised, clip_min=min_, clip_max=max_, 
                                                                                steps=steps)
        return recon_mse, recon_mse_terms

    def avg_mse_loop(self, x_start, clip_denoised=False, steps=None):
        avg_mse = self.diffusion_process.avg_mse_loop(self.denoiser, x_start, steps=steps)
        return avg_mse

    def avg_mse(self, x_start, clip_denoised=False, steps=None):
        avg_mse = self.diffusion_process.mses_parallel(self.denoiser, x_start, steps=steps).mean(dim=(0, -1))
        return avg_mse

    def mses(self, x_start, clip_denoised=False, steps=None):
        mses = self.diffusion_process.mses_parallel(self.denoiser, x_start, steps=steps).mean(dim=-1)
        return mses

    def get_loss_iter(self, x_start):
        N, _ = x_start.shape
        t, weights = self.timestep_sampler.sample(batch_size=N, device=x_start.device)
        loss = self.diffusion_process.p_losses(denoise_fn=self.denoiser, x_start=x_start, t=t, reduce=False)
        loss = (loss * weights).mean()
        return loss
    
    def normalize(self, x):
        return self.normalization(x)

