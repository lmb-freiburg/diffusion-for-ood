import importlib
import collections
import re
import random

import numpy as np
import torch
# from torch._six import string_classes
string_classes = str
import pytoml


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))
    
    
def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
                + (mean1 - mean2)**2 * torch.exp(-logvar2))
    
    
def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def gaussian_log_likelihood(x, *, means, log_stddevs, eps=None):
    """
    Compute the log-likelihoods of values x of Gaussian distributions with given means and .
    :param x: the target values.
    :param means: the Gaussian mean Tensor.
    :param log_stddevs: the Gaussian log stddev Tensor.
    :param eps: the minimum variance. If None, no clipping is applied. torch.nn.GaussianNLLLoss uses 1e-6.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_stddevs.shape
    centered_x = x - means
    
    log_var = 2 * log_stddevs
    if eps is not None:
        log_var = log_var.clamp(min=np.log(eps))
        log_stddevs = log_var * 0.5
    var = torch.exp(log_var)
    inv_var = torch.exp(-log_var)
    
    return -((centered_x**2) * 0.5 * inv_var) - log_stddevs - (0.5 * np.log(2 * np.pi))

