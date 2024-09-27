import numpy as np
import torch
from scipy.special import expit

from ..utils import normal_kl, gaussian_log_likelihood, mean_flat


def get_beta_schedule(schedule_type, *, num_diffusion_steps, beta_start=None, beta_end=None):
    if schedule_type == 'linear':
        # default beta_start and beta_end values taken from https://github.com/openai/point-e/blob/main/point_e/diffusion/gaussian_diffusion.py
        beta_start = 1000 / num_diffusion_steps * 0.0001 if beta_start is None else beta_start
        beta_end = 1000 / num_diffusion_steps * 0.02 if beta_end is None else beta_end
        betas = np.linspace(beta_start, beta_end, num_diffusion_steps)
        
    elif schedule_type == 'warm0.1':
        betas = beta_end * np.ones(num_diffusion_steps, dtype=np.float64)
        warmup_time = int(num_diffusion_steps * 0.1)
        betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
        
    elif schedule_type == 'warm0.2':
        betas = beta_end * np.ones(num_diffusion_steps, dtype=np.float64)
        warmup_time = int(num_diffusion_steps * 0.2)
        betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
        
    elif schedule_type == 'warm0.5':
        betas = beta_end * np.ones(num_diffusion_steps, dtype=np.float64)
        warmup_time = int(num_diffusion_steps * 0.5)
        betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    
    elif schedule_type == 'sigmoid':
        beta_start = 1000 / num_diffusion_steps * 0.0001 if beta_start is None else beta_start
        beta_end = 1000 / num_diffusion_steps * 0.02 if beta_end is None else beta_end
        betas = expit(np.linspace(-10, 10, num_diffusion_steps))*(beta_end-beta_start)+beta_start
        
    else:
        raise NotImplementedError(schedule_type)
    
    return betas


class GaussianDiffusion:
    def __init__(self, betas, loss_type, model_var_type, num_steps_in_parallel=100):
        self.loss_type = loss_type
        self.model_var_type = model_var_type
        assert isinstance(betas, np.ndarray)
        self.np_betas = betas = betas.astype(np.float64)  # computations here in float64 for accuracy
        assert (betas > 0).all() and (betas <= 1).all()
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.num_steps_in_parallel = num_steps_in_parallel

        alphas = 1. - betas
        alphas_cumprod = torch.from_numpy(np.cumprod(alphas, axis=0)).float()
        alphas_cumprod_prev = torch.from_numpy(np.append(1., alphas_cumprod[:-1])).float()

        # nth element of the list is beta_(n+1), e.g. element at index 0 is beta_1 which is used in the distribution q(x_1 | x_0) (equation 2 in the DDPM paper)
        self.betas = torch.from_numpy(betas).float()
        self.alphas_cumprod = alphas_cumprod.float()
        self.alphas_cumprod_prev = alphas_cumprod_prev.float()

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod).float()
        self.log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod).float()
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod).float()
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1).float()

        betas = torch.from_numpy(betas).float()
        alphas = torch.from_numpy(alphas).float()
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = posterior_variance
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]]))
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)

    @staticmethod
    def _extract(a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        bs, = t.shape
        assert x_shape[0] == bs
        out = torch.gather(a, 0, t)
        assert out.shape == torch.Size([bs])
        return torch.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))

    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start
        variance = self._extract(1. - self.alphas_cumprod.to(x_start.device), t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape)
        return mean, variance, log_variance
        
    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (t == 0 means diffused for 1 step). 
        Compute the mean and variance of q(x_t | x_0)
        """
        if noise is None:
            noise = torch.randn(x_start.shape, device=x_start.device)
        assert noise.shape == x_start.shape
        
        return (
                self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start +
                self._extract(self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0).
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                self._extract(self.posterior_mean_coef1.to(x_start.device), t, x_t.shape) * x_start +
                self._extract(self.posterior_mean_coef2.to(x_start.device), t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance.to(x_start.device), t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped.to(x_start.device), t, x_t.shape)
        
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] == x_start.shape[0])
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, denoise_fn, x_t, t, clip_denoised: bool, clip_min=None, clip_max=None):
        """
        Compute the mean and variance of the diffusion forward process p(x_{t-1} | x_t) (the mu and sigma in equation 1 in the DDPM paper).
        Returns the mu and sigma of equation 1 in the DDPM paper, the log of the sigma, and the reconstructed x0 (via the equation in the second-to-last
        text line on page 3).
        """
        
        B, C = x_t.shape[:2]
        assert t.shape == (B,)
        
        model_output = denoise_fn(x_t, t)
        
        if self.model_var_type == 'learned_range' or self.model_var_type == 'learned':
            assert model_output.shape == (B, C * 2, *x_t.shape[2:])
            
            model_output, model_var_values = torch.split(model_output, C, dim=1)
            
            if self.model_var_type == "learned":
                model_log_variance_x = model_var_values
                model_variance_x = torch.exp(model_log_variance_x)
            else:
                min_log = self._extract(self.posterior_log_variance_clipped.to(x_t.device), t, x_t.shape)
                max_log = self._extract(torch.log(self.betas).to(x_t.device), t, x_t.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance_x = frac * max_log + (1 - frac) * min_log
                model_variance_x = torch.exp(model_log_variance_x)
            
        else:
            model_variance, model_log_variance = {
                # I never checked out why improved-diffusion does what it does here, but I think it is because of quantization
                'fixedlarge': (self.betas.to(x_t.device),  # improved-diffusion uses: torch.cat([self.posterior_variance[1:2], self.betas[1:]])
                            torch.log(self.betas).to(x_t.device)),  # improved-diffusion uses: torch.cat([self.posterior_variance[1:2], self.betas[1:]])
                'fixedsmall': (self.posterior_variance.to(x_t.device), self.posterior_log_variance_clipped.to(x_t.device)),
            }[self.model_var_type]
        
            model_variance_x = self._extract(model_variance, t, x_t.shape) * torch.ones_like(x_t)
            model_log_variance_x = self._extract(model_log_variance, t, x_t.shape) * torch.ones_like(x_t)
        
        x_reconstructed = self._predict_xstart_from_eps(x_t, t=t, eps=model_output)
        if clip_denoised:
            x_reconstructed = torch.clamp(x_reconstructed, clip_min, clip_max)
        
        model_t_minus_1_mean, _, _ = self.q_posterior_mean_variance(x_start=x_reconstructed, x_t=x_t, t=t)

        return model_t_minus_1_mean, model_variance_x, model_log_variance_x, x_reconstructed

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t - self._extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape) * eps
        
    def _predict_eps_from_xstart(self, x_t, t, x_start):
        assert x_start.shape == x_t.shape
        return (self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t - x_start) / self._extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_start.shape)

    def _vb_terms_bpd(self, denoise_fn, x_start, x_t, t, clip_denoised: bool, clip_min=None, clip_max=None, no_t0=False):
        
        model_x_mean, model_x_variance, model_x_log_variance, x_recon = self.p_mean_variance(denoise_fn, x_t=x_t, t=t, clip_denoised=clip_denoised, clip_min=clip_min, clip_max=clip_max)
        true_x_mean, _, true_x_log_variance_clipped = self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)
        kl = normal_kl(true_x_mean, true_x_log_variance_clipped, model_x_mean, model_x_log_variance)
        kl = mean_flat(kl) / np.log(2.)  # mean over all non-batch dimensions and express in bits/dim
        decoder_nll = -gaussian_log_likelihood(x_start, means=model_x_mean, log_stddevs=0.5 * model_x_log_variance)
        # equivalent, but with clipping: decoder_nll = torch.nn.GaussianNLLLoss(reduction='none', full=True)(input=model_x_mean, target=x_start, var=model_x_variance)
        decoder_nll = mean_flat(decoder_nll) / np.log(2.)  # mean over all non-batch dimensions and express in bits/dim
        if no_t0:
            decoder_nll = torch.zeros_like(decoder_nll)
        kl = torch.where((t == 0), decoder_nll, kl)
        return kl, x_recon

    def p_losses(self, denoise_fn, x_start, t, noise=None, reduce=True):
        """
        Training loss calculation
        """
        N, dim = x_start.shape
        assert t.shape == torch.Size([N])
        
        if noise is None:
            noise = torch.randn(x_start.shape, dtype=x_start.dtype, device=x_start.device)
        assert noise.shape == x_start.shape and noise.dtype == x_start.dtype
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)

        if self.loss_type == 'mse':
            model_output = denoise_fn(x_t, t)
            
            if self.model_var_type == 'learned_range' or self.model_var_type == 'learned':
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = torch.split(model_output, C, dim=1)
                frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)
                kl, _ = self._vb_terms_bpd(
                    denoise_fn=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )
                # Divide by 1000 for equivalence with initial implementation.
                # Without a factor of 1/1000, the VB term hurts the MSE term.
                # Comment Philipp: I took this from improved-diffusion.
                kl  *= self.num_timesteps / 1000.0
            else:
                kl = 0.
            
            loss = ((noise - model_output)**2).mean(-1)
            loss = loss + kl
        
        elif self.loss_type == 'rescaled_kl' or self.loss_type == 'kl':
            kl, _ = self._vb_terms_bpd(
                denoise_fn=denoise_fn,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
            )
            
            if self.loss_type == 'rescaled_kl':
                kl *= self.num_timesteps
                
            loss = kl
        
        elif self.loss_type == 'rescaled_kl_no_t0' or self.loss_type == 'kl_no_t0':
            kl, _ = self._vb_terms_bpd(
                denoise_fn=denoise_fn,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                no_t0=True,
            )
            
            if self.loss_type == 'rescaled_kl_no_t0':
                kl *= self.num_timesteps
                
            loss = kl
            
        if reduce: 
            loss = loss.mean()
            
        return loss

    '''debug'''

    def _prior_bpd(self, x_start):

        with torch.no_grad():
            N, T = x_start.shape[0], self.num_timesteps
            t_ = torch.empty(N, dtype=torch.int64, device=x_start.device).fill_(T-1)
            qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t=t_)
            kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance,
                                 mean2=torch.tensor([0.]).to(qt_mean), logvar2=torch.tensor([0.]).to(qt_log_variance))
            assert kl_prior.shape == x_start.shape
            return kl_prior.mean(dim=list(range(1, len(kl_prior.shape)))) / np.log(2.)
        
    @torch.no_grad()
    def calc_ode_bdp(self, denoise_fn, x_start, clip_denoised=False, clip_min=None, clip_max=None, steps=None, eps=1e-5, rtol=1e-5, atol=1e-5, method='RK45'):
        from scipy import integrate
        shape = x_start.shape
        epsilon = torch.randint_like(x_start, low=0, high=2).float() * 2 - 1.
        
        def drift_fn(denoise_fn, sample, vec_t):
            beta_min = float(self.betas[0])*1000
            beta_max = float(self.betas[-1])*1000
            beta_t = beta_min + (beta_max - beta_min) * vec_t
            drift = -0.5 * beta_t[:,None] * sample
            diffusion = torch.sqrt(beta_t)
            t = vec_t * (self.num_timesteps-1)
            score = denoise_fn(sample, t)
            log_mean_coeff = -0.25 * vec_t ** 2 * (beta_max - beta_min) - 0.5 * vec_t * beta_min
            std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
            score = -score / std[:, None]
            drift = drift - diffusion[:, None] ** 2 * score * 0.5
            return drift
        
        def get_div_fn(fn):
            def div_fn(x, t, eps):
                with torch.enable_grad():
                    x.requires_grad_(True)
                    fn_eps = torch.sum(fn(x, t) * eps)
                    grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
                x.requires_grad_(False)
                return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))
            return div_fn
        
        def div_fn(denoise_fn, x, t, noise):
            fn = lambda xx, tt: drift_fn(denoise_fn, xx, tt)
            div_fn = get_div_fn(fn)
            div = div_fn(x, t, noise)
            return div
        
        def ode_func(t, x):
            sample = x[:-shape[0]].reshape(shape)
            sample = torch.from_numpy(sample).to(x_start.device).to(x_start.dtype)
            vec_t = torch.ones(sample.shape[0], device=x_start.device).to(x_start.dtype) * t
            drift = drift_fn(denoise_fn, sample, vec_t)
            logp_grad = div_fn(denoise_fn, sample, vec_t, epsilon)
            drift = drift.detach().cpu().numpy().reshape((-1,))
            logp_grad = logp_grad.detach().cpu().numpy().reshape((-1,))
            return np.concatenate([drift, logp_grad], axis=0)
        
        x_init = x_start.detach().cpu().numpy().reshape((-1,))
        logp_grad_init = np.zeros((shape[0], ))
        init = np.concatenate([x_init, logp_grad_init], axis=0)
        solution = integrate.solve_ivp(ode_func, (eps, 1), init, rtol=rtol, atol=atol, method=method)
        zp = solution.y[:, -1]
        z = zp[:-shape[0]]
        z = z.reshape(shape)
        z = torch.from_numpy(z).to(x_start.device).to(x_start.dtype)
        N = np.prod(shape[1:])
        prior_logp = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=1) / 2.
        
        delta_logp = zp[-shape[0]:]
        delta_logp = delta_logp.reshape((shape[0],))
        delta_logp = torch.from_numpy(delta_logp).to(x_start.device).to(x_start.dtype)
        bpd = -(prior_logp + delta_logp) / np.log(2.)
        dimensions = np.prod(shape[1:])
        bpd = bpd / dimensions
        total_bpd = bpd+7  # TODO: add prior_bpd (computed from z)
        return total_bpd, None, None, None, None

    @torch.no_grad()
    def calc_bpd_loop(self, denoise_fn, x_start, clip_denoised=False, clip_min=None, clip_max=None, steps=None):

        assert clip_denoised == False or (clip_min is not None and clip_max is not None)
        
        batch_size = x_start.shape[0]
        feat_dim = x_start.shape[1]
        device = x_start.device
        
        vb_terms = []
        if steps is None:
            steps = list(range(self.num_timesteps))[::-1]
            include_prior = True
        else:
            steps = sorted(steps, reverse=True)
            include_prior = self.num_timesteps in steps
            steps = [s for s in steps if s != self.num_timesteps]
            
        noises_for_all_steps = torch.randn((len(steps), *x_start.shape), device=device)
        noises_for_all_steps = noises_for_all_steps.reshape(len(steps)*batch_size, *x_start.shape[1:])
            
        num_steps = len(steps)
        num_steps_in_parallel = self.num_steps_in_parallel
        steps = [steps[i:i+num_steps_in_parallel] for i in range(0, len(steps), num_steps_in_parallel)]
        assert num_steps == sum([len(s) for s in steps])
        assert all([len(s) <= num_steps_in_parallel for s in steps])
        
        x_start_repeated = x_start.repeat(num_steps_in_parallel, 1)
        
        t_batch = []
        for ts in steps:
            for t in ts:
                t_batch += [t]*batch_size
        t_batch_all = torch.tensor(t_batch, device=device)
        
        num_steps_done = 0
        for cur_steps in steps:  # go through reverse trajectory in order (e.g. from x_T to x_T-1, ..., to x_0); t=999 means going from x_1000 to x_999
            noise = noises_for_all_steps[num_steps_done *batch_size : (num_steps_done + len(cur_steps)) * batch_size]
            t_batch = t_batch_all[num_steps_done * batch_size : (num_steps_done + len(cur_steps)) * batch_size]
            cur_x_start_repeated = x_start_repeated[:(len(cur_steps)) * batch_size]
            num_steps_done += len(cur_steps)
            
            x_t = self.q_sample(x_start=cur_x_start_repeated, t=t_batch, noise=noise)
                
            vb_terms_bpd_result = self._vb_terms_bpd(
                denoise_fn,
                x_start=cur_x_start_repeated, x_t=x_t,
                t=t_batch,
                clip_denoised=clip_denoised,
                clip_min=clip_min,
                clip_max=clip_max,
            )
            
            kl, _ = vb_terms_bpd_result
            kls = kl.split(batch_size)
            for kl in kls:
                vb_terms.append(kl)
        
        if include_prior:
            prior_bpd = self._prior_bpd(x_start)  # L1000
            vb_terms = [prior_bpd] + vb_terms
            
        vb_terms = torch.stack(vb_terms, dim=1)  # ordered like L1000, L999, ..., L1, L0
        vb = vb_terms.sum(dim=1)
            
        return vb, vb_terms  # shapes [N] and [N, T+1]

    @torch.no_grad()
    def calc_eps_mse_loop(self, denoise_fn, x_start, steps=None):
        batch_size = x_start.shape[0]
        feat_dim = x_start.shape[1]
        device = x_start.device
        dtype = x_start.dtype
        
        eps_mse_terms = []
        if steps is None:
            steps = list(range(self.num_timesteps))[::-1]
        else:
            steps = sorted(steps, reverse=True)
        assert self.num_timesteps not in steps  # eps_mse can not be computed for timestep T+1
            
        noises_for_all_steps = torch.randn((len(steps), *x_start.shape), device=device)
        noises_for_all_steps = noises_for_all_steps.reshape(len(steps)*batch_size, *x_start.shape[1:])
            
        num_steps = len(steps)
        num_steps_in_parallel = self.num_steps_in_parallel
        steps = [steps[i:i+num_steps_in_parallel] for i in range(0, len(steps), num_steps_in_parallel)]
        assert num_steps == sum([len(s) for s in steps])
        assert all([len(s) <= num_steps_in_parallel for s in steps])
        
        x_start_repeated = x_start.repeat(num_steps_in_parallel, 1)
        
        t_batch = []
        for ts in steps:
            for t in ts:
                t_batch += [t]*batch_size
        t_batch_all = torch.tensor(t_batch, device=device)
        
        num_steps_done = 0
        for cur_steps in steps:  # go through reverse trajectory in order (e.g. from x_T to x_T-1, ..., to x_0); t=999 means going from x_1000 to x_999
            noise = noises_for_all_steps[num_steps_done *batch_size : (num_steps_done + len(cur_steps)) * batch_size]
            t_batch = t_batch_all[num_steps_done * batch_size : (num_steps_done + len(cur_steps)) * batch_size]
            cur_x_start_repeated = x_start_repeated[:(len(cur_steps)) * batch_size]
            num_steps_done += len(cur_steps)
            
            x_t = self.q_sample(x_start=cur_x_start_repeated, t=t_batch, noise=noise)
                
            pred_eps = denoise_fn(x_t.to(dtype), t_batch)
            if self.model_var_type == 'learned_range' or self.model_var_type == 'learned':
                assert pred_eps.shape == (x_t.shape[0], feat_dim * 2)
                pred_eps, pred_var = torch.split(pred_eps, feat_dim, dim=1)
            eps_mses = ((noise - pred_eps)**2).mean(-1)
            
            eps_mses = eps_mses.split(batch_size)
            for eps_mse in eps_mses:
                eps_mse_terms.append(eps_mse)
            
        eps_mse_terms = torch.stack(eps_mse_terms, dim=1)  # ordered like eps_mse for x_1000 to x_999, ... x_1 to x_0
        eps_mse = eps_mse_terms.sum(dim=1)
            
        return eps_mse, eps_mse_terms  # shapes [N] and [N, T]


    @torch.no_grad()
    def calc_recon_mse_loop(self, denoise_fn, x_start, clip_denoised=False, clip_min=None, clip_max=None, steps=None):

        assert clip_denoised == False or (clip_min is not None and clip_max is not None)
        
        batch_size = x_start.shape[0]
        feat_dim = x_start.shape[1]
        device = x_start.device
        dtype = x_start.dtype
        
        recon_mse_terms = []
        if steps is None:
            steps = list(range(self.num_timesteps))[::-1]
        else:
            steps = sorted(steps, reverse=True)
        assert self.num_timesteps not in steps  # recon_mse can not be computed for timestep T+1
            
        noises_for_all_steps = torch.randn((len(steps), *x_start.shape), device=device)
        noises_for_all_steps = noises_for_all_steps.reshape(len(steps)*batch_size, *x_start.shape[1:])
            
        num_steps = len(steps)
        num_steps_in_parallel = self.num_steps_in_parallel
        steps = [steps[i:i+num_steps_in_parallel] for i in range(0, len(steps), num_steps_in_parallel)]
        assert num_steps == sum([len(s) for s in steps])
        assert all([len(s) <= num_steps_in_parallel for s in steps])
        
        x_start_repeated = x_start.repeat(num_steps_in_parallel, 1)
        
        t_batch = []
        for ts in steps:
            for t in ts:
                t_batch += [t]*batch_size
        t_batch_all = torch.tensor(t_batch, device=device)
        
        num_steps_done = 0
        for cur_steps in steps:  # go through reverse trajectory in order (e.g. from x_T to x_T-1, ..., to x_0); t=999 means going from x_1000 to x_999
            noise = noises_for_all_steps[num_steps_done *batch_size : (num_steps_done + len(cur_steps)) * batch_size]
            t_batch = t_batch_all[num_steps_done * batch_size : (num_steps_done + len(cur_steps)) * batch_size]
            cur_x_start_repeated = x_start_repeated[:(len(cur_steps)) * batch_size]
            num_steps_done += len(cur_steps)
            
            x_t = self.q_sample(x_start=cur_x_start_repeated, t=t_batch, noise=noise)
                
            pred_eps = denoise_fn(x_t.to(dtype), t_batch)
            if self.model_var_type == 'learned_range' or self.model_var_type == 'learned':
                assert pred_eps.shape == (x_t.shape[0], feat_dim * 2)
                pred_eps, pred_var = torch.split(pred_eps, feat_dim, dim=1)
            x_recon = self._predict_xstart_from_eps(x_t, t=t_batch, eps=pred_eps)
            if clip_denoised:
                x_recon = torch.clamp(x_recon, clip_min, clip_max)
            recon_mses = ((x_recon-cur_x_start_repeated)**2).mean(-1)
            
            recon_mses = recon_mses.split(batch_size)
            for recon_mse in recon_mses:
                recon_mse_terms.append(recon_mse)
            
        recon_mse_terms = torch.stack(recon_mse_terms, dim=1)  # ordered like recon_mse for x_1000 to x_999, ... x_1 to x_0
        recon_mse = recon_mse_terms.sum(dim=1)
            
        return recon_mse, recon_mse_terms  # shapes [N] and [N, T]

    def avg_mse_loop(self, denoise_fn, x_start, clip_denoised=False, clip_min=None, clip_max=None, steps=None):
        # assert clip_denoised == False or (clip_min is not None and clip_max is not None)
        assert not clip_denoised
        
        with torch.no_grad():
            batch_size, T = x_start.shape[0], self.num_timesteps
            device = x_start.device
            
            mses = []
            
            if steps is None:
                steps = list(range(self.num_timesteps))[::-1]
            else:
                steps = sorted(steps, reverse=True)
                
            for t in steps:  # go through reverse trajectory in order (e.g. from x_T to x_T-1, ..., to x_0); t=999 means going from x_1000 to x_999

                t_batch = torch.tensor([t] * batch_size, device=device)
                
                noise = torch.randn_like(x_start)
                x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
                
                pred_eps = denoise_fn(x_t, t_batch)
                x_recon = self._predict_xstart_from_eps(x_t, t=t_batch, eps=pred_eps)
                    
                xstart_mse = mean_flat((x_recon-x_start)**2)
                
                mses.append(xstart_mse)
                
            mses = torch.stack(mses, dim=1).mean(1)
                
            return mses

    @torch.no_grad()
    def calc_eps_cosine_loop(self, denoise_fn, x_start, steps=None):
        batch_size = x_start.shape[0]
        feat_dim = x_start.shape[1]
        device = x_start.device
        dtype = x_start.dtype
        
        eps_mse_terms = []
        if steps is None:
            steps = list(range(self.num_timesteps))[::-1]
        else:
            steps = sorted(steps, reverse=True)
        assert self.num_timesteps not in steps  # eps_mse can not be computed for timestep T+1
            
        noises_for_all_steps = torch.randn((len(steps), *x_start.shape), device=device)
        noises_for_all_steps = noises_for_all_steps.reshape(len(steps)*batch_size, *x_start.shape[1:])
            
        num_steps = len(steps)
        num_steps_in_parallel = self.num_steps_in_parallel
        steps = [steps[i:i+num_steps_in_parallel] for i in range(0, len(steps), num_steps_in_parallel)]
        assert num_steps == sum([len(s) for s in steps])
        assert all([len(s) <= num_steps_in_parallel for s in steps])
        
        x_start_repeated = x_start.repeat(num_steps_in_parallel, 1)
        
        t_batch = []
        for ts in steps:
            for t in ts:
                t_batch += [t]*batch_size
        t_batch_all = torch.tensor(t_batch, device=device)
        
        num_steps_done = 0
        for cur_steps in steps:  # go through reverse trajectory in order (e.g. from x_T to x_T-1, ..., to x_0); t=999 means going from x_1000 to x_999
            noise = noises_for_all_steps[num_steps_done *batch_size : (num_steps_done + len(cur_steps)) * batch_size]
            t_batch = t_batch_all[num_steps_done * batch_size : (num_steps_done + len(cur_steps)) * batch_size]
            cur_x_start_repeated = x_start_repeated[:(len(cur_steps)) * batch_size]
            num_steps_done += len(cur_steps)
            
            x_t = self.q_sample(x_start=cur_x_start_repeated, t=t_batch, noise=noise)
                
            pred_eps = denoise_fn(x_t.to(dtype), t_batch)
            if self.model_var_type == 'learned_range' or self.model_var_type == 'learned':
                assert pred_eps.shape == (x_t.shape[0], feat_dim * 2)
                pred_eps, pred_var = torch.split(pred_eps, feat_dim, dim=1)
            pred_eps = -pred_eps
            eps_mses = (noise * pred_eps).sum(-1, keepdim=True) / (noise.norm(dim=-1, keepdim=True) * pred_eps.norm(dim=-1, keepdim=True))
            eps_mses = (eps_mses).mean(-1)
            
            eps_mses = eps_mses.split(batch_size)
            for eps_mse in eps_mses:
                eps_mse_terms.append(eps_mse)
            
        eps_mse_terms = torch.stack(eps_mse_terms, dim=1)  # ordered like eps_mse for x_1000 to x_999, ... x_1 to x_0
        eps_mse = eps_mse_terms.sum(dim=1)
            
        return eps_mse, eps_mse_terms  # shapes [N] and [N, T]

    def mses_parallel(self, denoise_fn, x_start, clip_denoised=False, clip_min=None, clip_max=None, steps=None):
        
        dtype = x_start.dtype

        with torch.no_grad():
            batch_size, T = x_start.shape[0], self.num_timesteps
            device = x_start.device
            
            mses = []
            
            if steps is None:
                steps = list(range(self.num_timesteps))[::-1]
            # else:
            #     steps = sorted(steps, reverse=True)
            
            x_ts = []
            ts = []
            noises = []
            noises = torch.randn(len(steps), *x_start.shape, device=x_start.device)
            for t_i, t in enumerate(steps):

                t_batch = torch.tensor([t] * batch_size, device=device)
                
                # noise = torch.randn_like(x_start)
                # x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
                x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noises[t_i])
                
                x_ts.append(x_t)
                ts.append(t_batch)
                # noises.append(noise)
            # noises = torch.cat(noises, 0)
            B, C = x_ts[0].shape

            x_starts = x_start.unsqueeze(0).expand(len(steps), -1, -1).reshape(len(steps)*B, C)
            x_ts = torch.cat(x_ts, 0)
            ts = torch.cat(ts, 0)

            pred_eps = denoise_fn(x_ts.to(dtype=dtype), ts)
            x_recons = self._predict_xstart_from_eps(x_ts, t=ts, eps=pred_eps)

            mses = ((x_recons-x_starts)**2).view(len(steps), B, C)

            return mses
