import torch
from ..diffusion.diffusion_processes.gaussian_diffusion import GaussianDiffusion, get_beta_schedule
from ..diffusion.denoisers.unet_0d import UNet0D
from ..diffusion.diffusion_model import DiffusionModel


def get_denoising_steps_from_string(steps_string, score_fn, total_steps=1000):
    if not steps_string:
        val_diffusion_steps = None
    elif steps_string.startswith("even"):
        num_steps = int(steps_string.replace("even", ""))
        val_diffusion_steps = list(range(0, total_steps, total_steps//num_steps))
    elif "every" in steps_string and steps_string.startswith("last"):
        raise NotImplementedError
    elif steps_string.startswith("last"):
        num_steps = int(steps_string.replace("last", ""))
        val_diffusion_steps = list(range(num_steps))
    elif steps_string.startswith("repeat"):
        step_n, n_repeats = [int(n) for n in steps_string.replace("repeat", "").split("x")]
        val_diffusion_steps = [step_n]*n_repeats
    elif steps_string.startswith("full"):
        if "x" in steps_string:
            n_repeats = int(steps_string.replace("full", "").split("x")[1])
        else:
            n_repeats = 1
        val_diffusion_steps = (list(range(total_steps+1))[::-1])*n_repeats
    else:
        raise ValueError
    
    if score_fn in ["eps_mse", "recon_mse"]:
        val_diffusion_steps = [x for x in val_diffusion_steps if x!=total_steps]
    val_diffusion_steps = sorted(val_diffusion_steps, reverse=True)

    return val_diffusion_steps


def get_diffusion_model(
        *,
        ft_size, 
        denoiser_type="unet0d", 
        diffusion_denoiser_channels=256, 
        diffusion_model_channel_mult=(1, 1, 1, 1, 1, 1),
        diffusion_model_num_middle_blocks=3,
        diffusion_model_time_embed_dim_factor=1,
        diffusion_model_time_embed_mode="periodic+embed",
        diffusion_model_dropout=0.0,
        beta_schedule_type="linear",
        num_diffusion_steps=1000, 
        beta_start=0.0001, beta_end=0.02, 
        loss_type='mse',
        model_var_type='fixedsmall',
        batch_size=None,
        normalization_queue_size=4096,
        scale_to_unit_gauss=False,
        scale_per_axis=False,
        use_max_std=False, 
        half_scale=False, 
        scale_scaling_factor=None,
        uniform_timestep_sampler=True,
        num_steps_in_parallel=100,  
        only_first_betas=None,
        **kwargs,
        ):
    
    betas = get_beta_schedule(schedule_type=beta_schedule_type, num_diffusion_steps=num_diffusion_steps, beta_start=beta_start, beta_end=beta_end)
    if only_first_betas:
        assert isinstance(only_first_betas, int)
        assert only_first_betas > 0
        betas = betas[:only_first_betas]

    out_channels = 2 * ft_size if model_var_type == 'learned' or model_var_type == 'learned_range' else ft_size

    if denoiser_type.lower() == "unet0d":
        denoiser = UNet0D(in_channels=ft_size, 
                          model_channels=diffusion_denoiser_channels, 
                          out_channels=out_channels, 
                          channel_mult=diffusion_model_channel_mult, 
                          num_middle_blocks=diffusion_model_num_middle_blocks, 
                          time_embed_dim_factor=diffusion_model_time_embed_dim_factor, 
                          timestep_embedding_mode=diffusion_model_time_embed_mode, 
                          dropout=diffusion_model_dropout)
    else:
        raise ValueError

    diffusion_process = GaussianDiffusion(betas, loss_type, model_var_type, num_steps_in_parallel=num_steps_in_parallel)
    # diffusion_model = build_dp(DiffusionModel(diffusion_process, denoiser, train_min_max=None), "cuda", device_ids=[0])
    diffusion_model = DiffusionModel(diffusion_process, denoiser, train_min_max=None, normalization_queue_size=normalization_queue_size, 
                                     uniform_timestep_sampler=uniform_timestep_sampler, scale_per_axis=scale_per_axis,
                                     scale_to_unit_gauss=scale_to_unit_gauss, use_max_std=use_max_std, half_scale=half_scale, scale_scaling_factor=scale_scaling_factor)
    diffusion_model = diffusion_model.cuda()
    diffusion_model.batch_size = batch_size

    return diffusion_model


def load_diffusion_checkpoint(diffusion_model, checkpoint_file):
    incompatible = diffusion_model.load_state_dict(torch.load(checkpoint_file)['state_dict'], strict=False)
    assert len(incompatible.unexpected_keys)==0 and all(k in ['normalization.min', 'normalization.max', 'normalization.shift', 'normalization.scale', 'normalization.means', 'normalization.mean', 'normalization.stds', 'normalization.std'] for k in incompatible.missing_keys), str(incompatible)
    if len(incompatible.missing_keys)>0 and 'normalization.min' in incompatible.missing_keys:
        diffusion_model.normalization._update_min_max_mean_std_shift_scale()


def get_diffusion_scores(features, diffusion_model, diffusion_steps, ood_eval_scores_type, normalize=True, dtype=torch.float32):
    if normalize:
        features = diffusion_model.normalize(features)

    features = features.to(dtype)

    if ood_eval_scores_type == "bpd":
        scores, scores_per_step = diffusion_model.compute_vb(x_start=features, clip_denoised=True, steps=diffusion_steps)
    elif ood_eval_scores_type == "eps_mse":
        scores, scores_per_step = diffusion_model.compute_eps_mse(x_start=features, steps=diffusion_steps)
        scores_per_step = None
    elif ood_eval_scores_type == "eps_cos":
        scores, scores_per_step = diffusion_model.compute_eps_cos(x_start=features, steps=diffusion_steps)
        scores_per_step = None
    elif ood_eval_scores_type == "recon_mse":
        scores, scores_per_step = diffusion_model.compute_recon_mse(x_start=features, clip_denoised=True, steps=diffusion_steps)
        scores_per_step = None
    else:
        raise ValueError
    
    return scores, scores_per_step


def get_diffusion_stats(features, diffusion_model, diffusion_steps, ood_eval_scores_type, ft_key, dtype=torch.float32):
    all_scores = []
    for batch in torch.split(features[ft_key], 1024, 0):
        scores, _ = get_diffusion_scores(batch.cuda(), diffusion_model, diffusion_steps, ood_eval_scores_type, normalize=True, dtype=dtype)
        scores = scores.cpu()
        all_scores.append(scores)
    all_scores = torch.cat(all_scores)
    return all_scores.mean().item(), all_scores.std().item()
