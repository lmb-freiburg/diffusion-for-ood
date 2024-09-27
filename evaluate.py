import os
import argparse
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt

import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmseg.apis import init_model
from mmengine.logging.logger import MMLogger

from dood.models import *
from dood.datasets import *
from dood.utils.reference_features import get_reference_features, fwd_pass_mmseg
from dood.probes import get_feature_probe
from dood.utils.evaluation import ood_score_functions, upsample_scores, combine_scores_standardize, StreamingEval
from dood.utils.diffusion import get_diffusion_model, load_diffusion_checkpoint, get_diffusion_scores, get_diffusion_stats, get_denoising_steps_from_string

logger = MMLogger.get_instance('mmengine', log_level='INFO')


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a DOoD model for out-of-distribution detection.')
    
    # setup
    parser.add_argument('--model_config', required=True, type=str, help='Path to the config file')
    parser.add_argument('--test_data_config', required=True, type=str, help='Path to the test data config file')
    parser.add_argument('--segmentation_ckpt', required=True, type=str, help='Path to the segmentation model checkpoint file')
    parser.add_argument('--diffusion_ckpt', required=True, type=str, help='Path to the diffusion model checkpoint file')
    
    # reference data config: if not provided, it will looked for in the model config
    parser.add_argument('--reference_data_config', required=False, type=str, help='Path to the reference data config file')

    # evaluation
    parser.add_argument('--num_reference_samples', type=int, default=25, help='Number of reference samples for normalization statistics.')
    parser.add_argument('--parametric_score_fn', default="logsumexp")
    parser.add_argument('--diffusion_score_fn', default="eps_cos", choices=["eps_mse", "eps_cos", "recon_mse"])
    parser.add_argument('--num_diffusion_steps', default=25, type=int)

    # save
    parser.add_argument('--save_dir', default=None, type=str, help='Directory to save scoremaps')

    args = parser.parse_args()
    if args.save_dir is not None: 
        os.makedirs(args.save_dir, exist_ok=True)
        logger.info(f"Saving scoremaps to {args.save_dir}")
    return args


@torch.no_grad()
def get_reference_stats(model, diffusion_model, reference_dataset, feature_probe,
                        num_reference_samples, 
                        parametric_score_fn, 
                        diffusion_score_fn,
                        diffusion_steps):
    reference_features, _, (param_mean_score, param_std_score) = get_reference_features(
        model, reference_dataset, feature_probe,
        num_samples=num_reference_samples,
        ood_scoring_fn_for_scaling=ood_score_functions[parametric_score_fn],
        reduce_zero_label=reference_dataset.reduce_zero_label,
        seed=0)
    mean_diff_score, std_diff_score = get_diffusion_stats(
        reference_features,
        diffusion_model,
        diffusion_steps=diffusion_steps,
        ood_eval_scores_type=diffusion_score_fn,
        ft_key=list(reference_features.keys())[0])
    
    return {"reference_features": reference_features,
            "param_mean_std_scores": (param_mean_score, param_std_score),
            "diff_mean_std_scores": (mean_diff_score, std_diff_score)}


@torch.no_grad()
def eval_ood(model, diffusion_model, test_dataset, feature_probe,
             parametric_score_fn,
             diffusion_score_fn,
             diffusion_steps,
             param_scores_mean_std=None, 
             diff_scores_mean_std=None,
             save_dir=None, save_scoremaps=False,
             dtype=torch.float16):
    
    model.eval()
    diffusion_model.eval()
    diffusion_model.denoiser.to(dtype=dtype)

    param_evaluator = StreamingEval(ood_id=test_dataset.ood_index, ignore_ids=255)
    diff_evaluator = StreamingEval(ood_id=test_dataset.ood_index, ignore_ids=255)
    evaluator = StreamingEval(ood_id=test_dataset.ood_index, ignore_ids=255)

    progress = tqdm(total=len(test_dataset))
    for sample_index, sample in enumerate(test_dataset):
        if test_dataset.reduce_zero_label:
            raise NotImplementedError
        segm = sample['data_samples'].gt_sem_seg.data.squeeze(0)
        logits = fwd_pass_mmseg(sample, model)[None,:]
        param_scores = upsample_scores(ood_score_functions[parametric_score_fn](logits)[0], segm)

        for k, test_ft in feature_probe.get_features().items():
            h, w, c = test_ft.shape
            
            # normalize before (so we can set a custom dtype), get scores
            test_ft = diffusion_model.normalize(test_ft.view(h*w, c))
            scores, scores_per_step = get_diffusion_scores(test_ft.view(h*w, c), diffusion_model, diffusion_steps, diffusion_score_fn, normalize=False, dtype=dtype)
            diff_scores = upsample_scores(scores.view(h, w), segm)
        
        param_evaluator.add(param_scores.cpu(), segm)
        diff_evaluator.add(diff_scores.cpu(), segm)
        comb_scores = combine_scores_standardize(param_scores, diff_scores, *param_scores_mean_std, *diff_scores_mean_std)

        if args.save_dir is not None:
            plt.imsave(os.path.join(args.save_dir, f"dood_scores_{sample_index:03d}.png"), comb_scores.squeeze().cpu().numpy(), cmap="turbo")

        evaluator.add(comb_scores.cpu(), segm)
        progress.update()
    progress.close()

    logger.info(f"Computing metrics...")
    # logger.info("Parametric - AP: {}, FPR: {}".format(*param_evaluator.get_results()[1:]))
    # logger.info("Diffusion - AP: {}, FPR: {}".format(*diff_evaluator.get_results()[1:]))

    _, combined_ap, combined_fpr = evaluator.get_results()
    logger.info(f"Results:")
    logger.info(f"AP: {combined_ap:.4f}, FPR: {combined_fpr:.4f}")


def main(args):
    # segmentation model
    model_config = Config.fromfile(args.model_config)
    model = init_model(model_config, device='cuda:0').eval()
    model.load_state_dict(torch.load(args.segmentation_ckpt)['state_dict'])
    feature_probe = get_feature_probe(model)

    # diffusion model
    diff_setup_hook_cfg = [h for h in model_config.custom_hooks if h.type=="DiffusionSetupHook"][0]
    diffusion_model = get_diffusion_model(**diff_setup_hook_cfg).eval().cuda()
    load_diffusion_checkpoint(diffusion_model, args.diffusion_ckpt)
    diffusion_steps = get_denoising_steps_from_string(f"last{args.num_diffusion_steps}", args.diffusion_score_fn, total_steps=diffusion_model.diffusion_process.num_timesteps)

    # data
    ref_data_cfg = Config.fromfile(ref_data_cfg) if args.reference_data_config is not None else model_config
    reference_loader = Runner.build_dataloader(ref_data_cfg.reference_dataloader)
    test_loader = Runner.build_dataloader(Config.fromfile(args.test_data_config).test_dataloader)

    # reference stats
    logger.info(f"Getting reference statistics from {args.num_reference_samples} samples...")
    reference_stats = get_reference_stats(model, diffusion_model, reference_loader.dataset, feature_probe, 
                                          args.num_reference_samples,
                                          args.parametric_score_fn,
                                          args.diffusion_score_fn,
                                          diffusion_steps)
    
    logger.info("Reference stats:")
    logger.info(f"Parametric: mean = {reference_stats['param_mean_std_scores'][0]:.03f}, std = {reference_stats['param_mean_std_scores'][1]:.03f}")
    logger.info(f"Diffusion:  mean = {reference_stats['diff_mean_std_scores'][0]:.03f}, std = {reference_stats['diff_mean_std_scores'][1]:.03f}")

    # evaluation
    logger.info(f"OoD evaluation with {args.num_diffusion_steps} diffusion steps...")
    eval_ood(model, diffusion_model, test_loader.dataset, feature_probe,
            args.parametric_score_fn,
            args.diffusion_score_fn,
            diffusion_steps,
            reference_stats["param_mean_std_scores"],
            reference_stats["diff_mean_std_scores"],
            save_dir=None, save_scoremaps=False,
            dtype=torch.float16)


if __name__ == '__main__':
    args = parse_args()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        main(args)

