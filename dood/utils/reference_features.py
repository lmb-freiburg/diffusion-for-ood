import os
from tqdm import tqdm
import random
import torch

ON_CLUSTER = "PBS_JOBID" in os.environ and os.environ["PBS_JOBNAME"] != "STDIN"


def gt_reduce_zero_label(label):
    # that's how mmseg does it...
    label[label == 0] = 255
    label = label - 1
    label[label == 254] = 255
    return label


def fwd_pass_mmseg(data_sample, model):
    data_sample['inputs'] = [data_sample['inputs'].cuda()]
    data_sample['data_samples'] = [data_sample['data_samples']]
    model_out = model.test_step(data_sample)
    assert len(model_out) == 1, f"Expected single output, got {len(model_out)}"
    logits = model_out[0].seg_logits.data
    return logits


def get_reference_features(model, dataset, feature_probe, num_samples, ood_scoring_fn_for_scaling, reduce_zero_label=False, seed=None):
    ft_list = []
    avg_valids = []
    scores = []
    coordinates = []
    labels_list = []

    if seed is not None:
        random.seed(seed)    
    indices = random.sample(range(len(dataset)), num_samples)

    model.eval()
    for smpl_i, index in tqdm(enumerate(indices), disable=ON_CLUSTER):
        sample = dataset[index]
        smpl_ft_list = []
        smpl_labels = []

        segm = sample['data_samples'].gt_sem_seg.data.squeeze(0)
        logits = fwd_pass_mmseg(sample, model)
        scores.append(ood_scoring_fn_for_scaling(logits).flatten())
        proc_act_dict = feature_probe.get_features()
        features = [proc_act_dict[k] for k in proc_act_dict]
        
        if reduce_zero_label:
            segm = gt_reduce_zero_label(segm)
        for ft in features:
            h, w, c = ft.shape
            downsampled_segm = torch.nn.functional.interpolate(segm.view((1, 1, *segm.shape)).byte(), size=(h, w))
            valid = (downsampled_segm.squeeze() != 255).flatten()
            smpl_ft_list.append(ft.view(h * w, c).cpu()[valid])
            smpl_labels.append(downsampled_segm.flatten()[valid])
        ft_list.append(smpl_ft_list)
        avg_valids.append(valid.float().mean())
        labels_list.append(smpl_labels)

    scores = torch.cat(scores, 0).flatten()

    ft_lists = zip(*ft_list)  # from list of feature tuples, to lists of features (n_ft, n_imgs, samples_per_img)
    lb_lists = zip(*labels_list)  # from list of label tuples, to lists of labels (n_ft, n_imgs, samples_per_img)
    ft_dict = {k: torch.cat(fl, dim=0) for k, fl in zip(proc_act_dict.keys(), ft_lists)}  # concatenate for each ft type
    lb_dict = {k: torch.cat(ll, dim=0) for k, ll in zip(proc_act_dict.keys(), lb_lists)}  # concatenate for each ft type

    return ft_dict, lb_dict, (scores.mean().item(), scores.std().item())




