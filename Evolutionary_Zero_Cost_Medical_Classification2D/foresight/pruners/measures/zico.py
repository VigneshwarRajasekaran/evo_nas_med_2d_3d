import numpy as np
import torch
import torch.nn.functional as F

import copy

from . import measure
from ..p_utils import get_layer_metric_array




@measure('zico', bn=True)
def compute_zico_score(net, medmnist_dataset, inputs, targets, loss_fn=F.cross_entropy, split_data=1):
    grad_dict = {}
    net.train()

    for sp in range(split_data):
        st = sp * inputs.shape[0] // split_data
        en = (sp + 1) * inputs.shape[0] // split_data

        net.zero_grad()
        logits, _ = net(inputs[st:en])
        targets = torch.squeeze(targets, 1).long()
        loss = loss_fn(logits, targets[st:en])
        loss.backward()

        for name, mod in net.named_modules():
            if isinstance(mod, torch.nn.Conv2d) or isinstance(mod, torch.nn.Linear):
                g = mod.weight.grad
                if g is not None:
                    g = g.data.cpu().reshape(-1).numpy()
                    grad_dict.setdefault(name, []).append(g)

    for k in grad_dict:
        grad_dict[k] = np.array(grad_dict[k])

    nsr_mean_sum_abs = 0
    for g in grad_dict.values():
        std = np.std(g, axis=0)
        mean = np.mean(np.abs(g), axis=0)
        nonzero = np.nonzero(std)[0]
        ratio = mean[nonzero] / std[nonzero]
        if np.sum(ratio) > 0:
            nsr_mean_sum_abs += np.log(np.sum(ratio))
    return nsr_mean_sum_abs
