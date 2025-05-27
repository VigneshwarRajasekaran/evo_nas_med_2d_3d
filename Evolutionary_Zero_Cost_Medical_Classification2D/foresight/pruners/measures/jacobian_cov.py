import numpy as np
import torch
import torch.nn.functional as F

import copy

from . import measure
from ..p_utils import get_layer_metric_array



@measure('jacobian_cov', bn=True)
def jacob_perclass_corr_score(net, medmnist_dataset, inputs, targets, loss_fn=None, split_data=1):
    def get_batch_jacobian(net, x):
        net.zero_grad()
        x.requires_grad_(True)
        _, y = net(x)
        y.backward(torch.ones_like(y))
        jacob = x.grad.detach()
        x.requires_grad_(False)
        return jacob

    def eval_score_perclass(jacob, labels=None, n_classes=10):
        k = 1e-5
        per_class = {}
        for i, label in enumerate(labels):
            if label in per_class:
                per_class[label] = np.vstack((per_class[label], jacob[i]))
            else:
                per_class[label] = jacob[i]
        ind_corr_matrix_score = {}
        for c in per_class.keys():
            try:
                corrs = np.corrcoef(per_class[c])
                s = np.sum(np.log(np.abs(corrs) + k))
                ind_corr_matrix_score[c] = s
            except:
                continue
        score = sum(np.abs(v) for v in ind_corr_matrix_score.values())
        return score

    jacob = get_batch_jacobian(net, inputs)
    jacob = jacob.reshape(jacob.size(0), -1).cpu().numpy()
    labels = targets.cpu().tolist()
    # flatten any 1-element lists into ints
    labels = [l[0] if isinstance(l, list) and len(l) == 1 else l for l in labels]
    return eval_score_perclass(jacob, labels)
