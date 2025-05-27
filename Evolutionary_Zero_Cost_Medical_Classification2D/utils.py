import math
import os
import random

import numpy as np
import torch
import shutil
from torch.autograd import Variable
import augment
import augmentations
from dataset import Dataset
import genotype
import operations
from operations_mapping import operations_mapping,attentions

import os
import numpy as np
import torch
import shutil
from torch.autograd import Variable
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init


class AverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    """

    :param output: logits, [b, classes]
    :param target: [b]
    :param topk:
    :return:
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)



term_width=5
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def create_param_choices(primitives, nas_setup):
    nn_param_choices = {}
    for i in range(len(nas_setup)):
        if i % 2 == 0:
            nn_param_choices[str(i)] = primitives
        else:
            end_index = int(nas_setup[i])
            nn_param_choices[str(i)] = np.arange(end_index).tolist()
    return nn_param_choices


def accuracy(output, target, topk=(1,)):
    """

    :param output: logits, [b, classes]
    :param target: [b]
    :param topk:
    :return:
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res

# def accuracy(output, target, topk=(1,)):
#     maxk = max(topk)
#     batch_size = target.size(0)
#
#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))
#
#     res = []
#     for k in topk:
#         correct_k = correct[:k].view(-1).float().sum(0)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path_fp16(x, drop_prob):
    x = x.half()
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)).half()
        x.div_(keep_prob)
        x.mul_(mask)
    return x


import torch

def drop_path(x, drop_prob: float, training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    If not in training mode or drop_prob == 0, returns x unchanged.
    """
    if drop_prob == 0. or not training:
        return x

    keep_prob = 1.0 - drop_prob
    # shape = [batch_size, 1, 1, ..., 1]
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)

    # generate binary mask
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    binary_mask = torch.floor(random_tensor)

    # scale the surviving paths, apply mask
    return x.div(keep_prob) * binary_mask


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def find_nearest(pop, value):
    n = [abs(i.params - value) for i in pop if i.front == True]
    idx = n.index(min(n))
    return pop[idx]


def save_seed(val, filename):
    """ saves val. Called once in simulation1.py """
    with open(filename, "w") as f:
        f.write(str(val))


def load_seed(filename):
    """ loads val. Called by all scripts that need the shared seed value """
    with open(filename, "r") as f:
        # change datatype accordingly (numpy.random.random() returns a float)
        return int(f.read())


def get_classes(args):
    if args.dataset == 'cifar10':
        args.classes = 10
    elif args.dataset == 'cifar100':
        args.classes = 100
    else:
        args.classes = 1000
    return args

class Cutout:
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def progressive_layer(args):
    result = []
    p = 4
    for i in range(1, args.generations + 1):
        if (i % 6) == 0:
            result.append(p)
            p += args.p_layers
        else:
            result.append(p)
    return result


def decode_cell(chromosome):
    normal_cell = []
    reduce_cell = []
    size = int(len(chromosome) / 2)
    count = 0
    for key, val in chromosome.items():
        if count < size:
            normal_cell.append(val)
        else:
            reduce_cell.append(val)
        count += 1

    normal, normal_concat = [], list(range(2, int(len(normal_cell) / 6) + 2))
    reduce, reduce_concat = [], list(range(2, int(len(reduce_cell) / 6) + 2))

    for i in range(0,len(normal_cell),3):
        # if i % 2 == 1:
        normal.append((normal_cell[i], normal_cell[i+1], normal_cell[i+2]))
            # if isinstance(normal_cell[i], int) and normal_cell[i] in normal_concat:
            #  normal_concat.remove(normal_cell[i])

    for i in range(0,len(reduce_cell),3):
        # if i % 2 == 1:
        reduce.append((reduce_cell[i], reduce_cell[i + 1], reduce_cell[i+2]))
            # if isinstance(reduce_cell[i], int) and reduce_cell[i] in reduce_concat:
            #  reduce_concat.remove(reduce_cell[i])

    return genotype.Genotype(normal=normal,
                             normal_concat=normal_concat,
                             reduce=reduce,
                             reduce_concat=reduce_concat,normal_bottleneck='',reduce_bottleneck='')
def model_to_syncbn(model):
    preserve_state_dict = model.state_dict()
    _convert_module_from_bn_to_syncbn(model)
    model.load_state_dict(preserve_state_dict)
    return model
def _convert_module_from_bn_to_syncbn(module):
    for child_name, child in module.named_children():
        if hasattr(nn, child.__class__.__name__) and \
            'batchnorm' in child.__class__.__name__.lower():
            TargetClass = globals()['Synchronized'+child.__class__.__name__]
            arguments = TargetClass.__init__.__code__.co_varnames[1:]
            kwargs = {k: getattr(child, k) for k in arguments}
            setattr(module, child_name, TargetClass(**kwargs))
        else:
            _convert_module_from_bn_to_syncbn(child)
def decode_operations(pop, indexes):
    network = {}
    for i in range(0,len(pop),3):
        network[str(i)] = operations_mapping.get(math.floor((pop[i]) * len(operations_mapping)))
        network[str(i+1)] = int(pop[i+1])
        network[str(i+2)] = attentions.get(pop[i+2])
    # for i, indv in enumerate(pop):
    #     # print(OPS.get(thisdict.get(indv)))
    #     if i % 2 == 0:
    #         network[str(i)] = operations_mapping.get(math.floor((indv) * len(operations_mapping)))
    #         # print(operations_mapping.get(indv))
    #     else:
    #         #   #print(int(random.choice(indexes)))
    #         network[str(i)] = indv
    # print(network)
    return network


# function which returns the index of minimum value in the list

def get_minvalue(inputlist):
    # get the minimum value in the list
    min_value = min(inputlist)

    # return the index of minimum value

    min_index = []

    for i in range(0, len(inputlist)):

        if min_value == inputlist[i]:
            min_index.append(i)

    return min_index



class Transform3D:

    def __init__(self, mul=None):
        self.mul = mul

    def __call__(self, voxel):

        if self.mul == '0.5':
            voxel = voxel * 0.5
        elif self.mul == 'random':
            voxel = voxel * np.random.uniform()

        return voxel.astype(np.float32)
