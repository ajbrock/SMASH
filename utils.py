#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' Utilities file
Andy's Notes: Need to properly credit things based on where we got them.

'''

from __future__ import print_function
import sys
import time
import json
import logging
import path
import math
import numpy as np

from PIL import Image
from functools import reduce
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def train_parser():
    usage = 'Trains and tests SMASH on CIFAR.'
    parser = ArgumentParser(description=usage)
    parser.add_argument(
        '-D', '--depth', type=int, default=12,
        help='Reference Network depth in layers/block (default: %(default)s)')
    parser.add_argument(
        '-k', '--width', type=int, default=4,
        help='Reference network widening factor (default: %(default)s)')
    parser.add_argument(
        '-N', type=int, default=8,
        help='Base memory bank size (default: %(default)s)')
    parser.add_argument(
        '-N-max', type=int, default=64,
        help='Maximum memory bank size (default: %(default)s)')
    parser.add_argument(
        '--which-dataset', type=str, default='C100',
        help='Which Dataset to train on, out of C10, C100, MN10, MN40, STL10 (default: %(default)s)')
    parser.add_argument(
        '--bottleneck', type=int, default=4,
        help='Bottleneck (default: %(default)s)')
    parser.add_argument(
        '--max-bottleneck', type=int, default=2,
        help='Max Bottleneck (default: %(default)s)')
    parser.add_argument(
        '--depth-compression', type=int, default=2,
        help='Depth compression (default: %(default)s)')
    parser.add_argument(
        '--max-dilate', type=int, default=3,
        help='Maximum allowable dilation factor (default: %(default)s)')
    parser.add_argument(
        '--max-kernel', type=int, default=7,
        help='Maximum allowable kernel size (default: %(default)s)')
    parser.add_argument(
        '--max-groups', type=int, default=8,
        help='Maximum number of groups (default: %(default)s)')
    parser.add_argument(
        '--var-op', action='store_true', default=False,
        help='Use variable op structure (default: %(default)s)')
    parser.add_argument(
        '--big-op', action='store_true', default=False,
        help='Use full op structure (default: %(default)s)')
    parser.add_argument(
        '--long-op', action='store_true', default=False,
        help='Use two-stacked-conv op structure (default: %(default)s)')
    parser.add_argument(
        '--gates', action='store_true', default=False,
        help='Allow for LSTM-style multiplicative gates CURRENTLY BROKEN (default: %(default)s)')
    parser.add_argument(
        '--op-bn', action='store_true', default=False,
        help='Use batchnorm in the fixed conv elements (default: %(default)s)')
    parser.add_argument(
        '--postac', action='store_false', dest='preactivation', default=True,
        help='Use post activation instead of preactivation (default: %(default)s)') 
    parser.add_argument(
        '--var-nl', action='store_true', default=False,
        help='Allow for different nonlinearities. (default: %(default)s)')
    parser.add_argument(
        '--static-ks', action='store_false', dest='var_ks', default=True,
        help='Disallow variable kernel sizes (default: %(default)s)')
    parser.add_argument(
        '--static-group', action='store_false', dest='var_group', default=True,
        help='Disallow variable groups (default: %(default)s)')   
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Random seed to use.')
    parser.add_argument(
        '--augment', action='store_true', default=True,
        help='Perform data augmentation (enabled by default)')
    parser.add_argument(
        '--no-augment', action='store_false', dest='augment',
        help='Disable data augmentation')
    parser.add_argument(
        '--validate', action='store_true', default=True,
        help='Perform validation on validation set (ensabled by default)')
    parser.add_argument(
        '--no-validate', action='store_false', dest='validate',
        help='Disable validation')
    parser.add_argument(
        '--validate-test', action='store_const', dest='validate',
        const='test', help='Evaluate on test set after every epoch.')
    parser.add_argument(
        '--epochs', type=int, default=100,
        help='Number of training epochs (default: %(default)s)')
    parser.add_argument(
        '--save-weights', type=str, default='default_save', metavar='FILE',
        help=('Save network weights to given .pth file'+
              '(default: automatically name based on input args'))    
    parser.add_argument(
        '--batch-size', type=int, default=50,
        help='Images per batch (default: %(default)s)')
    parser.add_argument(
        '--resume', type=bool, default=False,
        help='Whether or not to resume training')
    parser.add_argument(
        '--model', type=str, default='SMASHv8', metavar='FILE',
        help='Which model to use')
    parser.add_argument(
        '--SMASH', type=str, default=None,
        help='If False, train a SMASH network;\
              if given, train a network derived from the SMASH network')
    parser.add_argument(
        '--rank', type=int, default=0,
        help='If training a derived network, which rank (of all eval''d nets)\
               to use (default: %(default)s')
    parser.add_argument(
        '--init-from-SMASH', action='store_true', default=False,
        help='Initialize using SMASH params (default: %(default)s')
    parser.add_argument(
        '--fp16', action='store_true', default=False,
        help='Train with half-precision (default: %(default)s)')
    parser.add_argument(
        '--parallel', action='store_true', default=False,
        help='Train with multiple GPUs (default: %(default)s)')
    parser.add_argument(
        '--validate-every', type=int, default=1,
        help='Test after every this many epochs (default: %(default)s)')
    parser.add_argument(
        '--duplicate-at-checkpoints', action='store_true', default=False,
        help='Save an extra copy every 5 checkpoints (default: %(default)s)')
    parser.add_argument(
        '--fold', type=int, default=10,
        help='Which STL-10 training fold to use, 10 uses all (default: %(default)s)')
    parser.add_argument(
        '--top5', action='store_true', default=False,
        help='Measure top-5 error on valid/test instead of top-1 (default: %(default)s')
    return parser

def eval_parser():
    usage = 'Samples SMASH architectures and tests them on CIFAR.'
    parser = ArgumentParser(description=usage)
    parser.add_argument(
        '--SMASH', type=str, default=None, metavar='FILE',
        help='The SMASH network .pth file to evaluate.')
    parser.add_argument(
        '--batch-size', type=int, default=100,
        help='Images per batch (default: %(default)s)')
    parser.add_argument(
        '--which-dataset', type=str, default='C100',
        help='Which Dataset to train on (default: %(default)s)')
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Random seed to use.')
    parser.add_argument(
        '--validate', action='store_true', default=True,
        help='Perform validation on validation set (ensabled by default)')
    parser.add_argument(
        '--validate-test', action='store_const', dest='validate',
        const='test', help='Evaluate on test set after every epoch.')
    parser.add_argument(
        '--num-random', type=int, default=500,
        help='Number of random architectures to sample (default: %(default)s)')
    parser.add_argument(
        '--num-perturb', type=int, default=100,
        help='Number of random perturbations to sample (default: %(default)s)')
    parser.add_argument(
        '--num-markov', type=int, default=100,
        help='Number of markov steps to take after perturbation (default: %(default)s)')
    parser.add_argument(
        '--perturb-prob', type=float, default=0.05,
        help='Chance of any individual element being perturbed (default: %(default)s)')
    parser.add_argument(
        '--arch-SGD', action='store_true', default=False,
        help='Perturb archs with architectural SGD. (default: %(default)s)')
    parser.add_argument(
        '--fp16', action='store_true', default=False,
        help='Evaluate with half-precision. (default: %(default)s)')
    parser.add_argument(
        '--parallel', action='store_true', default=False,
        help='Evaluate with multiple GPUs. (default: %(default)s)')
    return parser


if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data

# Chunk Jiterrer for Modelnet data augmentation
def jitter_chunk(src):

    dst = src.copy()
    if np.random.binomial(1, .2):
        dst[ :, ::-1, :, :] = dst
    if np.random.binomial(1, .2):
        dst[ :, :, ::-1, :] = dst
    max_ij = 2
    max_k = 2
    shift_ijk = [np.random.randint(-max_ij, max_ij),
                 np.random.randint(-max_ij, max_ij),
                 np.random.randint(-max_k, max_k)]
    for axis, shift in enumerate(shift_ijk):
        if shift != 0:
            # beware wraparound
            dst = np.roll(dst, shift, axis+1)
    return dst

class MN40(data.Dataset):
    """`Modelnet-40 <modelnet.cs.princeton.edu>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``modelnet40`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        num_rotations: how many rotations of each example to use.
    """


    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False,
                 num_rotations=12):
        self.root = root
        self.transform = jitter_chunk
        self.train = train  # training set or test set
        self.num_rotations = num_rotations # how many rotations to train on


        # now load the picked numpy arrays
        if self.train:
            self.train_data = np.load(self.root + '/modelnet40_rot24_train.npz')['data'][:9840]
            self.train_labels = np.load(self.root + '/modelnet40_rot24_train.npz')['labels'][:9840]
        else:
            self.test_data = np.load(self.root + '/modelnet40_rot24_test.npz')['data']
            self.test_labels = np.load(self.root + '/modelnet40_rot24_test.npz')['labels']


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # Assume for now we're randomly sampling rotations.
        if self.train:
            img, target = self.train_data[index,[np.random.choice(range(0, 24, 24//self.num_rotations))]], self.train_labels[index]
            if self.transform is not None:    
                img = self.transform(img)
        else:
            # Grab lots of rotated examples of instance but note that we will be averaging across rotations so no need to duplicate labels.
            # img = np.asarray([individual_rotation for all_rotations in self.test_data[index] 
                                                          # for individual_rotation in all_rotations[range(0, 24, 24//self.num_rotations)]])
            img = np.asarray([individual_rotation for individual_rotation in self.test_data[index,range(0, 24, 24//self.num_rotations)]])
            target = self.test_labels[index]

        
            
        img = torch.from_numpy(np.float32(img) * 6 - 1)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

class MN10(MN40):
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False,
                 num_rotations=12):
        self.root = root
        self.transform = jitter_chunk
        self.train = train  # training set or test set
        self.num_rotations = num_rotations # how many rotations to train on


        # now load the picked numpy arrays
        if self.train:
            self.train_data = np.load(self.root + '/modelnet10_rot24_train.npz')['data'][:3990]
            self.train_labels = np.load(self.root + '/modelnet10_rot24_train.npz')['labels'][:3990]
        else:
            self.test_data = np.load(self.root + '/modelnet10_rot24_test.npz')['data']
            self.test_labels = np.load(self.root + '/modelnet10_rot24_test.npz')['labels']

# ImageNet32x32 dataset. 
class I1000(data.Dataset):
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
         
        if self.train:
            self.train_data = np.load(self.root + '/imagenet32_train.npz')['data']
            self.train_labels = np.load(self.root + '/imagenet32_train.npz')['labels']
            self.train_data = self.train_data.transpose((0, 2, 3, 1)) 
        else:
            self.test_data = np.load(self.root + '/imagenet32_val.npz')['data']
            self.test_labels = np.load(self.root + '/imagenet32_val.npz')['labels']
            self.test_data = self.test_data.transpose((0, 2, 3, 1)) 
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
            
# Convenience function to centralize all data loaders
def get_data_loader(which_dataset,augment=True,validate=True,batch_size=50,fold='all',num_workers=3):
    class CIFAR10(dset.CIFAR10):
        def __len__(self):
            if self.train:
                return len(self.train_data)
            else:
                return len(self.test_data)


    class CIFAR100(dset.CIFAR100):
        def __len__(self):
            if self.train:
                return len(self.train_data)
            else:
                return len(self.test_data)
    
    # Only need to subclass STL10 to make __init__ args the same as CIFAR
    class STL10(dset.STL10):
        def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False):
            super(STL10, self).__init__(root, split='train' if train else 'test',
                 transform=transform, target_transform=target_transform, download=download)
            self.fold = fold
            fold_indices = np.int64(np.loadtxt('STL/stl10_binary/fold_indices.txt'))
            
            if train:
                if fold != 10: # there are 10 folds so if fold is 10 we presume we're using all
                    print('using fold #%i...'%fold)
                    self.data = np.asarray(self.data)[fold_indices[fold]]
                    self.labels = np.asarray(self.labels)[fold_indices[fold]]
                self.train_data = self.data
                self.train_labels = self.labels
            else:
                self.test_data = self.data
                self.test_labels = self.labels
                
                
    test_batch_size = batch_size
    if which_dataset == 'C10':
        print('Loading CIFAR-10...')
        root = 'cifar'
        norm_mean = [0.49139968, 0.48215827, 0.44653124]
        norm_std = [0.24703233, 0.24348505, 0.26158768]
        dataset = CIFAR10

    elif which_dataset == 'C100' or which_dataset == 100:
        print('Loading CIFAR-100...')
        root = 'cifar'
        norm_mean = [0.50707519, 0.48654887, 0.44091785]
        norm_std = [0.26733428, 0.25643846, 0.27615049]
        dataset = CIFAR100
    elif which_dataset == 'MN40' or which_dataset == 40:
        print('Loading ModelNet-40...')
        root = 'modelnet'
        norm_mean = [0,0,0] # dummy mean
        norm_std = [1,1,1] # dummy std
        dataset = MN40
        test_batch_size = int(np.ceil(batch_size / 10.)) # For parallelism
    elif which_dataset == 'MN10':
        print('Loading ModelNet-10...')
        root = 'modelnet'
        norm_mean = [0,0,0] # dummy mean
        norm_std = [1,1,1] # dummy std
        dataset = MN10
        test_batch_size = int(np.ceil(batch_size / 10.)) # For parallelism
    elif which_dataset == 'I1000' or which_dataset == 1000:
        print('Loading 32x32 Imagenet-1000...')
        root = 'imagenet'
        norm_mean = [0.48109809447859192, 0.45747185440340027, 0.40785506971129742]
        norm_std = [0.26040888585626459, 0.25321260169837184, 0.26820634393704579]
        dataset = I1000
    elif which_dataset == 'STL10':
        root = 'STL'
        print('Loading STL-10...')
        norm_mean = [0.4467106206597222, 0.439809839835240, 0.406646447099673]
        norm_std =  [0.2603409782662329, 0.256577273113443, 0.271267381452256]
        dataset = STL10
    if augment:
        print('Data will be augmented...')
    
    # Prepare transforms and data augmentation
    norm_transform = transforms.Normalize(norm_mean, norm_std)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        norm_transform
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        norm_transform
    ])
    kwargs = {'num_workers': num_workers, 'pin_memory': True}

    train_set = dataset(
        root=root,
        train=True,
        download=True,
        transform=train_transform if augment else test_transform)
    # If we're evaluating on the test set, load the test set
    if validate == 'test':
        print('Using test set...')
        test_set = dataset(root=root, train=False, download=True,
                           transform=test_transform)

    # If we're evaluating on the validation set, prepare validation set
    # as the last 5,000 samples in the training set.
    elif validate:
        print('Using validation set...')
         
        test_set = dataset(root=root, train=True, download=True,
                           transform=test_transform)
        # Validation split size
        val_split = int(0.1 * len(test_set.train_data))
        if which_dataset != 'STL':
            test_set.train_data = test_set.train_data[-val_split:]
            test_set.train_labels = test_set.train_labels[-val_split:]
            train_set.train_data = train_set.train_data[:-val_split]
            train_set.train_labels = train_set.train_labels[:-val_split]
        else:
            test_set.data = test_set.data[-val_split:]
            test_set.labels = test_set.labels[-val_split:]
            train_set.data = train_set.data[:-val_split]
            train_set.labels = train_set.labels[:-val_split]
    # Prepare data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, **kwargs)
    test_loader = DataLoader(test_set, batch_size=test_batch_size,
                             shuffle=False, **kwargs)
    return train_loader, test_loader
    
''' MetricsLogger originally stolen from VoxNet source code.'''
class MetricsLogger(object):

    def __init__(self, fname, reinitialize=False):
        self.fname = path.Path(fname)
        self.reinitialize = reinitialize
        if self.fname.exists():
            if self.reinitialize:
                logging.warn('{} exists, deleting'.format(self.fname))
                self.fname.remove()

    def log(self, record=None, **kwargs):
        """
        Assumption: no newlines in the input.
        """
        if record is None:
            record = {}
        record.update(kwargs)
        record['_stamp'] = time.time()
        with open(self.fname, 'a') as f:
            f.write(json.dumps(record, ensure_ascii=True)+'\n')


def read_records(fname):
    """ convenience for reading back. """
    skipped = 0
    with open(fname, 'rb') as f:
        for line in f:
            if not line.endswith('\n'):
                skipped += 1
                continue
            yield json.loads(line.strip())
        if skipped > 0:
            logging.warn('skipped {} lines'.format(skipped))
            
"""
Very basic progress indicator to wrap an iterable in.

Author: Jan Schlüter
"""
def progress(items, desc='', total=None, min_delay=0.1):
    """
    Returns a generator over `items`, printing the number and percentage of
    items processed and the estimated remaining processing time before yielding
    the next item. `total` gives the total number of items (required if `items`
    has no length), and `min_delay` gives the minimum time in seconds between
    subsequent prints. `desc` gives an optional prefix text (end with a space).
    """
    total = total or len(items)
    t_start = time.time()
    t_last = 0
    for n, item in enumerate(items):
        t_now = time.time()
        if t_now - t_last > min_delay:
            print("\r%s%d/%d (%6.2f%%)" % (
                    desc, n+1, total, n / float(total) * 100), end=" ")
            if n > 0:
                t_done = t_now - t_start
                t_total = t_done / n * total
                print("(ETA: %d:%02d)" % divmod(t_total - t_done, 60), end=" ")
            sys.stdout.flush()
            t_last = t_now
        yield item
    t_total = time.time() - t_start
    print("\r%s%d/%d (100.00%%) (took %d:%02d)" % ((desc, total, total) +
                                                   divmod(t_total, 60)))

    
# Get factors of a given number
# Taken from this stackexchange answer:
# https://stackoverflow.com/a/6800214
# I haven't bothered to grok its internals, but it works.
def factors(n):
    assert n>0
    return sorted(set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0))))
                
# Simple softmax
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
    

#count_params.py
# This function calculates the number of parameters of a resulting (non-SMASH)
# architecture. It supports either specifying the standard 1x1-followed-by-3x3s,
# or allows for there to be no preceding 1x1.
# It currently makes a lot of assumptions about the network, so be wary of its accuracy.
# 
def count_params(arch):
    (incoming, outgoing, G, 
     ops, gate, dilation, activation,
     bank_sizes, kernel_sizes, groups,
     N, N_max, bottleneck, max_bottleneck, in_channels, stdev, err) = arch
    # Initialize params with cost of first conv 
    # params = 3*(3*3)* net.base * net.width
    params = 0
    # Memory bank sizes
    m = [[None 
              for _ in range(max(max([max(item) for item in outgo])+ 1, inch // (bank_size * N)))]
           for hw, outgo, inch, bank_size in zip((32, 16, 8), outgoing, in_channels, bank_sizes)]
    for block_index, (incoming_channels, outgoing_channels,g_values,
            op_values, gate_values, dilation_values, nl_values, 
            bs, kernel_values, group_values) in enumerate(zip(
                incoming, outgoing, G, 
                ops, gate, dilation, activation,
                bank_sizes, kernel_sizes, groups)):

        for op_index, (read, write, g,
            op, gated, dilate, nls,
            ks, group) in enumerate(zip(incoming_channels, 
                                          outgoing_channels, 
                                          g_values,
                                          op_values,
                                          gate_values,
                                          dilation_values,
                                          nl_values,
                                          kernel_values,
                                          group_values)):
            
            # Number of incoming channels
            n_in = bank_sizes[block_index] * len(read) * N
            # Bottleneck_size
            n_b = min(min(min(max(n_in // (g * N), 1), bottleneck) * g * N, n_in), max_bottleneck * N_max)
            # Number of outgoing channels
            n_out = g * N 
            
            # Add 1x1 BN+Conv params
            params += n_in + n_in * n_b 
            
            # Add   BN + (ks x ks) conv params for each active conv.
            for oi, o in enumerate(op):
                if o:
                    params += (n_out if oi %2 else n_b) + (n_out if oi %2 else n_b) * n_out * ks[oi][0] * ks[oi][1] // group[oi]
        # Add transition parameters
        n_in = len(m[block_index]) * bank_sizes[block_index] * N           
        params += n_in + n_in * (in_channels[block_index+1] if block_index < 2 else 100)
    return params
 
# Note that this is more like "ballpark flops" and ignores some costs like
# batchnorm and the initial conv. 
# It currently makes a lot of assumptions about the network, so be wary of its accuracy.
def count_flops(arch, which_dataset='cifar'):
    (incoming, outgoing, G, 
     ops, gate, dilation, activation,
     bank_sizes, kernel_sizes, groups,
     N, N_max, bottleneck, max_bottleneck, in_channels, stdev, err) = arch
    # Initialize params with cost of first conv 
    # params = 3*(3*3)* net.base * net.width
    flops = 0
    
    # Heights and widths
    hws = (48,24,12) if which_dataset=='STL' else (32,16,8)
    # Memory bank sizes
    m = [[None 
              for _ in range(max(max([max(item) for item in outgo])+ 1, inch // (bank_size * N)))]
           for hw, outgo, inch, bank_size in zip(hws, outgoing, in_channels, bank_sizes)]
    for block_index, (incoming_channels, outgoing_channels,g_values,
            op_values, gate_values, dilation_values, nl_values, 
            bs, kernel_values, group_values, hw) in enumerate(zip(
                incoming, outgoing, G, 
                ops, gate, dilation, activation,
                bank_sizes, kernel_sizes, groups,hws)):

        for op_index, (read, write, g,
            op, gated, dilate, nls,
            ks, group) in enumerate(zip(incoming_channels, 
                                          outgoing_channels, 
                                          g_values,
                                          op_values,
                                          gate_values,
                                          dilation_values,
                                          nl_values,
                                          kernel_values,
                                          group_values)):
            
            # Number of incoming channels
            n_in = bank_sizes[block_index] * len(read) * N
            # Bottleneck_size
            n_b = min(min(min(max(n_in // (g * N), 1), bottleneck) * g * N, n_in), max_bottleneck * N_max)
            # Number of outgoing channels
            n_out = g * N 
            
            # Add 1x1 Conv ops
            flops += hw * hw * n_in * n_b
            
            # params += n_in + n_in * n_b 
            
            # Add   BN + (ks x ks) conv params for each active conv.
            for oi, o in enumerate(op):
                if o:
                    flops+=  hw * hw * (n_out if oi %2 else n_b) * n_out * ks[oi][0] * ks[oi][1] // group[oi]
        # Add transition parameters
        # n_in = len(m[block_index]) * bank_sizes[block_index] * N           
        # params += n_in + n_in * (in_channels[block_index+1] if block_index < 2 else 100)
    return flops