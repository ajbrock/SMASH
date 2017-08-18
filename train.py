#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
SMASH Training Function
Andy Brock, 2017

This script trains and tests a SMASH network, or a resulting network.

Based on Jan Schlüter's DenseNet training code:
https://github.com/Lasagne/Recipes/blob/master/papers/densenet
'''

import os
import logging
import sys


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils import train_parser, get_data_loader, MetricsLogger, progress, count_flops

# Set the recursion limit to avoid problems with deep nets
sys.setrecursionlimit(5000)
def train_test(depth, width, N, N_max, which_dataset,
               bottleneck, max_bottleneck, depth_compression,
               max_dilate, max_kernel, max_groups, var_op, big_op, long_op,
               gates, op_bn, preactivation,
               var_nl, var_ks, var_group,
               seed, augment, validate, epochs, save_weights, batch_size, 
               resume, model, SMASH, rank, init_from_SMASH,
               fp16, parallel, validate_every, duplicate_at_checkpoints, fold,
               top5):
    
    # Quick hack to get # classes
    i = 2
    nClasses = int(which_dataset[-i:])
    while nClasses == 0:
        i += 1
        nClasses = int(which_dataset[-i:])
    
    # Check to see if we're actually using SMASH or just using this as 
    # boilerplate.
    SMASHING = True if SMASH is None and model[:5] == 'SMASH' else False
    
    # Seed RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    # Create logs and weights folder if they don't exist
    if not os.path.exists('logs'):
        os.mkdir('logs')
    if not os.path.exists('weights'):
        os.mkdir('weights')
        
    # Name of the file to which we're saving losses and errors.
    if save_weights is 'default_save':
        save_weights = '_'.join([item for item in 
                                [model,
                                'D' + str(depth) if SMASH is None else None,
                                'K' + str(width) if SMASH is None else None,
                                'N'+str(N) if SMASHING else None,
                                'Nmax' + str(N_max) if SMASHING else None,
                                'maxbneck' + str(max_bottleneck) if SMASHING else None,
                                'op_bn' if op_bn else None,
                                'varop' if var_op else None,
                                'bigop' if big_op else None,
                                'longop' if long_op else None,
                                'static_ks' if not var_ks else None,
                                'static_group' if not var_group else None,
                                'gates' if gates else None,
                                'postac' if not preactivation else None,
                                'SMASH' if SMASHING else 'Main_'+SMASH if model[:5] == 'SMASH' else None,
                                'Rank' + str(rank) if SMASH is not None else None,
                                'fp16' if fp16 else None,
                                which_dataset,
                                'seed' + str(seed),
                                str(epochs)+'epochs'] if item is not None])
    metrics_fname = 'logs/' + save_weights + '_log.jsonl'
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s| %(message)s')
    logging.info('Metrics will be saved to {}'.format(metrics_fname))
    mlog = MetricsLogger(metrics_fname, reinitialize=(not resume))
    
    # Import the model module
    model_module = __import__(model)

    # Build network, either by initializing it or loading a pre-trained
    # network.  
    if resume:
        logging.info('loading network ' + save_weights + '...')
        net = torch.load('weights/'+save_weights + '.pth')

        if parallel:
            parnet = torch.nn.DataParallel(net)
        if fp16:
            net = net.half()
            
        # Which epoch we're starting from
        start_epoch = net.epoch + 1 if hasattr(net, 'epoch') else 0
        
        # Rescale iteration counter if batchsize requires it.
        if not hasattr(net,'batch_size'):
            print('resetting batch size')
            net.batch_size = 50
        net.j = int(net.j * net.batch_size / float(batch_size))
        net.batch_size = batch_size

    # If the model name doesn't start with SMASH, assume we're trying out
    # either VGG or a ResNet or a DenseNet
    elif model[:5] != 'SMASH':
        net = model_module.Network(width, depth,
                                    nClasses=nClasses,
                                    epochs=epochs)
        net = net.cuda()
        net.batch_size = batch_size
        if fp16:
            net = net.half()
        if parallel:
            parnet = torch.nn.DataParallel(net)
        start_epoch = 0
    # If we're SMASHing, instantiate SMASH net
    elif SMASH is None:
        logging.info('Instantiating SMASH network with model ' + model + '...')
        net = model_module.SMASH(depth=depth, width=width,
                                 N=N, N_max=N_max,
                                 nClasses=nClasses,
                                 bottleneck=bottleneck,  
                                 max_bottleneck=max_bottleneck,
                                 depth_compression=depth_compression,
                                 max_dilate=max_dilate,
                                 max_kernel=max_kernel,
                                 max_groups=max_groups,
                                 var_op=var_op, 
                                 big_op=big_op,
                                 long_op=long_op,
                                 gates=gates,
                                 batchnorm=op_bn,
                                 preactivation=preactivation, 
                                 var_nl=var_nl,
                                 var_ks=var_ks,
                                 var_group=var_group)
        net = net.cuda()
        net.batch_size = batch_size
        if fp16:
            net = net.half()
            if hasattr(net,'c'):
                net.c = net.c.half()
            if hasattr(net,'z'):
                net.z = net.z.half()
        if parallel:
            parnet = torch.nn.DataParallel(net)
        start_epoch = 0
    
    # If we're not SMASHing we must be using a derivative network.
    else:
        logging.info('Instantiating main network with model ' + model
                    + ', and SMASH network ' + SMASH +'...')
        
        
        # Load results of SMASH evaluation
        archs = np.load(SMASH+'_archs.npz', encoding='bytes')['archs']
        
        print('Using architecture rank %i with SMASH score of %f and estimated FLOPs of %e.'%(rank, archs[rank][-1], count_flops(archs[rank],which_dataset)))
        
        # logging.info('Using architecture rank ' + str(rank)
                    # + ' with SMASH score of ' + str(archs[rank][-1])
                    # +', estimated FLOPs are ' + str(count_flops(archs[rank])))
        
        net = model_module.MainNet(list(archs[rank]), nClasses=nClasses,
                                   var_ks=var_ks, var_op=var_op,
                                   big_op=big_op, op_bn=op_bn)
        net = net.cuda()
        net.batch_size = batch_size
        if parallel:
            parnet = torch.nn.DataParallel(net)
        if fp16:
            net = net.half()
        start_epoch = 0
        
        # If we wish to initialize our derivative network using the parameters
        # predicted by the SMASH network, call this.
        if init_from_SMASH:
            logging.info('Initializing using SMASH parameters...')
            SMASH_net = torch.load(SMASH+'.pth')
            net.init_from_SMASH(SMASH_net.sample_weights(*archs[rank][:10]),
                                SMASH_net.W, SMASH_net.conv1,
                                SMASH_net.trans1, SMASH_net.trans2,
                                SMASH_net.bn1, SMASH_net.fc)
            del(SMASH_net)

    logging.info('Number of params: {}'.format(
                 sum([p.data.nelement() for p in net.parameters()]))
                 )
    # Get information specific to each dataset
    train_loader,test_loader = get_data_loader(which_dataset, augment,
                                               validate, batch_size,
                                               fold)
    # Training Function, presently only returns training loss
    # x: input data
    # y: target labels
    def train_fn(x, y):
        net.optim.zero_grad()
        input = V(x.cuda().half()) if fp16 else V(x.cuda())
        # If training a SMASH net, we will not have been provided a pre-trained SMASH net.
        if SMASH is None and model[:5] == 'SMASH':
            arch = net.sample_architecture()
            w = net.sample_weights(*arch)
            # input = [input,w,arch]
            if parallel:
                # the cat() call here is to ensure that one w gets passed to each
                # GPU. Probably more robust to have it be something like
                # [w]*num_devices but we'll roll with this for now.
                output = parnet(input, torch.cat([w]*len(parnet.device_ids),0), *arch) 
            else:
                output = net(input,w,*arch)
        else:
            if parallel:
                output = parnet(input)
            else:
                output = net(input)

        loss = F.nll_loss(output, V(y.cuda()))
        training_error = output.data.max(1)[1].cpu().ne(y).sum()
        loss.backward()
        net.optim.step()
        return loss.data[0], training_error

    # Testing function, returns test loss and test error for a batch
    # x: input data
    # y: target labels
    def test_fn(x, y):
    
        # the cat() call here is to ensure that one w gets passed to each
        # GPU. Probably more robust to have it be something like
        # [w]*num_devices but we'll roll with this for now.
        input = V(x.cuda().half(), volatile=True) if fp16 else V(x.cuda(), volatile=True)
        if SMASH is None and model[:5] == 'SMASH':
            arch = net.sample_architecture()
            w = net.sample_weights(*arch)
            # input = [input,w,arch]
            if parallel:
                output = parnet(input, torch.cat([w]*len(parnet.device_ids),0), *arch) 
            else:
                output = net(input,w,*arch)
        else:
            if parallel:
                output = parnet(input)
            else:
                output = net(input)
            
        test_loss = F.nll_loss(output, V(y.cuda(), volatile=True)).data[0]

        # If we're running Imagenet, we may want top-5 error:
        if top5:
            top5_preds = np.argsort(output.data.cpu().numpy())[:,:-6:-1]
            test_error = len(y) - np.sum([np.any(top5_i == y_i) for top5_i, y_i in zip(top5_preds,y)])
        else:
            # Get the index of the max log-probability as the prediction.
            pred = output.data.max(1)[1].cpu()
            test_error = pred.ne(y).sum()

        return test_loss, test_error

    # Finally, launch the training loop.
    logging.info('Starting training at epoch '+str(start_epoch)+'...')
    for epoch in range(start_epoch, epochs):

        # Pin the current epoch on the network.
        net.epoch = epoch

        # shrink learning rate at scheduled intervals, if desired
        if 'epoch' in net.lr_sched and epoch in net.lr_sched['epoch']:

            logging.info('Annealing learning rate...')

            # Optionally checkpoint at annealing
            if net.checkpoint_before_anneal:
                torch.save(net,'weights/' + str(epoch) + '_' + save_weights + '.pth')

            for param_group in net.optim.param_groups:
                param_group['lr'] *= 0.1

        # List where we'll store training loss
        train_loss, train_err = [], []

        # Prepare the training data
        batches = progress(
            train_loader, desc='Epoch %d/%d, Batch ' % (epoch + 1, epochs),
            total=len(train_loader.dataset) // batch_size)

        # Put the network into training mode
        net.train()

        # Execute training pass
        for x, y in batches:
        
            # Update LR if using cosine annealing
            if 'itr' in net.lr_sched:
                net.update_lr(epochs*len(train_loader.dataset) // batch_size)
            loss, err = train_fn(x, y)
            train_loss.append(loss)
            train_err.append(err)

        # Report training metrics
        train_loss = float(np.mean(train_loss))
        train_err = 100 * float(np.sum(train_err)) / len(train_loader.dataset)
        print('  training loss:\t%.6f, training error: \t%.2f%%' % (train_loss, train_err))
        mlog.log(epoch=epoch, train_loss=train_loss, train_err=train_err)

        # Optionally, take a pass over the validation or test set.
        if validate and not ((epoch+1) % validate_every):

            # Lists to store
            val_loss, val_err = [], []

            # Set network into evaluation mode
            net.eval()

            # Execute validation pass
            for x, y in test_loader:
                loss, err = test_fn(x, y)
                val_loss.append(loss)
                val_err.append(err)

            # Report validation metrics
            val_loss = float(np.mean(val_loss))
            val_err =  100 * float(np.sum(val_err)) / len(test_loader.dataset)
            print('  validation loss:\t%.6f' % val_loss)
            print('  validation error:\t%.2f%%' % val_err)
            mlog.log(epoch=epoch, val_loss=val_loss, val_err=val_err)

        # Save weights for this epoch
        print('saving weights to ' + save_weights + '...')
        torch.save(net, 'weights/' + save_weights + '.pth')
        
        # If requested, save a checkpointed copy with a different name
        # so that we have them for reference later.
        if duplicate_at_checkpoints and not epoch%5:
            torch.save(net, 'weights/' + save_weights + '_e' + str(epoch) + '.pth')

    # At the end of it all, save weights even if we didn't checkpoint.
    if save_weights:
        torch.save(net, 'weights/' + save_weights + '.pth')


def main():
    # parse command line
    parser = train_parser()
    args = parser.parse_args()
    print(args)
    # run
    train_test(**vars(args))


if __name__ == '__main__':
    main()
