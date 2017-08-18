### Evaluate.py
# 
# Andy Brock, 2017
#
# This script takes in a SMASH network, then samples and evaluates a set
# number of network architectures on a validation set.

import time
import torch
import numpy as np
from torch.autograd import Variable as V
import torch.nn.functional as F
from utils import get_data_loader, factors, eval_parser, count_params, count_flops
import perturb_arch
# from importlib import reload # For use in debugging in python3
from copy import deepcopy

def evaluate(SMASH, which_dataset, batch_size, seed, validate, 
             num_random, num_perturb, num_markov,
             perturb_prob, arch_SGD, fp16, parallel):


    # Random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


    num_runs = num_random + num_perturb + num_markov
    random_sample = True
    perturb = False
    markov = False


    net = torch.load('weights/'+SMASH+'.pth')
    net.eval()
    
    # Backwards compatibility hack; If you're trying to understand this code,
    # ignore this line.
    if not hasattr(net,'factors'):
        net.factors = factors(net.N)

    _,test_loader = get_data_loader(which_dataset=which_dataset, augment=False, validate=validate, batch_size=batch_size)


    # Prepare lists that hold errors 
    ensemble_err, err, flops, params = [],[], [], []

    # Array to which we save configurations and errors
    save_archs = []

    # Prepare ensemble predictions
    ensemble_out = torch.zeros(len(test_loader.dataset),net.fc.out_features).cuda()

    # Start the stopwatch and begin testing
    start_time = time.time()
    mode = 'training' if net.training else 'testing'
    print('Evaluating %s in %s mode...'%(SMASH,mode))
    for test in range(num_runs):
        
        # If we've done all our random samples, switch to random perturbation mode
        if test == num_random:
            sorted_archs = sorted(save_archs, key = lambda item: item[-1])
            print('Random sampling complete with best error of %f, starting perturbation...'%(sorted_archs[0][-1]))
            base_arch = sorted_archs[0][:10]
            perturb = True
            random_sample = False
        
        # If we've done all our perturbations, switch to markov chain mode
        elif test == num_random + num_perturb:
            sorted_archs = sorted(save_archs, key = lambda item: item[-1])
            print('Random perturbation complete with best error of %f, starting markov chain...'%(sorted_archs[0][-1]))
            base_arch = sorted_archs[0][:10]
            current_error = sorted_archs[0][-1]
            markov = True
        
            
        # Sample a random architecture, as in training
        if random_sample:
            arch = net.sample_architecture()
            
        # Slightly change a sampled (and, presumably, high-scoring) architecture
        elif perturb:
            arch = perturb_arch.perturb_architecture(net, deepcopy(base_arch), perturb_prob)
        
        #Sample Weights
        w1x1 = net.sample_weights(*arch)
        
        # Error counters
        e,ensemble_e = 0, 0
        
        # Loop over validation set
        for i, (x, y) in enumerate(test_loader):

            # Get outputs
            o = net(V(x.cuda(),volatile=True), w1x1, *arch)
            
            # Get predictions ensembled across multiple configurations
            ensemble_out[i*batch_size:(i + 1)*batch_size] += o.data
            
            # Update error
            e += o.data.max(1)[1].cpu().ne(y).sum()
            
            # Update ensemble error
            ensemble_e += ensemble_out[i*batch_size:(i + 1)*batch_size].max(1)[1].cpu().ne(y).sum()

        # Save ensemble error thus far
        ensemble_err.append(float(ensemble_e) / ensemble_out.size(0))

        # Save individual error thus far
        err.append(float(e) / ensemble_out.size(0))
        
        # While in markov mode, update the base arch if we get a better SMAS hscore.
        if markov and err[-1] < float(current_error):
            print('Error of %f superior to error of %f, accepting new architecture...'%(err[-1], current_error))
            base_arch = arch
            current_error = err[-1]
            
        # Save relevant architectural details along with error
        save_archs.append(arch + (net.N, net.N_max, net.bottleneck, net.max_bottleneck, net.in_channels, 0 ,err[-1]))
        
        params.append(count_params(save_archs[-1]))
        flops.append(count_flops(save_archs[-1],which_dataset))
        print('For run #%d/%d, Individual Error %2.2f Ensemble Err %2.2f, params %e, flops %e,  Time Elapsed %d.'%(test,num_runs, 100*err[-1], 100*ensemble_err[-1], params[-1], flops[-1], time.time()-start_time))#LogSof EnsErr %d, Softmax EnsErr %d ensemble_olgs_err[-1],  ensemble_os_err[-1],
    
    best_acc = sorted(err)[0]
    worst_acc = sorted(err)[-1]
    least_flops = sorted(flops)[0]
    most_flops = sorted(flops)[-1]
    least_params = sorted(params)[0]
    most_params = sorted(params)[-1]
    print('Best accuracy is '+str(best_acc)+', Worst accuracy is '+str(worst_acc))
    
    # Save results
    # np.savez(filename[:-4] + '_' + mode + '_errors.npz', **{'err':err, 'ensemble_err':ensemble_err})
    # save_archs = sorted(save_archs, key = lambda item: item[-1])
    np.savez(SMASH + '_archs.npz', **{'archs': sorted(save_archs, key = lambda item: item[-1]), 'unsorted_archs':save_archs})

def main():
    # parse command line
    parser = eval_parser()
    args = parser.parse_args()
    print(args)
    # run
    evaluate(**vars(args))

if __name__ == '__main__':
    main()
