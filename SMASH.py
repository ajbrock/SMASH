'''
SMASH

Andy Brock, 2017

This code contains the model definitions for SMASH networks and derivative
networks as described in my paper,
"One-Shot Model Architecture Search through HyperNetworks."

This code is thoroughly commented throughout, but is still rather complex.
If there's something that's unclear please feel free to ask and I'll do my best
to explain it or update the comments to better describe what's going on.
'''
import sys
import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn import Parameter as P
from torch.autograd import Variable as V

from utils import factors, softmax
from layers import (SMASHC2D, SMASHLayer, 
                   SMASHTransition, SMASHFC, 
                   SMASHBN2D, wn2d, WNC2D,
                   CL, Layer, Transition) 

# Class defining a SMASH network.
# This network

# Things to note:
# The parametric budget was originally defined with respect to a DenseNet, but
# is now more equivalent to that of a Wide-ResNet parameterization.
# so if you supply k = 12 and depth = 12 (where depth is the number of
# layers per block, rather than overall layers), you'll be allocating a SMASH
# network with a maximum parameter budget equivalent to a  WRN-40-12 with
# growth rate of 12 and overall depth of 40.
# In the comments where I talk about "reference blocks" I'm referring
# to arithmetic associated with calculating the parametric budget, as opposed to
# defining actual parameters that will be used.


class SMASH(nn.Module):
    def __init__(self, depth=12, width=4, N=8, N_max=64, nClasses=100,
                bottleneck=4, max_bottleneck=4, depth_compression=4,
                max_dilate=3, max_kernel=7,  max_groups=8,
                 var_op=False, big_op=False, long_op = False, 
                 gates=False, batchnorm=False, preactivation=True, 
                 var_nl=False, var_ks=True, var_group=True):
        super(SMASH, self).__init__()
    
        # Reference number of layers per block for defining parametric budget.
        # This is equal to twice the number of equivalent residual blocks,
        # e.g. the total number of convs in a standard resenet block.
        self.depth = depth
        
        # Reference widening factor for defining parametric budget.
        self.width = width

        # Base Number of channels in a memory bank
        self.N = N

        # Maximum number of units in a layer and max memory bank size
        self.N_max = N_max

        # Ratio of maximum number of units to memory bank size
        self.k = self.N_max // self.N
        
        # Number of output classes
        self.nClasses = nClasses
        
        # base factor, by default 16 for WRN. All block widths are defined as a
        # multiple of this value.
        self.base = 2 * self.N
        
        # Depth compression: 
        # Each trailing slot corresponds to depth_compression * k memory-bank-reads.
        self.depth_compression = depth_compression
        
        # Bottleneck: Every 1x1 conv has N_READ incoming channels and 
        # bottleneck * G outgoing channels, where G is in [N:N_max:N]
        self.bottleneck = bottleneck
        self.max_bottleneck = max_bottleneck
        
        # Maximum dilation value
        self.max_dilate = max_dilate
        
        # Maximum filter size, varying from 3:max_filter:2
        self.var_ks = var_ks
        # This is the value to which we'll anneal the kernel size
        self.final_max_kernel = max_kernel if self.var_ks else 3 
        self.max_kernel = 3 # Initial max kernel size.
        
        # The maximum number of convolutional groups
        self.max_groups = max_groups
        
        # Utility variable to ensure we only select properly divisible groups.
        self.factors = factors(self.N)[:factors(self.N).index(self.max_groups)+1]
        # Variable groups or static?
        self.var_group = var_group
        
        # Dictionary for variable nonlinearities
        # Presently we're just considering ReLU, but we might do others.
        self.var_nl = var_nl
        self.gates = gates
        if self.var_nl:
            self.nl_dict = {0: F.relu,
                        # 1: F.softplus,
                        # 2: F.softshrink,
                        1: F.tanh,
                        2: F.sigmoid
                        # 4: F.rrelu,
                        # 5: F.elu,
                         }
        elif self.gates:
            self.nl_dict = {0: F.relu,
                        1: F.tanh,
                        2: F.sigmoid}
        else:
            self.nl_dict = {0: F.relu,
                        # 1: F.softplus,
                        # 2: F.softshrink,
                        # 3: F.tanh,
                        # 4: F.rrelu,
                        # 5: F.elu,
                         }
        
        # Possible op configurations; 
        # Note that we don't bother to have the option to have w[2] 
        # alone, since although in the SMASH network that would 
        # be different, the resulting network would not be different
        # (i.e. it would just be two ways to define a single conv)
        self.var_op = var_op
        self.big_op = big_op
        self.long_op = long_op
        
        # Use batchnorm in ops?
        self.batchnorm = batchnorm
        
        # Pre or post activation?
        self.preactivation = preactivation
        
        # Possible op configurations
        if self.var_op:
            self.options = [[1, 0, 0, 0], [1, 0, 1, 0], 
                            [1, 1, 0, 0], [1, 1, 1, 0], 
                            [1, 1, 1, 1]]
            # Probability of each option, hand defined to give more preference
            # to using the full 2x2 op.
            self.options_probabilities = [0.05, 0.1, 0.15, 0.25, 0.45]

        elif self.big_op:
            self.options = [[1, 1, 1, 1]]        
            self.options_probabilities = [1.0]

        elif self.long_op:
            self.options = [[1, 1, 0, 0]]          
            self.options_probabilities = [1.0]
        else:
            self.options = [[1, 0, 0, 0]]       
            self.options_probabilities = [1.0]
        
        # W is the array of freely learned convolutions, shared across a block.
        # It can have up to 4 convs per block.
        self.W = nn.ModuleList([
                    SMASHLayer(n_in=self.max_bottleneck * self.N_max, # was (1 + 3 * ((wi + 1) % 2)) * self.N_max,
                               n_out=self.N_max,
                               batchnorm=self.batchnorm,
                               preactivation=self.preactivation,
                               kernel_size=self.final_max_kernel)
                    for bi in range(3)])

        # Our "stem" convolution
        self.conv1 = nn.Conv2d(3, self.base * self.width, kernel_size=7, padding=3,
                               bias=False, stride=1)
        
        
        ''' Parametric Budget Definition
            This snippet determines the maximum parametric budget, as well as
            the maximum number of banks in each block. It's nominally defined
            with respect to a baseline WRN.'''
        nChannels = self.base * self.width
        # List indicating number of channels incoming to a block
        self.in_channels = [nChannels]

        nch1 = nChannels  +  self.depth * (self.base * self.width) 
        nChannels = nChannels + 2 * self.base * self.width
        self.widths = [nChannels]
        self.trans1 = SMASHTransition(nChannels, nChannels // 2 // self.base * self.base)
        
        nChannels = nChannels // 2 // self.base * self.base
        
        # Number of input channels to the second block
        self.in_channels.append(nChannels)
        
        nch2 = nChannels  +  self.depth * (2 * self.base * self.width) 
        nChannels = nChannels + 2 * self.base * self.width
        
        # Max width of second block
        self.widths.append(nChannels)
        
        self.trans2 = SMASHTransition(nChannels, nChannels // 2 // self.base * self.base)
        
        nChannels = nChannels // 2 // self.base * self.base
        # Number of input channels to the third block
        self.in_channels.append(nChannels)
        
        nch3 = nChannels  + self.depth * (4 * self.base * self.width) 
        nChannels = 4 * self.base * width
        
        # Max width of third block
        self.widths.append(nChannels)
       

        # Output layer: BatchNorm followed by a fully-connected layer
        self.bn1 = SMASHBN2D(nChannels)
        self.fc = SMASHFC(nChannels, self.nClasses)

        # List of parametric budgets for each block
        self.nch_list = [nch1, nch2, nch3]

        # Total number of channels
        self.nch = nch1 + nch2 + nch3

        # Remember #channels at output of third block
        self.nChannels = nChannels
        
        # Maximum number of memory banks
        self.max_banks = self.nChannels // self.N
        
        # Print some details out to give a sense of the size of the network.
        print('Channels per block: '
              + ', '.join([str(item) for item in self.nch_list])
              + ', Total Channels: ' + str(self.nch)
              + ', Input Channels to each block: '
              + ', '.join([str(item) for item in self.in_channels])
              + ', Max Channels for each block: '
              + ', '.join([str(item) for item in self.widths])
              +', nChannels: ' +str(nChannels))

        # Random Embedding tensor, z~N(0,I)
        self.z = torch.randn(1,
                                  1,
                                  self.k,
                                  self.nch // self.N // self.depth_compression).cuda()
        
        # Architecture-conditional Embedding Tensor, c.
        self.c = torch.zeros(1,
                             (2 * self.max_banks # Read and write locations
                               + 3 # Block-conditional locations
                               + self.max_dilate * 2 *(1 + 3 * (self.var_op or self.big_op))  # Dilation conditional locations, 4 if using 2x2 ops, 1 otherwise.
                               + 3 * 2 *  (1 + 3 * (self.var_op or self.big_op)) * self.var_ks  # kernel conditional locations, 4 if using 2x2 ops, 1 otherwise.
                               + 4 * (self.var_op)# Op-conditional Locations
                               + len(self.factors) * (1 + 3 * (self.var_op or self.big_op)) * self.var_group
                               + 2 * self.gates # Gate-conditional locations
                               + self.k # G-conditional location
                               + (1 + 3 * (self.var_op or self.big_op)) * len(self.nl_dict) * self.var_nl), # NL-conditional locations
                             self.k,
                             self.nch // self.N // self.depth_compression).cuda()

        

        # Define the HyperNet
        # The hypernet I use is an ad-hoc DenseNet without downsampling,
        # that uses simple weight normalized 2D Convs without biases,
        # Leaky ReLU, and no batchnorm. Leaky ReLU was selected because early
        # in development I was getting NaNs and the regular ReLUs were hiding
        # them early in the network. One should presumably be able to change
        # from LReLU to regular ReLU but I haven't bothered, in case of NaNs.

        # HyperNet parameters, chosen ad-hoc (and never changed or tuned, lol)

        # Growth Rate for each block
        hG=[10, 10, 10]

        # Number of layers per block
        hD=[8, 10, 4]

        # This definition uses a set of nested for loops, but could also be
        # written more compactly as a list comprehension. The list comprehension
        # seemed ugly and unreadable to all who had not un-slain Tilpuduck,
        # the One Who Cannot Be Redeemed With Any Other Offer. So I changed it
        # to a more verbose version to accomodate the eyes.
        
        # Then, there was a bug in the verbose version, so I commented it out
        # and replaced it with the ugly version which I know to work. I left
        # the verbose version here in case anyone is struggling with the list
        # comprehension, but it's really just a simple DenseNet so I
        # wouldn't recommend wasting too much time trying to grok it.
        
        # self.hyperconv = [CL(WNC2D(1 + self.c.size(1),
                                                  # 7,
                                                  # kernel_size=3,
                                                  # padding=1,
                                                  # bias=False))]

        # # Number of channels thus far in the hypernet
        # hch=1 + self.c.size(1) + 7

        # for i, (d, g) in enumerate(zip(hD, hG)):
            # for j in range(d):
                # self.hyperconv.append(CL(
                    # nn.Sequential(nn.LeakyReLU(0.02),
                                    # WNC2D(hch,
                                    # g,
                                    # kernel_size=3,
                                    # padding=1,
                                    # bias=False)
                                  # )
                                 # ))
                # hch += g
            # self.hyperconv.append(CL(
                    # nn.Sequential(nn.LeakyReLU(0.02),
                                    # WNC2D(hch,
                                    # hch // 2 if i < 2 else 4 * self.N * self.N * 3,
                                    # kernel_size=3 if i < 2 else 1,
                                    # padding=1 if i < 2 else 0,
                                    # bias=False)
                                  # )
                                 # ))
            # hch = hch // 2 if i < 2 else 4 * self.N * self.N * 3
        
        # # Turn that list into a sequential and make sure it's properly registered.
        # self.hyperconv = nn.Sequential(*self.hyperconv)

        self.hyperconv = nn.Sequential(*[CL(WNC2D(1+self.c.size(1),7,kernel_size=3,padding=1,bias=False))]+        
                                      [CL(nn.Sequential(nn.LeakyReLU(0.02),WNC2D(self.c.size(1)+8+i*hG[0],hG[0],kernel_size=3,padding=1,bias=False))) for i in range(hD[0])]+
                                      [nn.Sequential(nn.LeakyReLU(0.02),WNC2D(self.c.size(1)+8+hD[0]*hG[0],(self.c.size(1)+8+hD[0]*hG[0])//2,kernel_size=3,padding=1,bias=False))]+
                                      [CL(nn.Sequential(nn.LeakyReLU(0.02),WNC2D((self.c.size(1)+8+hD[0]*hG[0])//2+i*hG[1],hG[1],kernel_size=3,padding=1,bias=False))) for i in range(hD[1])]+                                      
                                      [nn.Sequential(nn.LeakyReLU(0.02),WNC2D((self.c.size(1)+8+hD[0]*hG[0])//2+hD[1]*hG[1],((self.c.size(1)+8+hD[0]*hG[1])//2+hD[1]*hG[1])//2,kernel_size=3,padding=1,bias=False))]+
                                      [CL(nn.Sequential(nn.LeakyReLU(0.02),WNC2D(((self.c.size(1)+8+hD[0]*hG[0])//2+hD[1]*hG[1])//2+i*hG[2],hG[2],kernel_size=3,padding=1,bias=False))) for i in range(hD[2])]+
                                      [nn.Sequential(nn.LeakyReLU(0.02),WNC2D(((self.c.size(1)+8+hD[0]*hG[0])//2+hD[1]*hG[1])//2+hD[2]*hG[2],
                                      self.N * self.N * self.depth_compression,kernel_size=1,padding=0,bias=False))])
                                      
        # Initialize parameters with Orthogonal Initialization.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        # Define optimizer; note that I don't employ any weight decay
        # on the HyperNet weights because I initially wasn't normalizing it
        # but I believe one could turn weight decay back on and it wouldn't
        # change things. I also briefly experimented with using a different
        # learning rate for the HyperNet but no longer bother; the code
        # should just be a convenient template if you do want to use different
        # optim parameters for your hypernet.

        # By default I use ADAM for training SMASH, with settings borrowed from
        # DCGAN. I've found that ADAM works well for things that I don't have a
        # good set of optim hyperparams for, but tends to underperform well-
        # tuned SGD. Since part of the point of the project is keeping things as
        # turnkey as possible, I default to ADAM and provide workable params for
        # SGD with nesterov momentum.
        #
        # The high epsilon value is to support possible fp16 training; too low
        # of an eps and you'll be dividing by 0 in fp16.
        self.lr = 1e-3
        self.optim = optim.Adam([{'params': [param for param in self.parameters()
                                   if not any([param is hyperconvparam
                                   for hyperconvparam
                                   in self.hyperconv.parameters()])]},
                               {'params': self.hyperconv.parameters(),
                                  'lr': self.lr, 'weight_decay': 0}],
                                  lr=self.lr,
                                  betas=(0.5, 0.999),
                                  weight_decay=1e-4,
                                  eps=1e-4)
        # self.optim = optim.SGD([{'params':[param for param in self.parameters()
                                   # if not any([param is hyperconvparam
                                   # for hyperconvparam
                                   # in self.hyperconv.parameters()])]},
                               # {'params':self.hyperconv.parameters(),
                                  # 'lr':1e-1,'weight_decay':0}],
                                  # lr=1e-1,
                                  # momentum=0.9,
                                  # weight_decay=1e-4,
                                  # nesterov=True)
        
        # LR schedule, currently just filled with ITR to do cosine annealing
        self.lr_sched = {'itr':0}
        # iter counter
        self.j = 0
        
        # Min and max width: we sample between these two numbers to get the 
        # percentage of our overall allotted channel width at each block.
        # Min width as a percentage of total max block width
        self.min_width = 0.25
        # Max width as a percentage of total max block width
        self.max_width = 0.5
        
        # Maximum number of paths, to begin. This is a float percentage
        # that slowly anneals up to 1.0 and indicates how many paths we allow
        # at each block.
        self.max_paths = [0.0] * 3

        # block-wise parameter budget as a percent of total max block budget
        self.min_budget = 0.3
        self.max_budget = 0.5
        
        # Flag indicating whether we're in architectural SGD mode or not.
        self.arch_SGD = False
    
    # A simple function to update the LR with cosine annealing
    # Also updates the budget and max number of paths
    def update_lr(self,max_j):
        for param_group in self.optim.param_groups:
            param_group['lr'] = (0.5 * self.lr) * (1 + np.cos(np.pi * self.j / max_j))
        
        # Optionally anneal the width settings throughout training.
        
        # self.min_width = 0.25 + 0.25 * min(self.j / (max_j * 0.5), 1.0)
        # self.max_width = 0.50 + 0.50 * min(self.j / (max_j * 0.5), 1.0)
        
        # self.max_paths = [min(float(self.j) / (max_j * 0.5), 1.0)] * 3
        
        # self.min_budget = 0.25 + 0.25 * min(self.j / (max_j * 0.5), 1.0)
        self.max_budget = 0.50 + 0.50 * min(self.j / (max_j * 0.5), 1.0)
        
        # Anneal kernel sizes towards max kernel size
        self.max_kernel = 3 + int(((self.final_max_kernel - 3)//2) * min(self.j / (max_j * 0.5), 1.0) * 2) 
        
        self.j += 1
    

    def sample_budgets(self):
        # Parametric budget per block as a percentage of the baseline budget
        budgets = list(np.random.uniform(self.min_budget, self.max_budget, 3))
        
        # Size of memory banks in each block as a multiple of N
        # varies from 1 (each bank is N channels) to k (each bank is N_max chs)
        # Only choose bank sizes we can support with our in_channels!
        # That's what this conditional in the inner list comprehension does.
        bank_sizes = [int(np.random.choice( [item for item in range(1,self.k+1) 
                                            if not self.in_channels[i] // (item * self.N) < self.depth_compression * (item % self.depth_compression > 0) ])) 
                                            for i in range(3)]
        return budgets, bank_sizes
    
    # Function to sample an architecture
    # This method completely randoms
    def sample_architecture(self, budgets=None, bank_sizes=None):
    
        if budgets is None or bank_sizes is None:
            budgets, bank_sizes = self.sample_budgets()

        # Which banks we read from at each layer
        incoming=[[] for _ in range(3)]

        # Which banks we write to at each layer
        outgoing=[[] for _ in range(3)]

        # Number of units for each layer
        G=[[] for _ in range(3)]

        # Which convolutions are active within a layer.
        ops=[[] for _ in range(3)]  # definition of ops within a block
        
        # Whether to employ multiplicative gating at either in-layer junction.
        gate=[[] for _ in range(3)]
        
        # Filter dilation for each convolution within each layer
        dilation=[[] for _ in range(3)]
        
        # Activation function for each convolution within each layer.
        activation = [[] for _ in range(3)]
        
        # Filter sizes for each convolution within each layer.
        kernel_sizes = [[] for _ in range(3)]
        
        # Number of groups for each conv
        groups = [[] for _ in range(3)]

        # Vary the read budget to be 30-100% of the maximum budget,
        # then put it to the nearest number divisible by 3 for compatibility
        # with the layout of the embedding tensors (z and c), which compress
        # so that each slice on the trailing axis of z and c correspond to a set
        # of depth_compression memory banks in the output of the hypernet.
        
        # Budget is equivalent to number of slices allocated to this block;
        # since slices are currently produces as functions of N_max, this is
        # accounted for accordingly in the budget accounting.
        for i, budget in enumerate(
            [(block_budget * n  
            // (self.N * self.depth_compression)) for n, block_budget in zip(self.nch_list, budgets)]):

            # Initialize the number of input channels we've accumulated so far,
            # similar to computational budget
            used_budget=0

            # Accumulator indicating number of times we've written to a given
            # memory bank, initialized based on the input to this block.
            # This also allocates the maximum number of memory banks for a given block,
            # which in this case is going to allow us to act as if we only write to new
            # memory banks at each op.
            written_channels = [0] * min(int(self.in_channels[i] // (bank_sizes[i] * self.N) + np.ceil(budgets[i] * self.depth * self.k) // bank_sizes[i]), self.widths[i] // (bank_sizes[i] * self.N))
            
            # Commented out print statement for debugging
            # print('Number of banks is ' + str(len(written_channels))+', widths is' + str(self.widths[i]) + 'bank size is ' +str(bank_sizes[i] * self.N))
            
            # Initialize write accumulator
            for index in range(self.in_channels[i] // (bank_sizes[i] * self.N)):
                written_channels[index] += 1

            # Define the minimum budget per op and the space of all possible
            # read-write choices within a block based on N and this layer's
            # bank size.
            min_budget_per_op = bank_sizes[i] if bank_sizes[i] % self.depth_compression else bank_sizes[i] // self.depth_compression
            min_reads_per_op = min_budget_per_op * self.depth_compression // bank_sizes[i]
            max_reads_per_op = len(written_channels) // min_reads_per_op * min_reads_per_op # round to nearest multiple of min_reads
            num_input_choices = list(range(min_reads_per_op, max_reads_per_op + min_reads_per_op, min_reads_per_op)) #then add min_reads
            # num_input_choices = [index for index in range(1, len(written_channels) + 1) 
                            # if not (index * bank_sizes[i]) % (self.depth_compression)] 
            # print(len(written_channels), max_reads_per_op, bank_sizes[i],budgets[i], min_budget_per_op, num_input_choices)
                            
                            # (index * self.depth_compression * self.k) % bank_sizes[i]]
            # min_budget_per_op = num_input_choices[0] * 
            # print(num_input_choices,min_reads_per_op)
            while (budget - used_budget) >= min_budget_per_op: # this conditional also needs to stop not just at a less than but if we're near a multiple of 4 basically

                # Get all possible channels we can read from
                readable_channels = list(
                                     range(
                                      sum([item > 0\
                                        for item in written_channels])))

                # Max budgeted inputs is based on remaining budgeted slices, each of which gives depth_compression Ns, divided by N per bank
                max_budgeted_inputs = (budget - used_budget) * self.depth_compression  // bank_sizes[i]

                # max input banks has to be one of the allowable num_input values
                max_input_banks = max([path_choice for path_choice in num_input_choices if path_choice <= max_budgeted_inputs])
                
                # Select number of input paths               
                num_input_paths = min(
                                      int(np.random.choice([path_choice for path_choice in num_input_choices if path_choice <= len(readable_channels)])),
                                      max_input_banks)
                # Select read locations
                incoming[i].append(
                  sorted(
                    np.random.choice(
                      readable_channels,
                      num_input_paths,
                      replace=False)
                      )
                    )
                
                # Determine #of filters for this layer.
                # This is given as a multiple of N.
                G_choices = range(bank_sizes[i], self.k + 1, bank_sizes[i])
                
                # Most probable #filters based on our inputs
                most_probable_G = num_input_paths * bank_sizes[i]
                G_probabilities = [1./ (1e-2 + 10 * np.abs(g_choice - most_probable_G)) for g_choice in G_choices]
                
                # Normalize the probabilities.
                G_probabilities = [g_prob / sum(G_probabilities) for g_prob in G_probabilities]
                
                # Select number of filters.
                G[i].append(int(np.random.choice(G_choices,p=G_probabilities)))
                
                # Upate the budget                
                # The commented line here is for scaling the budget based on 
                # the number of output units, which won't accurately hold
                # the parametric budget but is more in line with a compute
                # budget.
                # int(np.ceil(number_of_inputs*float(G[i][-1]/self.N_max)))
                used_budget += num_input_paths * bank_sizes[i] // self.depth_compression

                # now, select outgoing channels
                # Channels we haven't written to yet
                empty_channels=[index for index, item
                                in enumerate(written_channels) if item == 0]
            
                # Select number of outputs to write to, writing to at least
                # as many locations as we have units for, and giving a light
                # preference towards writing to fewer (rather than more) 
                # banks.
                
                # This probability call is complicated and it was 4AM when I
                # wrote it but I sketched the curve somewhere and it basically 
                # just says that we should assign an exponentially higher
                # probability towards having fewer output paths, rather than
                # many.
                #
                # I've tried changing this or removing it and just sampling 
                # uniformly once or twice but things didn't work as well when I
                # did, and while that was always coupled with other changes,
                # this just seems to work, so I'm just going to trust 4AM Andy.
                
                probability = np.exp(
                                np.asarray(
                                    range(
                                        max(len(readable_channels) // 2, G[i][-1] // bank_sizes[i]), 
                                        G[i][-1] // bank_sizes[i] - 1, 
                                        -1))) 
                probability = [p / sum(probability) for p in probability]

                
                # Select how many outputs we're going to have based on the
                # probability defined above. Allow at G//bank_sizes writes.
                number_of_outputs = np.random.choice(
                                     list(range(G[i][-1] // bank_sizes[i], 
                                           1 + max(len(readable_channels) // 2, G[i][-1] // bank_sizes[i])
                                           )), 
                                      p=probability)

                # Select which channels we're writing to
                outgoing_channels = list(
                                      sorted(
                                        np.random.choice(
                                          readable_channels 
                                          + empty_channels[:G[i][-1] // bank_sizes[i]],
                                          number_of_outputs, replace=False)))
                                    
                # Make sure we only write sequentially to new empty channels, 
                # and don't skip any.
                num_empty_writes = len([o for o in outgoing_channels if o in empty_channels])
                outgoing_channels = ([o for o in outgoing_channels if o not in empty_channels]
                                    + empty_channels[:num_empty_writes])
                
                # Update output list and update which channels we've written to
                outgoing[i].append(outgoing_channels)
                
                # Commented out debugging print
                # print(i,used_budget,len(readable_channels), outgoing_channels,len(written_channels))
                
                # Update write accumulator
                for o in outgoing_channels:
                    written_channels[o] += 1

                # Possible op configurations; 
                # Note that we don't bother to have the option to have w[2] 
                # alone, since although in the SMASH network that would 
                # be different, the resulting network would not be different
                # (i.e. it would just be two ways to define a single conv)
                ops[i].append(
                    self.options[
                        int(np.random.choice(len(self.options), 
                                             p=self.options_probabilities))])
                
                # Decide if we're going to have a multiplicative tanh-sig gate
                # at either of the two parallel layers of the op.
                # Randomly sample activation functions;
                # note that this will be overriden in the main net if
                # a relevant gate is active, and is accordingly also
                # ignored where appropriate in the definition of self.c
                if self.var_nl:
                    activation[i].append([np.random.choice(
                                            list(
                                              self.nl_dict.keys())) 
                                          for _ in range(4)])
                else:
                    activation[i].append([0]*4)                  
                
                # If we're using gates and g//2 is divisible by bank size,
                # then roll for gates
                # If we're using preactivation, then only allow one add-split-mult gate,
                # else our channel count will be messy.
                if self.gates and (G[i][-1]//2 > 0 ) and not (G[i][-1]//2) % bank_sizes[i]:
                    gt = np.random.uniform() < 0.25 if ops[i][-1][0] and ops[i][-1][2] else 0
                    gt = [gt, np.random.uniform() < 0.25  if ops[i][-1][1] and ops[i][-1][3] and not gt else 0]
                    
                    gate[i].append(gt)
                    
                    # If not using preactivation, pass tanh and sigmoid NLs
                    if not self.preactivation:
                        if gate[i][0]:
                            activation[i][-1][0] = 1
                            activation[i][-1][2] = 2
                        if gate[i][1]:
                            activation[i][-1][1] = 1
                            activation[i][-1][3] = 2
                else:
                    gate[i].append([0,0])

                kernel_sizes[i].append([list(np.random.choice(range(3,self.max_kernel+2,2),2)) for _ in range(4)])
                
                # Randomly sample dilation factors for each conv,
                # limiting the upper dilation based on the kernel size.
                dilation[i].append([ [int(np.random.randint(1, 5-(kernel_sizes[i][-1][j][0]-1)//2)),
                                      int(np.random.randint(1, 5-(kernel_sizes[i][-1][j][1]-1)//2))]
                                   for j in range(4)])
                
                # Allow the number of groups to be up to the largest N factor.
                if self.var_group:
                    groups[i].append([np.random.choice(self.factors) for _ in range(4)])
                else:
                    groups[i].append([1]*4)
                

        return incoming, outgoing, G, ops, gate, dilation, activation, bank_sizes, kernel_sizes, groups

    # Sample Weights
    # This function takes in an architecture definition, constructs
    # the architectural-conditional vector, then generates the weights for the 
    # 1x1 convs of the network.
    def sample_weights(self, incoming, outgoing, G,
                       ops, gate, dilation, activation,
                       bank_sizes, kernel_sizes, groups, z=None, c=None):
        
        # Sample the random vector, z
        if z is None:
            z = self.z
            z.normal_(0, 1)
              
        if c is None:
            c = self.c

        # Zero the architectural-conditional vector
        c.fill_(-1)
        
        # This counter indicates the index of the trailing dimension in the 
        # embedding tensor to which we are currently writing.
        slice_index = 0
        
        # j is a rolling counter that tells us when we need to increment n. 
        # Since we compress the 1x1 convs there's some overlap (some locations
        # in self.c correspond to different 1x1 weights) and this counter
        # just increments in accordance with the weight compression scheme.
        j = 0

        # Build class_conditional vector
        # Loop across blocks
        for block_index, (incoming_channels, outgoing_channels,g_values,
            op_values, gate_values, dilation_values, nl_values, 
            bs, kernel_values, group_values) in enumerate(zip(
                incoming, outgoing, G, 
                ops, gate, dilation, activation,
                bank_sizes, kernel_sizes, groups)):
    
            # Loop across ops within a block.
            for (read, write, g,
                op, gated, dilate, nls,
                ks, group) in zip(incoming_channels, 
                                              outgoing_channels, 
                                              g_values,
                                              op_values,
                                              gate_values,
                                              dilation_values,
                                              nl_values,
                                              kernel_values,
                                              group_values):

                sub_index = 0
                slice_N = self.depth_compression # How many slices we have
                for i, r in enumerate(read):
                    
                    # A counter telling us where in self.c we are.
                    channel_counter = 0
                    
                    slice_start = slice_index + sub_index // self.depth_compression
                    slice_end = slice_index + (sub_index + bs) // self.depth_compression + (sub_index + bs) % self.depth_compression

                    c[:, r, :g, slice_start : slice_end] = 1
                    
                    channel_counter += self.max_banks

                    # Write write-conditional entries
                    for w in write:
                        c[:, channel_counter + w, :g, slice_start : slice_end] = 1
                    channel_counter += self.max_banks
                    
                    # Block conditional entry, tell the net which block we're in
                    c[:, channel_counter + block_index, :g, slice_start : slice_end] = 1
                    channel_counter += 3 # increment by number of blocks
                    
                    # G-conditional entry, can't be zero so the zero index corresponds to G=1
                    c[:, channel_counter + g - 1, :g, slice_start : slice_end] = 1
                    channel_counter += self.k
                    
                    # If using the 2x2 op config
                    if self.var_op or self.big_op or self.long_op:
                        # Write dilation-conditional entries
                        for di, d in enumerate(dilate):
                            # only denote the dilate if the op is active
                            c[:, -1 + d[0] + channel_counter, :g, slice_start : slice_end] = 1 if op[di] else -1 
                            channel_counter += self.max_dilate
                            
                            c[:, -1 + d[1] + channel_counter, :g, slice_start : slice_end] = 1 if op[di] else -1
                            channel_counter += self.max_dilate
                            
                        if self.var_ks:
                            for ki, k in enumerate(ks):
                                c[:,(-1 + k[0] // 2) + channel_counter, :g, slice_start : slice_end] = 1 if op[ki] else -1
                                channel_counter += self.max_kernel // 2
                                
                                c[:,(-1 + k[1] // 2) + channel_counter, :g, slice_start : slice_end] = 1 if op[ki] else -1
                                channel_counter += self.max_kernel // 2
                        # Write op-conditional entries (if a conv is active)
                        if self.var_op:
                            for o in list(np.where(np.asarray(op) > 0)[0]):
                                c[:, channel_counter + o, :g, slice_start : slice_end] = -1
                            channel_counter += 4
                        
                        if self.gates:
                            # Write gate-conditional entries
                            for gi, gt in enumerate(gated):
                                c[:, channel_counter + gi, :g, slice_start : slice_end] = gt
                            channel_counter += 2
                        
                        if self.var_nl:
                            # Write activation-conditional entries
                            for nli, nl in enumerate(nls):
                                c[:, channel_counter + nl, :g, slice_start : slice_end] = 1 if (op[nli] and not gated[nli//2]) else -1
                                channel_counter += len(self.nl_dict)
                        
                        # Group-conditional entries
                        if self.var_group:
                            for grp_i, grp in enumerate(group):
                            # only denote the dilate if the op is active
                                c[:, -1 + self.factors.index(grp) + channel_counter, :g, slice_start : slice_end] = 1 if op[grp_i] else -1
                                channel_counter += len(self.factors)
                    # If just using a single 3x3 conv
                    else:
                        # Write dilation-conditional entries
                        c[:, -1 + dilate[0][0] + channel_counter, :g, slice_start : slice_end] = 1
                        channel_counter += self.max_dilate
                        
                        c[:, -1 +  dilate[0][1] + channel_counter, :g, slice_start : slice_end] = 1
                        channel_counter += self.max_dilate
                        
                        if self.var_ks:
                            # Write kernel size-conditional entries
                            c[:,(-1 + ks[0][0] // 2) + channel_counter, :g, slice_start : slice_end] = 1
                            channel_counter += self.max_kernel // 2
                            
                            c[:,(-1 + ks[0][1] // 2) + channel_counter, :g, slice_start : slice_end] = 1
                            channel_counter += self.max_kernel // 2
                            
                        # Write activation-conditional entries    
                        if self.var_nl:
                            c[:, 2 * self.max_dilate + (2 * self.max_banks + 3) +nls[0], :g, slice_start : slice_end] = 1
                            channel_counter += len(self.nl_dict)
                            
                        if self.var_group:
                            # only denote the dilate if the op is active
                            c[:, -1 + self.factors.index(group[0]) + channel_counter, :g, slice_start : slice_end] = 1  
                            channel_counter += len(self.factors)
                        
                    sub_index += bs
                    
                    # Increment n every time we need to move to a new slice
                    # of the trailing dimension of self.c
                # Increment slice_index based on how many banks we just went through?
                slice_index += bs * len(read) // self.depth_compression
                # if not (j + 1) % (self.depth_compression):
                        # n += 1

        # Variable containing c. We keep this in a separate variable 
        # for use with architectural SGD.
        self.vc = V(c[:, :, :, :slice_index + 1],
                                    volatile=not self.training and not self.arch_SGD, requires_grad=self.arch_SGD)
        
        embedding = torch.cat( (V(z[:, :, :, :slice_index + 1],
                                   volatile=not self.training and not self.arch_SGD),
                                self.vc), 1)
        
        # This is a complex slice that basically is designed to "stack up" bits of the array--
        # normally a resize() or view() "rolls" numbers so that things on the row above end up on the row below,
        # but in this case we want to treat them like legos where stacks along particular axes stay together.
        # this is an efficient way to do that; I'm considering writing a more detailed tutorial on how this works,
        # and the reason I do it this way. Naive resizes are bad!
        return self.hyperconv(embedding).squeeze(0)\
                                        .transpose(0,1)\
                                        .contiguous()\
                                        .view(self.N_max, int((slice_index + 1) * self.depth_compression* self.N))\
                                        .index_select(1, V(torch.LongTensor([i for item in  
                                                           [range(q, (slice_index + 1) * self.depth_compression * self.N, slice_index + 1) for q in range(slice_index + 1)] for i in item]).cuda()))\
                                        .transpose(0,1)\
                                        .unsqueeze(2).unsqueeze(3)\
                                        .contiguous()
       
        # return self.hyperconv(embedding).resize(self.bottleneck * self.N_max, (n + 1) * self.depth_compression * self.N).unsqueeze(2).unsqueeze(3) # Naive resize!

    
    
    # Forward method
    # This method supports randomly sampling architectures and weights at
    # each training step, or using a fixed architecture (fed through the kwargs)
    def forward(self, x, w1x1=None, incoming=None, outgoing=None, 
               G=None, ops=None, gate=None, dilation=None, activation=None,
               bank_sizes=None, kernel_sizes=None, groups=None):

 
        # Sample architecture
        if any([item is None for item in [incoming, outgoing, G, ops,
                                          gate, dilation, activation,
                                          bank_sizes, kernel_sizes, groups]]):
            (incoming, outgoing, G, 
             ops, gate, dilation, activation,
             bank_sizes, kernel_sizes, groups) = self.sample_architecture()
        
        # Sample weights
        if w1x1 is None:
            w1x1 = self.sample_weights(incoming, outgoing, G, 
                                     ops, gate, dilation, activation,
                                     bank_sizes, kernel_sizes, groups)

        # Get stem convolution
        out = F.relu(self.conv1(x))
        
        # Allocate memory based on the maximum index of the bank we write to
        m = [[None 
              for _ in range(max(max([max(item) for item in outgo])+ 1, inch // (bank_size * self.N)))]
           for hw, outgo, inch, bank_size in zip((32, 16, 8), outgoing, self.in_channels, bank_sizes)]
        
        # Counter for which slice of the w1x1 trailing dimension we're on
        n = 0
        

        for i, (incoming_channels, outgoing_channels,  
                g_values, op_values, gate_values, 
                dilation_values, nl_values, bs, 
                kernel_values, group_values, trans) in enumerate(
            zip(incoming, outgoing, G, ops, gate, 
                dilation, activation, bank_sizes, 
                kernel_sizes, groups, [self.trans1, self.trans2, None])):

            # Write block input to memory banks
            # We always overwrite since the input should always be initial Nones in m banks.
            for j in range(out.size(1) // (bs * self.N) ):
                m[i][j] = out[:, j * bs * self.N : (j + 1) * bs * self.N]

            for read, write, g, op, gated, dilate, nls, ks, group in zip(
                    incoming_channels, outgoing_channels, 
                    g_values, op_values, gate_values,
                    dilation_values, nl_values, kernel_values, group_values):

                # Input to this layer
                inp = torch.cat([m[i][index]
                              for index in read], 1)
                
                # Number of channels in the input to tihs layer
                nch = inp.size(1)

                # Number of output units of the 1x1 is:
                # nch while nch < n_out of this op
                # nearest multiple of n_out while nch < bottleneck*n_out
                # At most either bottleneck*N_out or N_max, whichever is smaller.
                n_bottleneck = min(min(min(max(nch // (g * self.N), 1), self.bottleneck) * g * self.N, nch), self.max_bottleneck * self.N_max)
                out = F.relu(
                     F.conv2d(input=inp,
                              weight=wn2d(w1x1[n : n + nch, :n_bottleneck].transpose(0,1)),
                              padding=0,
                              bias=None))
                
                # Increment channel counter
                n += nch

                # Apply main convolutions
                out = self.W[i](out, g * self.N, op, gated, dilate, [self.nl_dict[nl] for nl in nls], ks, group) 

                # Write to output blocks
                # Note that the modulo calls are there so that if we write
                # to more banks than we have channels for we cycle through 
                # and start writing from the first out channel again.
                
                # If we're using add-split gates, then our actual g may be divided by 2?
                # Need to ensure, when producing the architecture, that this div-by-2 is feasible.
                if self.preactivation and any(gated):
                    g = g//2

                for j, w in enumerate(write):
                    
                    # Allocate dat memory if it's None
                    if m[i][w] is None:
                        m[i][w] = out[:, (j % (g // bs)) * (bs * self.N) : (j % (g // bs) + 1) * (bs * self.N)]
                    
                    # Else, if already written, add to it. 
                    else:
                        m[i][w] = m[i][w] + out[:, (j % (g // bs)) * (bs * self.N) : (j % (g // bs) + 1) * (bs * self.N)]

            # After all the ops of a block, grab all of the memory
            # This call could also include a list comprehension to dump
            # any leftover Nones in m, but since we SHOULD be allocating only
            # the exact correct number of m's, doing it this way
            # acts as a secondary check.
            # Having it this way actually helped me spot some errors!
            out = torch.cat(m[i], 1)
            
            # Then, if we're not at the last block, downsample
            # print(self.in_channels, len(m[i]),out.size(), trans)
            if trans is not None:
                out = trans(out, self.in_channels[i + 1])


        # Finally, feed the full memory bank of the last block to the fc layer
        # Batchorm it, globally average pool it, fc it, log-softmax it.
        out = F.relu(self.bn1(out))                
        out = torch.squeeze(F.avg_pool2d(out, out.size(2)))
        out = F.log_softmax(self.fc(out))
        return out

# Class for main nets with a SMASH-ranked architecture
# Note that we currently assume preactivation, don't quite support gates properly,
# and do not properly support variable nonlinearities.
class MainNet(nn.Module):
    def __init__(self, arch, nClasses=100, var_ks=True,
                 var_op=False, big_op=False,
                 op_bn=False):
        super(MainNet, self).__init__()
        
        # Bind architecture to net
        self.arch = arch
        
        # Get architecture details
        (self.incoming, self.outgoing, self.G, 
         self.ops, self.gate, self.dilation, 
         self.activation, self.bank_sizes, 
         self.kernel_sizes, self.groups, 
         self.N, self.N_max, self.bottleneck, self.max_bottleneck,
         self.in_channels, self.SMASH_STD, self.SMASH_ERROR) = self.arch
        
        # Initialize Op List
        self.mod = nn.ModuleList()
        
        # Define ops 
        # loop across blocks
        for block_index, (incoming_channels, outgoing_channels,g_values,
            op_values, gate_values, dilation_values, nl_values, 
            bs, kernel_values, group_values) in enumerate(zip(
                self.incoming, self.outgoing, self.G, 
                self.ops, self.gate, self.dilation, self.activation,
                self.bank_sizes, self.kernel_sizes, self.groups)):
            
            # Loop across ops within a block
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
 
                
                # Determine layer sizes within an op
                n_in = int(len(read) * bs * self.N)
                n_bottle = min(min(min(max(n_in // (g * self.N), 1), self.bottleneck) * g * self.N, n_in), self.max_bottleneck * self.N_max)
                n_out = int(g)  * self.N
                
                self.mod.append(Layer(n_in=n_in, 
                                      n_bottle=n_bottle,
                                      n_out=n_out,
                                      ops=op if var_op else [1]*4 if big_op else [1,0,0,0], 
                                      gate=gated,
                                      dilation=dilate,
                                      kernel_size=ks if var_ks else [[3]*2]*4,
                                      groups=group,
                                      norm_style='full' if op_bn else 'sandwich'))
  
        # Stem Convolution
        self.conv1 = nn.Conv2d(3, self.in_channels[0], kernel_size=7, padding=3,
                               bias=False, stride=1)
                               

        # Get maximum #channels per block
        self.D = [max(max([max(item) for item in outgo])+ 1, inch // (bank_size * self.N))
           for hw, outgo, inch, bank_size in zip((32, 16, 8), self.outgoing, self.in_channels, self.bank_sizes)]
           
       # Transition convolutions
        self.trans1 = Transition(int(self.D[0] * self.N * self.bank_sizes[0]), int(self.in_channels[1]))
        self.trans2 = Transition(int(self.D[1] * self.N * self.bank_sizes[1]), int(self.in_channels[2]))
        
        # Output layer
        self.bn1 = nn.BatchNorm2d(int(self.D[2] * self.N * self.bank_sizes[2]))
        self.fc = nn.Linear(int(self.D[2] * self.N * self.bank_sizes[2]), nClasses)
        
        
        # Initialize modules
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()    

        self.lr = 1e-1
        self.optim = optim.SGD(params=self.parameters(),lr=self.lr,
                               nesterov=True,momentum=0.9, 
                               weight_decay=1e-4)
        # LR schedule, currently just filled with ITR to do cosine annealing
        self.lr_sched = {'itr':0}
        # iter counter
        self.j = 0
    
    # A simple function to update the LR with cosine annealing
    def update_lr(self,max_j):
        for param_group in self.optim.param_groups:
            param_group['lr'] = (0.5 * self.lr) * (1 + np.cos(np.pi * self.j / max_j))
        self.j+=1
    
    
    
    # Optionally initialize using the output of the SMASH network.
    # Currently deprecated, may or may not work.
    # TO DO: modify this to allow for variable kernel sizes and groups.
    def init_from_SMASH(self, w1x1, W, conv1, trans1, trans2, bn1, fc):

        # Slice index for w1x1
        n = 0 
        mod_index = 0
        
        for i, inc in enumerate(self.incoming):
            for j in inc:
                
                this_w = w1x1.data[n : n + self.mod[mod_index].conv1.weight.size(1),:self.mod[mod_index].conv1.weight.size(0)].transpose(0,1)
                
                self.mod[mod_index].conv1.weight.data = this_w / float(torch.norm(this_w))
                                                        
                n += self.mod[mod_index].conv1.weight.size(1)
                
                for k, op in enumerate(self.mod[mod_index].op):
                    if type(op) is nn.Sequential:
                        self.mod[mod_index].op[k][1].weight.data = W[k][i].weight.data[:self.mod[mod_index].op[k][1].weight.size(0),
                                                                                    :self.mod[mod_index].op[k][1].weight.size(1)].contiguous()
                mod_index +=1
        
        self.conv1.weight.data = conv1.weight.data
        # Drop in the trans and batchnorm and fc weights
        self.trans1.bn1.weight.data = trans1.bn1.weight.data[:self.trans1.bn1.weight.data.size(0)]
        self.trans1.bn1.bias.data = trans1.bn1.bias.data[:self.trans1.bn1.bias.data.size(0)]
        self.trans1.bn1.running_mean = trans1.bn1.running_mean[:self.trans1.bn1.running_mean.size(0)]
        self.trans1.bn1.running_var = trans1.bn1.running_var[:self.trans1.bn1.running_var.size(0)]
        self.trans1.conv1.weight.data = trans1.conv1.weight.data[:self.trans1.conv1.weight.data.size(0),:self.trans1.conv1.weight.data.size(1)].contiguous()
        
        self.trans2.bn1.weight.data = trans2.bn1.weight.data[:self.trans2.bn1.weight.data.size(0)]
        self.trans2.bn1.bias.data = trans2.bn1.bias.data[:self.trans2.bn1.bias.data.size(0)]
        self.trans2.bn1.running_mean = trans2.bn1.running_mean[:self.trans2.bn1.running_mean.size(0)]
        self.trans2.bn1.running_var = trans2.bn1.running_var[:self.trans2.bn1.running_var.size(0)]
        self.trans2.conv1.weight.data = trans2.conv1.weight.data[:self.trans2.conv1.weight.data.size(0), :self.trans2.conv1.weight.data.size(1)].contiguous()
        
        self.bn1.weight.data = bn1.weight.data[:self.bn1.weight.data.size(0)]
        self.bn1.bias.data = bn1.bias.data[:self.bn1.bias.data.size(0)]
        self.bn1.running_mean = bn1.running_mean[:self.bn1.running_mean.size(0)]
        self.bn1.running_var = bn1.running_var[:self.bn1.running_var.size(0)]
        
        self.fc.weight.data = fc.weight.data[:self.fc.weight.data.size(0),:self.fc.weight.data.size(1)].contiguous()
        self.fc.bias.data = fc.bias.data[:self.fc.bias.data.size(0)].contiguous()
                    
        
    
    def forward(self,x):

        # Stem convolution
        out = self.conv1(x)
        
        # Allocate memory banks

        m = [[None for _ in range(d)] for d in self.D]
        module_index = 0
        for i,(incoming_channels,outgoing_channels,g_values, bs, trans) in enumerate(zip(
                self.incoming,self.outgoing, self.G, self.bank_sizes, [self.trans1,self.trans2,None])):
            
            # Write to initial memory banks
            for j in range(out.size(1) // (bs * self.N) ):
                m[i][j] = out[:, j * bs * self.N : (j + 1) * bs * self.N]
            
            for read,write,g in zip(incoming_channels,outgoing_channels,g_values):
                # Cat read tensors
                inp = torch.cat([m[i][index] for index in read], 1)
                
                # Apply module and increment op index
                out = self.mod[module_index](inp)
                module_index += 1
                
                for j, w in enumerate(write):
                    # Allocate dat memory if it's None
                    if m[i][w] is None:
                        m[i][w] = out[:, (j % (g // bs)) * (bs * self.N) : (j % (g // bs) + 1) * (bs * self.N)]
                    # Else, if already written, add to it. 
                    else:
                        m[i][w] = m[i][w] + out[:, (j % (g // bs)) * (bs * self.N) : (j % (g // bs) + 1) * (bs * self.N)]
    
    
            if trans is not None:
                out = trans(torch.cat(m[i], 1))
            else:
                out = torch.cat(m[i], 1)
    
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), out.size(2)))
        out = F.log_softmax(self.fc(out))
        return out
 