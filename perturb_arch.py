import numpy as np
import torch

# A simple function to perturb the architecture.
def perturb_architecture(net, arch, perturb_probability=0.05):
        
    (incoming, outgoing, G, 
         ops, gate, dilation, activation,
         bank_sizes, kernel_sizes, groups) = arch

    # We still keep track of the budget to make sure we don't overflow net.c
    for i, budget in enumerate(
        [(n // (net.N * net.depth_compression)) for n in net.nch_list]):

        # Initialize the number of input channels we've accumulated so far,
        # similar to computational budget
        used_budget=0

        # Still keep track of written channels to prevent reading from 
        # emtpy channels
        written_channels = [0] * min(int(net.in_channels[i] // (bank_sizes[i] * net.N) + np.ceil(net.depth * net.k) // bank_sizes[i]), net.widths[i] // (bank_sizes[i] * net.N))
        for index in range(net.in_channels[i] // (bank_sizes[i] * net.N)):
            written_channels[index] += 1

        min_budget_per_op = bank_sizes[i] if bank_sizes[i] % net.depth_compression else bank_sizes[i] // net.depth_compression
        min_reads_per_op = min_budget_per_op * net.depth_compression // bank_sizes[i]
        max_reads_per_op = len(written_channels) // min_reads_per_op * min_reads_per_op # round to nearest multiple of min_reads
        num_input_choices = list(range(min_reads_per_op, max_reads_per_op + min_reads_per_op, min_reads_per_op)) #then add min_reads
        
        op_index = 0
        ''' this has some failure cases since we don't currently delete
            later ops if we use up all the budget, but it should be fine.'''
        while (budget - used_budget) >= min_budget_per_op and op_index < len(G[i]): # this conditional also needs to stop not just at a less than but if we're near a multiple of 4 basically

            # Get all possible channels we can read from
            readable_channels = list(
                                 range(
                                  sum([item > 0\
                                    for item in written_channels])))
            
            empty_channels=[index for index, item
                                in enumerate(written_channels) if item == 0]
            # Max budgeted inputs is based on remaining budgeted slices, each of which gives depth_compression Ns, divided by N per bank
            max_budgeted_inputs = (budget - used_budget) * net.depth_compression  // bank_sizes[i]

            # max input banks has to be one of the allowable num_input values
            max_input_banks = max([path_choice for path_choice in num_input_choices if path_choice <= max_budgeted_inputs])
            
            # Don't think we don't really need to preference our incoming channels, 

            '''consider making this constant too? or allowing us to either randomize this or keep it the same'''
            if np.random.uniform() < perturb_probability or any([read in empty_channels for read in incoming[i][op_index]]):
                num_input_paths = min(
                                      int(np.random.choice([path_choice for path_choice in num_input_choices if path_choice <= len(readable_channels)])),
                                      max_input_banks)
                # Select read locations
                incoming[i][op_index] = sorted(np.random.choice(readable_channels, num_input_paths, replace=False))
            else:
                num_input_paths = len(incoming[i][op_index])
            
            
            if np.random.uniform() < perturb_probability:
                # Determine #of filters for this layer.
                # Presume it's an even multiple of bank sizes?
                G_choices = range(bank_sizes[i], net.k + 1, bank_sizes[i])
                # Most probable #filters based on our inputs
                most_probable_G = num_input_paths * bank_sizes[i]
                G_probabilities = [1./ (1e-2 + 10 * np.abs(g_choice - most_probable_G)) for g_choice in G_choices]
                # Normalize the probabilities.
                G_probabilities = [g_prob / sum(G_probabilities) for g_prob in G_probabilities]
                # Select number of filters.
                G[i][op_index] = int(np.random.choice(G_choices,p=G_probabilities))
            
            # Upate the budget                
            # The commented line here is for scaling the budget based on 
            # the number of output units, which won't accurately hold
            # the parametric budget but is more in line with a compute
            # budget.
            # int(np.ceil(number_of_inputs*float(G[i][-1]/net.N_max)))
            used_budget += num_input_paths * bank_sizes[i] // net.depth_compression
            
            if np.random.uniform() < perturb_probability:
                # now, select outgoing channels
                # Channels we haven't written to yet
                
            
       
                probability = np.exp(
                                np.asarray(
                                    range(
                                        max(len(readable_channels) // 2, G[i][op_index] // bank_sizes[i]), 
                                        G[i][op_index] // bank_sizes[i] - 1, 
                                        -1))) 
                probability = [p / sum(probability) for p in probability]

                
                # Select how many outputs we're going to have based on the
                # probability defined above. Allow at G//bank_sizes writes.
                number_of_outputs = np.random.choice(
                                     list(range(G[i][op_index] // bank_sizes[i], 
                                           1 + max(len(readable_channels) // 2, G[i][op_index] // bank_sizes[i])
                                           )), 
                                      p=probability)

                # Select which channels we're writing to
                outgoing_channels = list(
                                      sorted(
                                        np.random.choice(
                                          readable_channels 
                                          + empty_channels[:G[i][op_index] // bank_sizes[i]],
                                          number_of_outputs, replace=False)))
                                    
                # Make sure we only write sequentially to new empty channels, 
                # and don't skip any.
                num_empty_writes = len([o for o in outgoing_channels if o in empty_channels])
                outgoing_channels = ([o for o in outgoing_channels if o not in empty_channels]
                                    + empty_channels[:num_empty_writes])
                # Update output list and update which channels we've written to
                outgoing[i][op_index] = outgoing_channels
            # If we are writing to empty channels, ensure we're writing to the nearest empty channel.
            elif any([o in empty_channels for o in outgoing[i][op_index]]):
                num_empty_writes = len([o for o in outgoing[i][op_index] if o in empty_channels])
                outgoing_channels = ([o for o in outgoing[i][op_index] if o not in empty_channels]
                                    + empty_channels[:num_empty_writes])
                outgoing[i][op_index] = outgoing_channels
            else:
                outgoing_channels = outgoing[i][op_index]
            # print(i,used_budget,len(readable_channels), outgoing_channels,len(written_channels))
            for o in outgoing_channels:
                written_channels[o] += 1

            # Possible op configurations; 
            # Note that we don't bother to have the option to have w[2] 
            # alone, since although in the SMASH network that would 
            # be different, the resulting network would not be different
            # (i.e. it would just be two ways to define a single conv)
            if np.random.uniform() < perturb_probability:
                ops[i][op_index] = net.options[int(np.random.choice(len(net.options), p=net.options_probabilities))]
            
            # Decide if we're going to have a multiplicative tanh-sig gate
            # at either of the two parallel layers of the op.
            # Randomly sample activation functions;
            # note that this will be overriden in the main net if
            # a relevant gate is active, and is accordingly also
            # ignored where appropriate in the definition of net.c
            if np.random.uniform() < perturb_probability:
                if net.var_nl:
                    activation[i][op_index] = [np.random.choice(
                                            list(
                                              net.nl_dict.keys())) 
                                          for _ in range(4)]
                else:
                    activation[i][op_index] = [0]*4                  
            
            # If we're using gates and g//2 is divisible by bank size,
            # then roll for gates
            # If we're using preactivation, then only allow one add-split-mult gate,
            # else our channel count will be messy.
            if np.random.uniform() < perturb_probability:
                if net.gates and (G[i][op_index]//2 > 0 ) and not (G[i][op_index]//2) % bank_sizes[i]:
                    gt = np.random.uniform() < 0.25 if ops[i][op_index][0] and ops[i][op_index][2] else 0
                    gt = [gt, np.random.uniform() < 0.25  if ops[i][op_index][1] and ops[i][op_index][3] and not gt else 0]
                    
                    gate[i][op_index] = gt
                    # If not using preactivation, pass tanh and sigmoid NLs
                    if not net.preactivation:
                        if gate[i][0]:
                            activation[i][op_index][0] = 1
                            activation[i][op_index][2] = 2
                        if gate[i][1]:
                            activation[i][op_index][1] = 1
                            activation[i][op_index][3] = 2
                else:
                    gate[i][op_index] = [0,0]
            if np.random.uniform() < perturb_probability:
                kernel_sizes[i][op_index] = [list(np.random.choice(range(3,net.max_kernel+2,2),2)) for _ in range(4)]
            
            # Randomly sample dilation factors for each conv,
            # limiting the upper dilation based on the kernel size.
            if np.random.uniform() < perturb_probability:
                dilation[i][op_index] = [ [int(np.random.randint(1, 5-(kernel_sizes[i][op_index][j][0]-1)//2)),
                                      int(np.random.randint(1, 5-(kernel_sizes[i][op_index][j][1]-1)//2))]
                                   for j in range(4)]
            
            # Allow the number of groups to be up to the third-largest factor
            # of G, so for G=64, with factors of [1,2,4,8,16,32,64]
            # this would allow for 16 groups.
            if np.random.uniform() < perturb_probability:
                if net.var_group:
                    groups[i][op_index] = [np.random.choice(net.factors) for _ in range(4)]
                else:
                    groups[i][op_index] = [1]*4
            op_index += 1
            

    return incoming, outgoing, G, ops, gate, dilation, activation, bank_sizes, kernel_sizes, groups

# This function constructs the arrays containing all the various gradients of
# the architectural definition.
def construct_arch_grads(net,arch):

    (incoming, outgoing, G, 
         ops, gate, dilation, activation,
         bank_sizes, kernel_sizes, groups) = arch
    # Which banks we read from at each layer
    incoming_g = [[torch.zeros(net.widths[i] // (net.N * bank_sizes[i])) for _ in incoming[i]] for i in range(3)]

    # Which banks we write to at each layer
    outgoing_g = [[torch.zeros(net.widths[i] // (net.N * bank_sizes[i])) for _ in outgoing[i]] for i in range(3)]

    # Number of units for each layer
    G_g = [ [torch.zeros(net.k) for g in g_values] for g_values in G]

    # Which convolutions are active within a layer.
    ops_g = [ [torch.zeros(4) for op in op_values] for op_values in ops]
    
    # Whether to employ multiplicative gating at either in-layer junction.
    gate_g = [ [torch.zeros(2) for gated in gate_values] for gate_values in gate]
    
    # Filter dilation for each convolution within each layer
    ''' will still have to call dilation[i][j][0] without var_op for this'''
    dilation_g = [ [ [[torch.zeros(net.max_dilate), torch.zeros(net.max_dilate)] for _ in range(4 if net.var_op or net.big_op else 1)] for dil in dilate_values] for dilate_values in dilation ]
    
    # Activation function for each convolution within each layer.
    activation_g = [ [ [torch.zeros(len(net.nl_dict)) for _ in range(4 if net.var_op or net.big_op else 1)] for nl in nl_values] for nl_values in activation ]
    
    # Filter sizes for each convolution within each layer.
    kernel_sizes_g = [ [ [ [torch.zeros((net.final_max_kernel - 1) // 2), torch.zeros((net.final_max_kernel - 1) // 2)] 
                            for _ in range(4 if net.var_op or net.big_op else 1) ]
                            for ks in kernel_values] for kernel_values in kernel_sizes ]
    
    # Number of groups for each conv
    groups_g = [ [ [torch.zeros(len(net.factors)) for _ in range(4 if net.var_op or net.big_op else 1)]  for grp in group_values] for group_values in groups ]
    
    return incoming_g, outgoing_g, G_g, ops_g, gate_g, dilation_g, activation_g, kernel_sizes_g, groups_g
# Process:
# 1. Construct arrays to hold grads
# 2. Loop through c following the sample_weights loop, and accumulate gradients into 
#    grad arrays, regardless of how valid they are (or their length). We'll still keep
#    track of widths and stuff though so that we're not grabbing grads to read from banks
#    beyond a given block's max number of banks, but we won't care at this point if we're
#    getting grads to read from channels that have yet to be written to.
# 3. Loop through the grads following the sample_architecture loop and update the architecture
#   definition, replacing the random sampling with an SGD-style step.

def arch_grads(net, arch, arch_g=None, c=None):
    
    (incoming, outgoing, G, 
         ops, gate, dilation, activation,
         bank_sizes, kernel_sizes, groups) = arch
    
    if arch_g is None:
        arch_g = net.construct_arch_grads(arch)
    
    (incoming_g, outgoing_g, G_g, 
        ops_g, gate_g, dilation_g, activation_g, 
        kernel_sizes_g, groups_g) = arch_g
    if c is None:
        c = net.vc.grad.data.cpu()

    slice_index = 0
        

    
    # Build class_conditional vector

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

            sub_index = 0
            slice_N = net.depth_compression # How many slices we have
            
            for i, r in enumerate(read):
                # print(i,op_index,block_index)
                
                # A counter telling us where in net.c we are.
                channel_counter = 0
                
                slice_start = slice_index + sub_index // net.depth_compression
                slice_end = slice_index + (sub_index + bs) // net.depth_compression + (sub_index + bs) % net.depth_compression
     
                ''' take the mean across all but the channel dim.'''
                # c[:, r, :g, slice_start : slice_end] = 1
                incoming_g[block_index][op_index] += torch.mean(torch.mean(c[:, channel_counter : channel_counter + int(incoming_g[block_index][op_index].size(0)), :g, slice_start : slice_end],3),2).squeeze()                  
                channel_counter += net.max_banks

                # for w in write:
                    # c[:, channel_counter + w, :g, slice_start : slice_end] = 1
                outgoing_g[block_index][op_index] += torch.mean(torch.mean(c[:, channel_counter : channel_counter + int(outgoing_g[block_index][op_index].size(0)), :g, slice_start : slice_end],3),2).squeeze()
                channel_counter += net.max_banks
                
                # Block conditional entry, tell the net which block we're in
                # c[:, channel_counter + block_index, :g, slice_start : slice_end] = 1
                channel_counter += 3 # increment by number of blocks
                
                # G-conditional entry, can't be zero so the zero index corresponds to G=1
                # c[:, channel_counter + g - 1, :g, slice_start : slice_end] = 1
                G_g[block_index][op_index] += torch.mean(torch.mean(c[:, channel_counter : channel_counter + int(G_g[block_index][op_index].size(0)), :g, slice_start : slice_end],3),2).squeeze()
                channel_counter += net.k
                
                # If using the 2x2 op config
                if net.var_op or net.big_op or net.long_op:
                    
                    # Write dilation-conditional entries
                    for di, d in enumerate(dilate):
                        # c[:, -1 + d[0] + channel_counter, :g, slice_start : slice_end] = 1 if op[di] else 0
                        dilation_g[block_index][op_index][di][0] += torch.mean(torch.mean(c[:, channel_counter : channel_counter + int(dilation_g[block_index][op_index][di][0].size(0)), :g, slice_start : slice_end],3),2).squeeze()
                        channel_counter += 3
                        # c[:, -1 + d[1] + channel_counter, :g, slice_start : slice_end] = 1 if op[di] else 0
                        dilation_g[block_index][op_index][di][1] += torch.mean(torch.mean(c[:, channel_counter : channel_counter + int(dilation_g[block_index][op_index][di][1].size(0)), :g, slice_start : slice_end],3),2).squeeze()
                        channel_counter += 3
                        
                    if net.var_ks:
                        for ki, k in enumerate(ks):
                            # c[:,(-1 + k[0] // 2) + channel_counter, :g, slice_start : slice_end] = 1 if op[ki] else 0
                            kernel_sizes_g[block_index][op_index][ki][0] += torch.mean(torch.mean(c[:, channel_counter : channel_counter + int(kernel_sizes_g[block_index][op_index][ki][0].size(0)), :g, slice_start : slice_end],3),2).squeeze()
                            channel_counter += 3
                            # c[:,(-1 + k[1] // 2) + channel_counter, :g, slice_start : slice_end] = 1 if op[ki] else 0
                            kernel_sizes_g[block_index][op_index][ki][1] += torch.mean(torch.mean(c[:, channel_counter : channel_counter + int(kernel_sizes_g[block_index][op_index][ki][1].size(0)), :g, slice_start : slice_end],3),2).squeeze()
                            channel_counter += 3
                    
                    # Write op-conditional entries (if a conv is active)
                    if net.var_op:
                        # for o in list(np.where(np.asarray(op) > 0)[0]):
                            # c[:, channel_counter + o, :g, slice_start : slice_end] = 1
                        ops_g[block_index][op_index] += torch.mean(torch.mean(c[:, channel_counter : channel_counter + int(ops_g[block_index][op_index].size(0)), :g, slice_start : slice_end],3),2).squeeze()
                        channel_counter += 4
                    
                    if net.gates:
                        # Write gate-conditional entries
                        # for gi, gt in enumerate(gated):
                            # c[:, channel_counter + gi, :g, slice_start : slice_end] = gt
                        gate_g[block_index][op_index] += torch.mean(torch.mean(c[:, channel_counter : channel_counter + int(gate_g[block_index][op_index].size(0)), :g, slice_start : slice_end],3),2).squeeze()
                        channel_counter += 2
                    
                    if net.var_nl:
                        # Write activation-conditional entries
                        for nli, nl in enumerate(nls):
                            # c[:, channel_counter + nl, :g, slice_start : slice_end] = 1 if (op[nli] and not gated[nli//2]) else 0
                            activation_g[block_index][op_index][nli] += torch.mean(torch.mean(c[:, channel_counter : channel_counter + int(activation_g[block_index][op_index][nli].size(0)), :g, slice_start : slice_end],3),2).squeeze()
                            channel_counter += len(net.nl_dict)
                    
                    # Group-conditional entries
                    if net.var_group:
                        for grp_i, grp in enumerate(group):
                        # only denote the dilate if the op is active
                            groups_g[block_index][op_index][grp_i] += torch.mean(torch.mean(c[:, channel_counter : channel_counter + int(groups_g[block_index][op_index][grp_i].size(0)), :g, slice_start : slice_end],3),2).squeeze()
                           # c[:, -1 + factors(net.N).index(grp) + channel_counter, :g, slice_start : slice_end] = 1 if op[grp_i] else 0 
                            channel_counter += len(net.factors)
                # If just using a single 3x3 conv
                else:
                    # Write dilation-conditional entries
                    # c[:, -1 + dilate[0][0] + channel_counter, :g, slice_start : slice_end] = 1
                    # channel_counter += 3
                    # c[:, -1 +  dilate[0][1] + channel_counter, :g, slice_start : slice_end] = 1
                    # channel_counter +=3
                    dilation_g[block_index][op_index][0][0] += torch.mean(torch.mean(c[:, channel_counter : channel_counter + int(dilation_g[block_index][op_index][0][0].size(0)), :g, slice_start : slice_end],3),2).squeeze()
                    channel_counter += 3
                    dilation_g[block_index][op_index][0][1] += torch.mean(torch.mean(c[:, channel_counter : channel_counter + int(dilation_g[block_index][op_index][0][1].size(0)), :g, slice_start : slice_end],3),2).squeeze()
                    channel_counter += 3
                        
                    if net.var_ks:
                        # c[:,(-1 + ks[0][0] // 2) + channel_counter, :g, slice_start : slice_end] = 1
                        # channel_counter += 3
                        # c[:,(-1 + ks[0][1] // 2) + channel_counter, :g, slice_start : slice_end] = 1
                        # channel_counter += 3
                        kernel_sizes_g[block_index][op_index][0][0] += torch.mean(torch.mean(c[:, channel_counter : channel_counter + int(kernel_sizes_g[block_index][op_index][0][0].size(0)), :g, slice_start : slice_end],3),2).squeeze()
                        channel_counter += 3

                        kernel_sizes_g[block_index][op_index][0][1] += torch.mean(torch.mean(c[:, channel_counter : channel_counter + int(kernel_sizes_g[block_index][op_index][0][1].size(0)), :g, slice_start : slice_end],3),2).squeeze()
                        channel_counter += 3
                    # Write activation-conditional entries    
                    if net.var_nl:
                        # c[:, 2 * net.max_dilate + (2 * net.max_banks + 3) +nls[0], :g, slice_start : slice_end] = 1
                        activation_g[block_index][op_index][0] += torch.mean(torch.mean(c[:, channel_counter : channel_counter + int(activation_g[block_index][op_index][0].size(0)), :g, slice_start : slice_end],3),2).squeeze()  
                        channel_counter += len(net.nl_dict)
                        
                    if net.var_group:
                        # c[:, -1 + factors(net.N).index(group[0]) + channel_counter, :g, slice_start : slice_end] = 1
                        groups_g[block_index][op_index][0] += torch.mean(torch.mean(c[:, channel_counter : channel_counter + int(groups_g[block_index][op_index][0].size(0)), :g, slice_start : slice_end],3),2).squeeze()
                        channel_counter += len(net.factors)

                    
                sub_index += bs
                # Increment n every time we need to move to a new slice
                # of the trailing dimension of net.c
            # Increment slice_index based on how many banks we just went through?
            slice_index += bs * len(read) // net.depth_compression
    return incoming_g, outgoing_g, G_g, ops_g, gate_g, dilation_g, activation_g, kernel_sizes_g, groups_g
    
# Boilerplate to run architectural SGD, currently broken.
def perturb_SGD(net, arch, arch_g, perturb_probability=0):
        
    (incoming, outgoing, G, 
         ops, gate, dilation, activation,
         bank_sizes, kernel_sizes, groups) = arch
         
    (incoming_g, outgoing_g, G_g, 
        ops_g, gate_g, dilation_g, activation_g, 
        kernel_sizes_g, groups_g) = arch_g
    # We still keep track of the budget to make sure we don't overflow net.c
    for i, budget in enumerate(
        [(n // (net.N * net.depth_compression)) for n in net.nch_list]):

        # Initialize the number of input channels we've accumulated so far,
        # similar to computational budget
        used_budget=0

        # Still keep track of written channels to prevent reading from 
        # emtpy channels
        written_channels = [0] * min(int(net.in_channels[i] // (bank_sizes[i] * net.N) + np.ceil(net.depth * net.k) // bank_sizes[i]), net.widths[i] // (bank_sizes[i] * net.N))
        for index in range(net.in_channels[i] // (bank_sizes[i] * net.N)):
            written_channels[index] += 1

        min_budget_per_op = bank_sizes[i] if bank_sizes[i] % net.depth_compression else bank_sizes[i] // net.depth_compression
        min_reads_per_op = min_budget_per_op * net.depth_compression // bank_sizes[i]
        max_reads_per_op = len(written_channels) // min_reads_per_op * min_reads_per_op # round to nearest multiple of min_reads
        num_input_choices = list(range(min_reads_per_op, max_reads_per_op + min_reads_per_op, min_reads_per_op)) #then add min_reads
        
        op_index = 0
        ''' this has some failure cases since we don't currently delete
            later ops if we use up all the budget, but it should be fine.'''
        while (budget - used_budget) >= min_budget_per_op and op_index < len(G[i]): # this conditional also needs to stop not just at a less than but if we're near a multiple of 4 basically

            # Get all possible channels we can read from
            readable_channels = list(
                                 range(
                                  sum([item > 0\
                                    for item in written_channels])))
            
            empty_channels=[index for index, item
                                in enumerate(written_channels) if item == 0]
            # Max budgeted inputs is based on remaining budgeted slices, each of which gives depth_compression Ns, divided by N per bank
            max_budgeted_inputs = (budget - used_budget) * net.depth_compression  // bank_sizes[i]

            # max input banks has to be one of the allowable num_input values
            max_input_banks = max([path_choice for path_choice in num_input_choices if path_choice <= max_budgeted_inputs])
            
            # Don't think we don't really need to preference our incoming channels, 

            '''consider making this constant too? or allowing us to either randomize this or keep it the same'''
            if np.random.uniform() < perturb_probability or any([read in empty_channels for read in incoming[i][op_index]]):
                num_input_paths = min(
                                      int(np.random.choice([path_choice for path_choice in num_input_choices if path_choice <= len(readable_channels)])),
                                      max_input_banks)
                # Select read locations
                incoming[i][op_index] = sorted(np.random.choice(readable_channels, num_input_paths, replace=False))
            else:
                num_input_paths = len(incoming[i][op_index])
            
            
            if np.random.uniform() < perturb_probability:
                # Determine #of filters for this layer.
                # Presume it's an even multiple of bank sizes?
                G_choices = range(bank_sizes[i], net.k + 1, bank_sizes[i])
                # Most probable #filters based on our inputs
                most_probable_G = num_input_paths * bank_sizes[i]
                G_probabilities = [1./ (1e-2 + 10 * np.abs(g_choice - most_probable_G)) for g_choice in G_choices]
                # Normalize the probabilities.
                G_probabilities = [g_prob / sum(G_probabilities) for g_prob in G_probabilities]
                # Select number of filters.
                G[i][op_index] = int(np.random.choice(G_choices,p=G_probabilities))
            
            # Upate the budget                
            # The commented line here is for scaling the budget based on 
            # the number of output units, which won't accurately hold
            # the parametric budget but is more in line with a compute
            # budget.
            # int(np.ceil(number_of_inputs*float(G[i][-1]/net.N_max)))
            used_budget += num_input_paths * bank_sizes[i] // net.depth_compression
            
            if np.random.uniform() < perturb_probability:
                # now, select outgoing channels
                # Channels we haven't written to yet
                
            
       
                probability = np.exp(
                                np.asarray(
                                    range(
                                        max(len(readable_channels) // 2, G[i][op_index] // bank_sizes[i]), 
                                        G[i][op_index] // bank_sizes[i] - 1, 
                                        -1))) 
                probability = [p / sum(probability) for p in probability]

                
                # Select how many outputs we're going to have based on the
                # probability defined above. Allow at G//bank_sizes writes.
                number_of_outputs = np.random.choice(
                                     list(range(G[i][op_index] // bank_sizes[i], 
                                           1 + max(len(readable_channels) // 2, G[i][op_index] // bank_sizes[i])
                                           )), 
                                      p=probability)

                # Select which channels we're writing to
                outgoing_channels = list(
                                      sorted(
                                        np.random.choice(
                                          readable_channels 
                                          + empty_channels[:G[i][op_index] // bank_sizes[i]],
                                          number_of_outputs, replace=False)))
                                    
                # Make sure we only write sequentially to new empty channels, 
                # and don't skip any.
                num_empty_writes = len([o for o in outgoing_channels if o in empty_channels])
                outgoing_channels = ([o for o in outgoing_channels if o not in empty_channels]
                                    + empty_channels[:num_empty_writes])
                # Update output list and update which channels we've written to
                outgoing[i][op_index] = outgoing_channels
            # If we are writing to empty channels, ensure we're writing to the nearest empty channel.
            elif any([o in empty_channels for o in outgoing[i][op_index]]):
                num_empty_writes = len([o for o in outgoing[i][op_index] if o in empty_channels])
                outgoing_channels = ([o for o in outgoing[i][op_index] if o not in empty_channels]
                                    + empty_channels[:num_empty_writes])
                outgoing[i][op_index] = outgoing_channels
            else:
                outgoing_channels = outgoing[i][op_index]
            # print(i,used_budget,len(readable_channels), outgoing_channels,len(written_channels))
            for o in outgoing_channels:
                written_channels[o] += 1

            # Possible op configurations; 
            # Note that we don't bother to have the option to have w[2] 
            # alone, since although in the SMASH network that would 
            # be different, the resulting network would not be different
            # (i.e. it would just be two ways to define a single conv)
            if np.random.uniform() < perturb_probability:
                ops[i][op_index] = net.options[int(np.random.choice(len(net.options), p=net.options_probabilities))]
            
            # Decide if we're going to have a multiplicative tanh-sig gate
            # at either of the two parallel layers of the op.
            # Randomly sample activation functions;
            # note that this will be overriden in the main net if
            # a relevant gate is active, and is accordingly also
            # ignored where appropriate in the definition of net.c
            if np.random.uniform() < perturb_probability:
                if net.var_nl:
                    activation[i][op_index] = [np.random.choice(
                                            list(
                                              net.nl_dict.keys())) 
                                          for _ in range(4)]
                else:
                    activation[i][op_index] = [0]*4                  
            
            # If we're using gates and g//2 is divisible by bank size,
            # then roll for gates
            # If we're using preactivation, then only allow one add-split-mult gate,
            # else our channel count will be messy.
            if np.random.uniform() < perturb_probability:
                if net.gates and (G[i][op_index]//2 > 0 ) and not (G[i][op_index]//2) % bank_sizes[i]:
                    gt = np.random.uniform() < 0.25 if ops[i][op_index][0] and ops[i][op_index][2] else 0
                    gt = [gt, np.random.uniform() < 0.25  if ops[i][op_index][1] and ops[i][op_index][3] and not gt else 0]
                    
                    gate[i][op_index] = gt
                    # If not using preactivation, pass tanh and sigmoid NLs
                    if not net.preactivation:
                        if gate[i][0]:
                            activation[i][op_index][0] = 1
                            activation[i][op_index][2] = 2
                        if gate[i][1]:
                            activation[i][op_index][1] = 1
                            activation[i][op_index][3] = 2
                else:
                    gate[i][op_index] = [0,0]
            if np.random.uniform() < perturb_probability:
                kernel_sizes[i][op_index] = [list(np.random.choice(range(3,net.max_kernel+2,2),2)) for _ in range(4)]
            
            # Randomly sample dilation factors for each conv,
            # limiting the upper dilation based on the kernel size.
            if np.random.uniform() < perturb_probability:
                dilation[i][op_index] = [ [int(np.random.randint(1, 5-(kernel_sizes[i][op_index][j][0]-1)//2)),
                                      int(np.random.randint(1, 5-(kernel_sizes[i][op_index][j][1]-1)//2))]
                                   for j in range(4)]
            
            # Allow the number of groups to be up to the third-largest factor
            # of G, so for G=64, with factors of [1,2,4,8,16,32,64]
            # this would allow for 16 groups.
            if np.random.uniform() < perturb_probability:
                if net.var_group:
                    groups[i][op_index] = [np.random.choice(net.factors) for _ in range(4)]
                else:
                    groups[i][op_index] = [1]*4
            op_index += 1
            

    return incoming, outgoing, G, ops, gate, dilation, activation, bank_sizes, kernel_sizes, groups
