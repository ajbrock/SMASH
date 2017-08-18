# SMASH: One-Shot Model Architecture Search through HyperNetworks
An experimental technique for efficiently exploring neural architectures.

![SMASHGIF](http://i.imgur.com/OTOvstW.gif)

This repository contains code for the SMASH [paper](https://arxiv.org/abs/1708.05344) and [video](https://www.youtube.com/watch?v=79tmPL9AL48). 

SMASH bypasses the need for fully training candidate models by learning an auxiliary HyperNet to approximate model weights, allowing for rapid comparison of a wide range of network architectures at the cost of a single training run.


## Installation
To run this script, you will need [PyTorch](http://pytorch.org) and a CUDA-capable GPU. If you wish to run it on CPU, just remove all the .cuda() calls.

Note that this code was written in PyTorch 0.12, and is not guaranteed to work on 0.2 until next week when I get a chance to update my own version. Please also be aware that, while thoroughly commented, this is research code for a heckishly complex project. I'll be doing more cleanup work to improve legibility soon.

## Running
To run with default parameters, simply call

```sh
python train.py
```

This will by default train a SMASH net with nominally the same parametric budget as a WRN-40-4.
Note that validation scores during training are calculated using a random architecture for each batch, and are therefore sort of an "average" measure.

After training, to sample and evaluate SMASH scores, call

```sh
python train.py --SMASH=YOUR_MODEL_NAME_HERE.pth
```

This will by default sample 500 random architectures, then perturb the best-found architecture 100 times, then employ a sort of Markov Chain to further perturb the best found architecture.

There are lots of different options, including a number of experimental settings such as architectural gradient descent by proxy, in-op multiplicative gating, variable nonlinearities, setting specific op configuration types. Take a look at the train_parser in utils.py for details, though note that some of these weirder ones may be deprecated. 

This code has boilerplate for loading Imagenet32x32 and ModelNet, but doesn't download or preprocess them on its own. 
## Notes
This README doc is in very early stages, and will be updated soon.

## Acknowledgments
- Training and Progress code acquired in a drunken game of SpearPong with Jan Schlüter: https://github.com/Lasagne/Recipes/tree/master/papers/densenet
- Metrics Logging code extracted from ancient diary of Daniel Maturana: https://github.com/dimatura/voxnet

