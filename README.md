# Siamese Networks for One-Shot Learning
Forked from the link https://github.com/kevinzakka/one-shot-siamese <br/>
But customized to be executed on current environment and different datasets

## Paper Modifications

I've done some slight modifications to the paper to eliminate variables while I debug my code. Specifically, validation and test accuracy currently suck so I'm checking if there's a bug either in the dataset generation or trainer code.

- I'm using `Relu -> Maxpool` rather than `Maxpool - Relu`.
- I'm using batch norm between the conv layers.
- I'm using He et. al. initialization.
- I'm using a global learning rate, l2 reg factor, and momentum rather than per-layer parameters.

## Omniglot Dataset

Please use the dataset in the `Omniglot` folder

* Process the data using `data_prep_efficient.ipynb`

## Tips for excuting the codes for training Siamese Network

- All the needed hyperparameters including the dataset folder path are predefined in the `config.py` as default value
- The training is emulated on GPU, so please make sure the compatible CUDA and cuDNN are installed
- Current configurations are CUDA(11.7), Python(3.9), PyTorch(2.0) in the virtual environment `GPU_AmpStruc`
- If you want to switch to the tensorflow GPU, please change CUDA version in the CUDA_PATH from 11.7 to 11.2

Checkout [Playground.ipynb](https://github.com/kevinzakka/siamese-network/blob/master/Playground.ipynb) for a minimal working example.
