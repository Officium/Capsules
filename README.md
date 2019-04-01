# Capsules

CapsNet implementations with PyTorch. Randomness in codes is removed.


# Requirements

Python 3.5    
PyTorch 0.4.1

# Dynamic Routing Between Capsules

[Hinton G E, Sabour S, Frosst N. Matrix capsules with EM routing[J]. 2018.](http://papers.nips.cc/paper/6975-dynamic-routing-between-capsules.pdf)    
With the best hyperparameters on MNIST dataset in the paper, training will takes 5 hours with two GTX 1070.

|  | Best Test Accuracy |
| - | -------- |
| Paper | 99.75% |
| This implementation with fine-tune | 99.74% |
| This implementation | 99.72% |
| [Implementation by @gram-ai](https://github.com/gram-ai/capsule-networks) | 99.70% |
| [Implementation by @XifengGuo](https://github.com/XifengGuo/CapsNet-Keras) | 99.66% |
| [Implementation by @naturomics](https://github.com/naturomics/CapsNet-Tensorflow) | 99.64% |


# Matrix capsules with EM routing

[Sabour S, Frosst N, Hinton G E. Dynamic routing between capsules[C]//Advances in neural information processing systems. 2017: 3856-3866.](https://openreview.net/pdf?id=HJWLfGWRb)       
With the best hyperparameters on smallNORB dataset in the paper, training will takes 40 hours with eight Tesla P40.

|  | Best Test Accuracy | Training Speed |
| - | -------- | ------- |
| Paper | 98.2% | - |
| This Implementation | 90.9% | < 1s/iter with 8 Tesla P40 |
| [Implementation by @www0wwwjs1](https://github.com/www0wwwjs1/Matrix-Capsules-EM-Tensorflow) | 91.8% | 25s/iter with 1 Tesla P40 |
