# Decompose
Blind source separation based on the probabilistic tensor factorisation framework

## Installation
Decompose demands python 3.6 and tensorflow 1.6. The newest gibhub code of decompose can be installed using pip:
```bash
pip3 install git+https://github.com/bethgelab/decompose
```

## Quick start
Decompose provides an interface that is similar to the interface of scikit-learn:

```python
import numpy as np
from sklearn.datasets import make_low_rank_matrix

from decompose.sklearn import DECOMPOSE
from decompose.distributions.cenNormal import CenNormal


# create a numpy array containing a synthetic low rank dataset
X = make_low_rank_matrix(n_samples=1000, n_features=1000,
                         effective_rank=3, tail_strength=0.5)

# create an instance of a decompose model
model = DECOMPOSE(modelDirectory="/tmp/myNewModel",
                  priors=[CenNormal, CenNormal],
                  n_components=3)

# train the model and transform the training data
U0 = model.fit_transform(X)

# learned filter bank
U1 = model.components_

# variance ratio of the sources
varianceRatio = model.variance_ratio_

# reconstruction of the data
XHat = np.dot(U0.T, U1)
```

## Publication
The publication linked to this repository is available on arXiv:
[[1803.08882] Trace your sources in large-scale data: one ring to find them all](http://arxiv.org/abs/1803.08882)

## Version
The repository is still in alpha stage and does not yet contain all the code used in the publication above. In the next days documentation and features will be added.
