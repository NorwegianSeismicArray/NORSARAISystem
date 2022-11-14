### NORSAR Artificial Inteligence System (NAIS, yes, the acronym was chosen first)
This repository was originally developed to contain tools specific to aiding in AI research at NORSAR. 

This repository is based on Tensorflow and Keras. 

This repository contains: 
* Baseline models, for waveforms and spectrograms.
  * Classification
  * Regression
  * Masking (eg. arrival picking)
* Augmentation methods.

## Installation
``pip install git+ssh://git@github.com/NorwegianSeismicArray/NORSARAISystem.git#egg=nais``

If an error occurs when importing `nais`, likely ``sndfile`` library is not installed (check the error), and needs to be:

``apt-get install libsndfile1-dev``

## Quick example

```python
import numpy as np
from nais.Models import AlexNet1D

X = np.random.normal(size=(16,256,3)) #16 examples of three channel data.
y = np.random.randint(0,1,size=(16,)) #Labels

model = AlexNet1D(num_outputs=1) #binary 
model.compile('adam','binary_crossentropy')
model.fit(X,y)
```

## Contributing
We appreciate all contributions as long as they are related to core problems the repository will aid in solving. 
