import numpy as np
from nais.Models import AlexNet1D

X = np.random.normal(size=(16,256,3)) #16 examples of three channel data.
y = np.random.randint(0,1,size=(16,)) #Labels

model = AlexNet1D(num_outputs=1) #binary 
model.compile('adam','binary_crossentropy')
model.fit(X,y)
