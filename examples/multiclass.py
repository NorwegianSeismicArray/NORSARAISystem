import numpy as np
from nais.Models import AlexNet1D

num_classes = 3
num_channels = 9
num_samples = 32

X = np.random.normal(size=(num_samples,256,num_channels)) #16 examples of three channel data.
X_test = np.random.normal(size=(num_samples,256,num_channels)) #16 examples of three channel data.
y = np.random.randint(0,1,size=(num_samples,num_classes)) #Labels

model = AlexNet1D(num_outputs=num_classes, 
                  output_type == 'multiclass')
model.compile('adam','categorical_crossentropy')
model.fit(X,y)

p = model.predict(X_test)
np.save(p, 'predictions.npy')
