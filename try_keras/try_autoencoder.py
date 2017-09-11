__author__ = 'Brendan'
# based on https://blog.keras.io/building-autoencoders-in-keras.html

from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt


encoding_dim = 32 # number of floats

input_img = Input(shape=(784,)) # init
# encode
encoded = Dense(encoding_dim, activation='relu')(input_img)
# decode
decoded = Dense(784, activation='sigmoid')(encoded)

# define the mapping of input to reconstruction
autoencoder = Model(input_img, decoded)

### encoder alone
encoder = Model(input_img, encoded)
# decoder alone
encoded_input = Input(shape=(encoding_dim,))
decoder_layer= autoencoder.layers[-1] # last layer
decoder = Model(encoded_input, decoder_layer(encoded_input))

##### train (todo train adversarially)
autoencoder.compile(optimizer='adadelta',loss='binary_crossentropy')

# prep data
(x_train,_),(x_test,_) = mnist.load_data()
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.
x_train = x_train.reshape((len(x_train),np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test),np.prod(x_test.shape[1:])))
print x_train.shape
print x_test.shape

# train 50x
autoencoder.fit(x_train,x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test,x_test))
# test
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

n=10
plt.figure(figsize=(20,4))
for i in range(n):
    #original
    ax = plt.subplot(2,n,i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # reconstruction
    ax = plt.subplot(2,n,i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_vsibile(False)
plt.show()