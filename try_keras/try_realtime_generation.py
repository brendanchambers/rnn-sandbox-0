__author__ = 'Brendan'


from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras.datasets import mnist
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import time
import numpy
from keras import losses
from keras.layers import Input, Dense, Lambda, Layer
from keras import losses
from keras import metrics
from keras import regularizers
from keras import backend as K

batch_size=2000
original_dim=784
latent_dim=2
intermediate_dim=256
epochs=3
epsilon_std=1.0

x = Input(batch_shape=(batch_size, original_dim))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_sigma = Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                               stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])


############ loading ##############


autoencoder = load_model('vae_autoencoder.h5')
encoder = load_model('vae_encoder.h5')
generator = load_model('vae_generator.h5')


fig = plt.figure( 1 )
ax = fig.add_subplot( 111 )
ax.set_title("Try animating latent space")
im = ax.imshow( numpy.zeros( ( 28, 28 ) ) ) # Blank starting image
fig.show()
im.axes.figure.canvas.draw()

n = 15  # figure with 15x15 digits
digit_size = 28

# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        tstart = time.time()
        z_sample = np.array([[xi, yi]])
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        print np.shape(digit)
        #figure[i * digit_size: (i + 1) * digit_size,
        #       j * digit_size: (j + 1) * digit_size] = digit
        im.set_data(digit)
        im.axes.figure.canvas.draw()
        print ( 'FPS:', 1.0 / ( time.time() - tstart ) )



#plt.figure(figsize=(10, 10))
#plt.imshow(figure, cmap='Greys_r')





# https://stackoverflow.com/questions/9878200/plot-a-sequence-of-images-with-matplotlib-python




