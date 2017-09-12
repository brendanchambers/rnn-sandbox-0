__author__ = 'Brendan'


from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import time
import numpy
from keras import losses

autoencoder = load_model('vae_autoencoder.h5')
encoder = load_model('vae_encoder.h5')
generator = load_model('vae_generator.h5')


plt.figure() # canvas for animation
n = 15  # figure with 15x15 digits
digit_size = 28

# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        #figure[i * digit_size: (i + 1) * digit_size,
        #       j * digit_size: (j + 1) * digit_size] = digit
        plt.imshow(digit)
        plt.show()

#plt.figure(figsize=(10, 10))
#plt.imshow(figure, cmap='Greys_r')





# https://stackoverflow.com/questions/9878200/plot-a-sequence-of-images-with-matplotlib-python
fig = plt.figure( 1 )
ax = fig.add_subplot( 111 )
ax.set_title("My Title")

im = ax.imshow( numpy.zeros( ( 256, 256, 3 ) ) ) # Blank starting image
fig.show()
im.axes.figure.canvas.draw()

tstart = time.time()
for a in xrange( 100 ):
  data = numpy.random.random( ( 256, 256, 3 ) ) # Random image to display
  ax.set_title( str( a ) )
  im.set_data( data )
  im.axes.figure.canvas.draw()

print ( 'FPS:', 100 / ( time.time() - tstart ) )