import os
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint
from keras.datasets.mnist import load_data
from keras.optimizers import Adam
from keras.models import Sequential, load_model
from keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout
from matplotlib import pyplot
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
# Define the standalone discriminator model
def define_discriminator(in_shape=(28,28,1)):
    model = Sequential()
    
    model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    
    # compile model
    opt = Adam(learning_rate=0.0001, beta_1=0.5, clipvalue=1.0)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

# Define the standalone generator model
def define_generator(latent_dim):
    model = Sequential()
    # Foundation for 7x7 image
    n_nodes = 128 * 7 * 7
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((7, 7, 128)))
    # Upsample to 14x14
    model.add(Conv2DTranspose(256, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # Upsample to 28x28
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
    return model

# Define the combined generator and discriminator model for updating the generator
def define_gan(g_model, d_model):
    # Make weights in the discriminator not trainable
    d_model.trainable = False
    # Connect the generator and discriminator
    model = Sequential()
    # Add generator
    model.add(g_model)
    # Add discriminator
    model.add(d_model)
    # Compile model
    opt = Adam(learning_rate=0.0001, beta_1=0.5, clipvalue=1.0)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

# Load and prepare MNIST training images
def load_real_samples():
    # Load MNIST dataset
    (trainX, _), (testX, _) = load_data()
    # Expand to 3d, e.g., add channels dimension
    X = expand_dims(testX, axis=-1)
    # Convert from unsigned ints to floats
    X = X.astype('float32')
    # Scale from [0,255] to [0,1]
    X = X / 255.0
    return X

# Select real samples
def generate_real_samples(dataset, n_samples):
    # Choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # Retrieve selected images
    X = dataset[ix]
    # Generate 'real' class labels (1)
    y = ones((n_samples, 1))
    return X, y

# Generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # Generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # Reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

# Use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
    # Generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # Predict outputs
    X = g_model.predict(x_input)
    # Create 'fake' class labels (0)
    y = zeros((n_samples, 1))
    return X, y

# Create and save a plot of generated images (reversed grayscale)
def save_plot(examples, epoch, n=10):
    # Plot images
    for i in range(n * n):
        # Define subplot
        pyplot.subplot(n, n, 1 + i)
        # Turn off axis
        pyplot.axis('off')
        # Plot raw pixel data
        pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
    # Save plot to file
    filename = 'generated_plot_e%03d.png' % (epoch+1)
    pyplot.savefig(filename)
    pyplot.close()

# Evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
    # Prepare real samples
    X_real, y_real = generate_real_samples(dataset, n_samples)
    # Evaluate discriminator on real examples
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    # Prepare fake examples
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    # Evaluate discriminator on fake examples
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    # Summarize discriminator performance
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
    # Save plot
    save_plot(x_fake, epoch)
    # Save the generator model to file
    filename = 'generator_model_%03d.h5' % (epoch + 1)
    g_model.save(filename)

# Save the discriminator model (optional)
def save_discriminator(d_model, epoch):
    filename = f'discriminator_model_{epoch + 1}.h5'
    d_model.save(filename)
    print(f'Saved discriminator model to {filename}')

# Function to load models if they exist
def load_gan_models(generator_file='generator_model_010.h5', discriminator_file='discriminator_model.h5'):
    # Check if the generator model exists
    if os.path.exists(generator_file):
        print(f'Loading generator model from {generator_file}')
        g_model = load_model(generator_file)
    else:
        print('No saved generator model found, creating a new one...')
        g_model = define_generator(latent_dim)

    # Check if the discriminator model exists
    if os.path.exists(discriminator_file):
        print(f'Loading discriminator model from {discriminator_file}')
        d_model = load_model(discriminator_file)
    else:
        print('No saved discriminator model found, creating a new one...')
        d_model = define_discriminator()

    # Return the models
    return g_model, d_model

# Train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=50, n_batch=256):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # Manually enumerate epochs
    for i in range(n_epochs):
        # Enumerate batches over the training set
        for j in range(bat_per_epo):
            # Get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            
            # Generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # Create training set for the discriminator
            X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
            # Update discriminator model weights
            d_loss, _ = d_model.train_on_batch(X, y)
            # Prepare points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, n_batch)
            # Create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            # Update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            # Summarize loss on this batch
            d_loss_value = d_loss[0] if isinstance(d_loss, (list, tuple)) else d_loss
            g_loss_value = g_loss[0] if isinstance(g_loss, (list, tuple)) else g_loss

            # Summarize loss on this batch
            print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, bat_per_epo, d_loss_value, g_loss_value))
        
        # Save the models at the end of each epoch
        summarize_performance(i, g_model, d_model, dataset, latent_dim)
        
        # Save the discriminator model after each epoch (optional)
        save_discriminator(d_model, i)

# Size of the latent space
latent_dim = 100
# Load the models
generator_file = 'generator_model_010.h5'
discriminator_file = 'discriminator_model.h5'
g_model, d_model = load_gan_models(generator_file, discriminator_file)
gan_model = define_gan(g_model, d_model)

# Load image data
dataset = load_real_samples()

# Number of epochs
n_epochs = 5
# Batch size
batch_size = 512

# Train the model
train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs, batch_size)
