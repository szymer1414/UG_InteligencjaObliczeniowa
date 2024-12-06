from os import listdir
from numpy import asarray
from numpy import save
import os
import shutil
import random
from numpy import load
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import History
from keras.preprocessing import image
'''
# define location of dataset
folder = 'train/'
photos, labels = list(), list()
# enumerate files in the directory
for file in listdir(folder):
	# determine class
	output = 0.0
	if file.startswith('dog'):
		output = 1.0
	# load image
	photo = load_img(folder + file, target_size=(40, 40))
	# convert to numpy array
	photo = img_to_array(photo)
	# store
	photos.append(photo)
	labels.append(output)
# convert to a numpy arrays
photos = asarray(photos)
labels = asarray(labels)
print(photos.shape, labels.shape)
# save the reshaped photos
save('dogs_vs_cats_photos.npy', photos)
save('dogs_vs_cats_labels.npy', labels)

photos = load('dogs_vs_cats_photos.npy')
labels = load('dogs_vs_cats_labels.npy')
print(photos.shape, labels.shape)
'''

#dzielnie zbioru na train i test. 
'''
# create directories
dataset_home = 'dataset_dogs_vs_cats/'
subdirs = ['train/', 'test/']
for subdir in subdirs:
	# create label subdirectories
	labeldirs = ['dogs/', 'cats/']
	for labldir in labeldirs:
		newdir = dataset_home + subdir + labldir
		os.makedirs(newdir, exist_ok=True)
		
# seed random number generator

random.seed(10)
# define ratio of pictures to use for validation
val_ratio = 0.3
# copy training dataset images into subdirectories
src_directory = 'train/'
for file in listdir(src_directory):
	src = src_directory + '/' + file
	dst_dir = 'train/'
	if random.random() < val_ratio:
		dst_dir = 'test/'
	if file.startswith('cat'):
		dst = dataset_home + dst_dir + 'cats/'  + file
		shutil.copyfile(src, dst)
	elif file.startswith('dog'):
		dst = dataset_home + dst_dir + 'dogs/'  + file
		shutil.copyfile(src, dst)

'''
# baseline model for the dogs vs cats dataset
import sys
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input
from keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import math
from tensorflow.keras.callbacks import ModelCheckpoint

# define cnn model
def define_model():
    model = Sequential()
    model.add(Input(shape=(40, 40, 3)))  # Dodano warstwę Input
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = Adam(learning_rate=0.001)  # Zmieniono 'lr' na 'learning_rate'
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Plot diagnostic learning curves
def summarize_diagnostics(history):
    # plot loss
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', color='grey')
    
    # plot accuracy
    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', color='grey')
    
    # save plot to file
    filename = sys.argv[0].split('/')[-1].split('\\')[-1].split('.')[0]  # Obsługuje różne systemy plików
    plt.savefig(filename + '_plot.png')
    plt.close()

# Run the test harness for evaluating a model
def run_test_harness():
    # Define model
    model = define_model()
    
    # Create data generator
    datagen = ImageDataGenerator(rescale=1.0/255.0)
    
    # Prepare iterators
    train_it = datagen.flow_from_directory(
        'dataset_dogs_vs_cats/train/',
        class_mode='binary',
        batch_size=64,
        target_size=(40, 40)
    )
    test_it = datagen.flow_from_directory(
        'dataset_dogs_vs_cats/test/',
        class_mode='binary',
        batch_size=64,
        target_size=(40, 40)
    )
    
    # Oblicz steps_per_epoch i validation_steps
    steps_per_epoch = math.ceil(train_it.samples / train_it.batch_size)
    validation_steps = math.ceil(test_it.samples / test_it.batch_size)
    

    checkpoint = ModelCheckpoint(
        'best_model.keras',         
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    
    # Fit model
    history = model.fit(
        train_it,
        steps_per_epoch=steps_per_epoch,
        validation_data=test_it,
        validation_steps=validation_steps,
        epochs=5,
        verbose=1,
        callbacks=[checkpoint]  
    )
    
    # Evaluate model
    _, acc = model.evaluate(test_it, steps=validation_steps, verbose=0)
    print('> Test Accuracy: %.3f%%' % (acc * 100.0))
    
    # Learning curves
    summarize_diagnostics(history)



# Znalezienie błędnych klasyfikacji
    predictions = model.predict(test_it, steps=validation_steps, verbose=1)
    predicted_classes = (predictions > 0.5).astype("int32").flatten()  
    true_classes = test_it.classes 
    class_labels = list(test_it.class_indices.keys())  


    errors = np.where(predicted_classes != true_classes)[0]
    print(f"Liczba błędnych klasyfikacji: {len(errors)}")

    for i in range(min(5, len(errors))): 
        idx = errors[i]
        img_path = test_it.filepaths[idx]
        img = image.load_img(img_path, target_size=(40, 40))
        plt.imshow(img)
        plt.title(f"Prawda: {class_labels[true_classes[idx]]}, Przewidziane: {class_labels[predicted_classes[idx]]}")
        plt.axis('off')
        plt.show()
# Entry point, run the test harness
if __name__ == '__main__':
    run_test_harness()