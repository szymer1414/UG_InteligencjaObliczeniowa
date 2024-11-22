import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import History
# Load dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# Preprocess data
#[a]
#reshape dodaje do obrazów nowy wymiar, nastpenei dzielimy kazdy pixel przez 255, aby uzyksac znormlaizowane wartosci od [0-1]
#to_categorical zmienia reprezetnacje liczbowa na one-hot 
#np.argmax wybiera argument z najwiekszym prawdopodobienstwem?
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)).astype('float32') /255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
original_test_labels = np.argmax(test_labels, axis=1) # Save original labels for confusion matrix
# Define model

#[b]
model = Sequential([
Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),#naklada 32 razy filtr 3x3 aby wyszkuac malych wzorcow/28x28x1 -> 26x26x32 
MaxPooling2D((2, 2)),#2krotnie redukuje dane, zostawiajac jedynie najwaznijesze cechy / 26x26x32 - > 13x13x32
Flatten(),#zmienia dane na jednowymiarowy wektor /  13x13x32 -> wektor
Dense(64, activation='relu'),#program uczy sie przypisujac do kazdego wetkroa 62 cechy/ wektor -> 64 cechy
Dense(10, activation='softmax')#zwraca prawodpodobienstwo przynaleznosci do jedenj z 10klas / 64 cechy -> 10 liczb [0-1]
])
#[c]
#najczesciej mylone sa cyfry
#2-7 #4-9 #9-7 #5-3
#[d]
#wystepuje przeuczenie sie sieci. /na tyle drobne ze chyba mozna uznac model za dobrze dopasowany.
#[e]
from tensorflow.keras.callbacks import Callback

class saveCallback(Callback):
    def __init__(self, filepath):
        super().__init__()
        self.best_accuracy = 0
        self.filepath = filepath
    def on_epoch_end(self, logs=None):
        val_accuracy = logs.get('val_accuracy')
        if val_accuracy > self.best_accuracy: 
            self.best_accuracy = val_accuracy
            self.model.save(self.filepath) 



checkpoint = saveCallback(filepath='plikzad3.h5')
'''
from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint(
    'plikzad3.keras',      # Zmieniono rozszerzenie na .keras
    monitor='val_accuracy', 
    save_best_only=True,    
    mode='max'
)'''

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Train model
history = History()
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2,
callbacks=[checkpoint]) #checkpoit
# Evaluate on test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.4f}")
# Predict on test images
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)
# Confusion matrix
cm = confusion_matrix(original_test_labels, predicted_labels)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
# Plotting training and validation accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True, linestyle='--', color='grey')
plt.legend()
# Plotting training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', color='grey')
plt.legend()
plt.tight_layout()
plt.show()
# Display 25 images from the test set with their predicted labels
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i].reshape(28,28), cmap=plt.cm.binary)
    plt.xlabel(predicted_labels[i])
plt.show()