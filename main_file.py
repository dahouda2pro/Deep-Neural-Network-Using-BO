# SETUP
print("Setup")

from matplotlib import pyplot
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


# 1. Load the Data
(train_images, train_labels),(test_images, test_labels) = fashion_mnist.load_data()

# 2. Split data into train, validation and test sets
train_x, val_x, train_y, val_y = train_test_split(train_images, train_labels, stratify=train_labels, random_state=48, test_size=0.05)
(test_x, test_y) = (test_images, test_labels)

# 3. Normalize the pixels to range 0-1
train_x = train_x / 255.0
val_x = val_x / 255.0
test_x = test_x / 255.0

# 4. One-hot encode target variable
train_y = to_categorical(train_y)
val_y = to_categorical(val_y)
test_y = to_categorical(test_y)

# Print the Shape of all the Dataset
print("Training set:", train_x.shape)
print("Training set:", train_y.shape)
print("Validation set:", val_x.shape)
print("Validation set:", val_y.shape)
print("Testing set:", test_y.shape)
print("Testing set:", test_y.shape)