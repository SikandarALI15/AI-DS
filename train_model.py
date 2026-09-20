"""
train_model.py
Ye script MNIST dataset use karke ek simple ANN (Artificial Neural Network)
train karti hai aur usay 'mnist_ann_model.h5' ke naam se save karti hai.

Run karne ka tareeqa:
    python train_model.py
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1) MNIST dataset load karo (60,000 training + 10,000 test images, 28x28 grayscale)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 2) Pixel values ko 0-1 ke beech normalize karo (0-255 -> 0-1)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 3) ANN model banate hain (Flatten + Dense layers)
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),   # 28x28 image ko 784 values ki line bana dega
    layers.Dense(128, activation="relu"),    # Hidden layer 1
    layers.Dense(64, activation="relu"),     # Hidden layer 2
    layers.Dense(10, activation="softmax")   # Output layer: 10 classes (0-9)
])

# 4) Model compile karo
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 5) Model train karo
model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1
)

# 6) Test data par accuracy check karo
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")

# 7) Model save karo (isay API me load karenge)
model.save("mnist_ann_model.h5")
print("Model 'mnist_ann_model.h5' ke naam se save ho gaya hai.")
