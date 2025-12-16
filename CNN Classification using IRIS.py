import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

X = (X - X.min()) / (X.max() - X.min())

X_img = X.reshape(-1, 2, 2, 1)

y_cat = to_categorical(y, 3)

X_train, X_test, y_train, y_test = train_test_split(
    X_img, y_cat, test_size=0.2, random_state=42
)

model = models.Sequential([
    layers.Conv2D(16, (2, 2), activation='relu', input_shape=(2, 2, 1)),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=40, batch_size=8, verbose=0)


sample = X_test[0]
true_label = np.argmax(y_test[0])

prediction = model.predict(sample.reshape(1, 2, 2, 1))
predicted_label = np.argmax(prediction)

plt.imshow(sample.reshape(2, 2), cmap='gray')
plt.title(
    f"True: {class_names[true_label]}\nPredicted: {class_names[predicted_label]}"
)
plt.axis('off')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

X = (X - X.min()) / (X.max() - X.min())

X = X.reshape(-1, 2, 2, 1)

y = to_categorical(y, 3)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = models.Sequential([
    layers.Conv2D(16, (2, 2), activation='relu', input_shape=(2, 2, 1)),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=40, batch_size=8, verbose=0)

sample = X_test[0]
true_label = np.argmax(y_test[0])

prediction = model.predict(sample.reshape(1, 2, 2, 1))
predicted_label = np.argmax(prediction)

plt.figure(figsize=(4, 4))
plt.imshow(sample.reshape(2, 2), cmap='viridis')
plt.colorbar()
plt.axis('off')

plt.title(
    f"Iris Flower Classification\n\n"
    f"Actual Flower   : {class_names[true_label].capitalize()}\n"
    f"Predicted Flower: {class_names[predicted_label].capitalize()}",
    fontsize=10
)

plt.show()
