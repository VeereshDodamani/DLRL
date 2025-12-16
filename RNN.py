import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

text = "The beautiful girl whom I met last time is very intelligent also"

chars = sorted(list(set(text)))
char_to_index = {c: i for i, c in enumerate(chars)}
index_to_char = {i: c for i, c in enumerate(chars)}

vocab_size = len(chars)
seq_length = 5

sequences = []
labels = []

for i in range(len(text) - seq_length):
    seq = text[i:i + seq_length]
    label = text[i + seq_length]
    sequences.append([char_to_index[c] for c in seq])
    labels.append(char_to_index[label])

X = np.array(sequences)
y = np.array(labels)

X_one_hot = tf.one_hot(X, vocab_size)
y_one_hot = tf.one_hot(y, vocab_size)

model = Sequential()
model.add(SimpleRNN(
    units=50,
    activation='relu',
    input_shape=(seq_length, vocab_size)
))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_one_hot, y_one_hot, epochs=100, verbose=0)


start_seq = "The handsome boy whom I met "
generated_text = start_seq
generate_length = 50

for _ in range(generate_length):
    last_seq = generated_text[-seq_length:]
    x = np.array([[char_to_index[c] for c in last_seq]])
    x_one_hot = tf.one_hot(x, vocab_size)

    prediction = model.predict(x_one_hot, verbose=0)
    next_index = np.argmax(prediction)
    next_char = index_to_char[next_index]

    generated_text += next_char

print("Generated Text:")
print(generated_text)
