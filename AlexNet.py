from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization


class AlexNet(Sequential):
    def __init__(self, input_shape, num_classes):
        super(AlexNet, self).__init__()

        self.add(Conv2D(
            filters=96,
            kernel_size=(11, 11),
            strides=4,
            padding='valid',
            activation='relu',
            input_shape=input_shape
        ))
        self.add(BatchNormalization())
        self.add(MaxPooling2D(pool_size=(3, 3), strides=2))

        self.add(Conv2D(
            filters=256,
            kernel_size=(5, 5),
            strides=1,
            padding='same',
            activation='relu'
        ))
        self.add(BatchNormalization())
        self.add(MaxPooling2D(pool_size=(3, 3), strides=2))

        self.add(Conv2D(384, kernel_size=(3, 3), padding='same', activation='relu'))

        self.add(Conv2D(384, kernel_size=(3, 3), padding='same', activation='relu'))

        self.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
        self.add(MaxPooling2D(pool_size=(3, 3), strides=2))

        self.add(Flatten())

        self.add(Dense(4096, activation='relu'))
        self.add(Dropout(0.5))

        self.add(Dense(4096, activation='relu'))
        self.add(Dropout(0.5))

        self.add(Dense(num_classes, activation='softmax'))


if __name__ == "__main__":
    input_shape = (224, 224, 3)
    num_classes = 1000

    model = AlexNet(input_shape, num_classes)
    model.summary()
