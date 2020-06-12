import tensorflow as tf
from tensorflow import keras

class VGG16:
  def __init__(self, input_shape=(224, 224, 3), classes=1000):
    self.model = None
    self.input_shape = input_shape
    self.classes = classes

  def build_model(self):

    model = keras.models.Sequential([
        #Layer 1
        keras.layers.Conv2D(64, kernel_size=(3, 3), strides=1, padding='same',
                            input_shape=self.input_shape, activation='relu'),

        #Layer 2
        keras.layers.Conv2D(64, kernel_size=(3, 3), strides=1, padding='same',
                            activation='relu'),

        #Pooling Layer
        keras.layers.MaxPool2D((3, 3), strides=2),

        #Layer 3
        keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1, padding='same',
                            activation='relu'),

        #Layer 4
        keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1, padding='same',
                            activation='relu'),

        #Pooling Layer
        keras.layers.MaxPool2D((3, 3), strides=2),

        #Layer 5
        keras.layers.Conv2D(256, kernel_size=(3, 3), strides=1, padding='same',
                            activation='relu'),

        #Layer 6
        keras.layers.Conv2D(256, kernel_size=(3, 3), strides=1, padding='same',
                            activation='relu'),

        #Pooling Layer
        keras.layers.MaxPool2D((3, 3), strides=2),

        #Layer 7
        keras.layers.Conv2D(512, kernel_size=(3, 3), strides=1, padding='same',
                            activation='relu'),

        #Layer 8
        keras.layers.Conv2D(512, kernel_size=(3, 3), strides=1, padding='same',
                            activation='relu'),

        #Layer 9
        keras.layers.Conv2D(512, kernel_size=(3, 3), strides=1, padding='same',
                            activation='relu'),

        #Pooling Layer
        keras.layers.MaxPool2D((3, 3), strides=2),

        #Layer 10
        keras.layers.Conv2D(512, kernel_size=(3, 3), strides=1, padding='same',
                            activation='relu'),

        #Layer 11
        keras.layers.Conv2D(512, kernel_size=(3, 3), strides=1, padding='same',
                            activation='relu'),

        #Layer 12
        keras.layers.Conv2D(512, kernel_size=(3, 3), strides=1, padding='same',
                            activation='relu'),

        #Pooling Layer
        keras.layers.MaxPool2D((3, 3), strides=2),

        #Flatten Layer (or 13)
        keras.layers.Flatten(),

        #Layer 14
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),

        #Layer 15
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),

        #Output Layer (or 16)
        keras.layers.Dense(self.classes, activation='softmax')
    ])

    self.model = model
    return self.model

  def compile_model(self, learning_rate=0.1):
    sgd = keras.optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    self.model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return self.model

  def train_model(self, x_train, y_train, batch_size=32, epochs=10, val_data=None):

    history = self.model.fit(x_train, y_train, batch_size=batch_size,
                     epochs=epochs, validation_data=val_data)
    return history


  def predict(self, image):
    prediction = self.model.predict(image)
    return prediction

  def model_summary(self):
    self.model.summary()

  def plot_model(self, name='model.png'):
    return keras.utils.plot_model(self.model, 'model.png')
