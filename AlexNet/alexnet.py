import tensorflow as tf
from tensorflow import keras

#AlexNet model
class AlexNet:
  def __init__(self, input_shape=(227, 227, 3), classes=10):
    self.model = None
    self.input_shape = input_shape
    self.classes = classes

  def build_model(self):
    model = keras.models.Sequential([
          #First Layer
          keras.layers.Conv2D(input_shape=self.input_shape, filters=96, kernel_size=(11, 11),
                              strides=(4, 4), padding='valid', activation='relu'),
          keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'),

          #Second Layer
          keras.layers.Conv2D(filters=256, kernel_size=(11, 11), strides=(1, 1),
                              padding='valid', activation='relu'),
          keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),

          #Third Layer
          keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1),
                              padding='valid', activation='relu'),

          #Fourth Layer
          keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1),
                              padding='valid', activation='relu'),

          #Fifth Layer
          keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),
                              padding='valid', activation='relu'),
          keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),

          keras.layers.Flatten(),

          #Sixth Layer
          keras.layers.Dense(4096, activation='relu'),
          keras.layers.Dropout(0.4),

          #Seventh Layer
          keras.layers.Dense(1000, activation='relu'),
          keras.layers.Dropout(0.4),

          #Eighth Layer
          keras.layers.Dense(1000, activation='relu'),
          keras.layers.Dropout(0.4),

          #Output Layer
          keras.layers.Dense(self.classes, activation='softmax')
    ])

    self.model = model
    return self.model

  def compile_model(self):
    self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam',
                  metrics=['accuracy'])
    return self.model

  def train_model(self, x_train, y_train, batch_size=32, epochs=10, val_data=None):
    if val_data != None:
      history = self.model.fit(x_train, y_train, batch_size=batch_size,
                     epochs=epochs)
    else:
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
