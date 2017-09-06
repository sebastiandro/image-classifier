import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing import image
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D

img_width = 150
img_height = 150

model = Sequential()
model.add(Convolution2D(32, (3, 3), input_shape=(img_width, img_height,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))

# To prevent overfiting we use Dropout
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


model.load_weights('models/cnn-image-recognition2.h5')

img = image.load_img('happy.jpg', target_size=(150,150))

x = image.img_to_array(img)

x = np.divide(x, 255)

x = np.expand_dims(x, axis=0)

prediction = np.ndarray.flatten(model.predict(x, 1))[0]
prediction_class = np.ndarray.flatten(model.predict_classes(x, 1))[0]

if prediction_class == 1:
    print("I'm %f %% that you are a girl" % (prediction * 100,))
else:
    print("I'm %f %% that you are a boy" % ((1 - prediction) * 100,))