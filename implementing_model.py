#importing all the required packages
from PIL import Image
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Flatten,Dense
from keras.utils.np_utils import to_categorical

     #loading all images to a dictionary
images_dir1_dict = {os.path.splitext(os.path.basename(x))[0]: 'C:\\Users\\Admin\\Data\\HAM10000_images_part_1\\'+x
                     for x in os.listdir('C:\\Users\\Admin\\Data\\HAM10000_images_part_1')}
images_dir2_dict2 = {os.path.splitext(os.path.basename(x))[0]: 'C:\\Users\\Admin\\Data\\HAM10000_images_part_2\\'+x
                    for x in os.listdir('C:\\Users\\Admin\\Data\\HAM10000_images_part_2')}}
images_dir1_dict.update(images_dir2_dict2)

#dictionary for describing names in short
disease_dict = {
        'akiec': 'Actinic keratoses',
        'bcc': 'Basal cell carcinoma',
        'bcc': 'Basal cell carcinoma',
        'bkl': 'Benign keratosis',
        'df': 'Dermatofibroma',
        'mel': 'Melanoma',
        'nv': 'Melanocytic nevi',
        'vasc': 'Vascular lesions'
}

#loading the data to dataframe
data_df = pd.read_csv("C:\\Users\\Admin\\Data\\HAM10000_metadata.csv")
data_df['path'] = data_df['image_id'].map(images_dir1_dict.get)
data_df['cell_type'] = data_df['dx'].map(disease_dict.get) 
data_df['cell_type_idx'] = pd.Categorical(data_df['cell_type']).codes

#resizing
dims = (32, 32)
shape = dims + (3,)
data_df['image'] = data_df['path'].map(lambda x: np.asarray(Image.open(x).resize(dims)))

      #dividing the dataset
y = data_df.cell_type_idx
x_train1, x_test1, y_train1, y_test1 = train_test_split(data_df, y, test_size=0.30)

x_train = np.asarray(x_train1['image'].tolist())
x_test = np.asarray(x_test1['image'].tolist())

#scaling
x_train = (x_train)/255
x_test = (x_test)/255

y_train = to_categorical(y_train1, num_classes = 7)
y_test = to_categorical(y_test1, num_classes = 7)

#building CNN model
inp_shape = (32, 32, 3)
no_classes = 7
batch_size = 32 
epochs = 25

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=inp_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(no_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])
model.summary()

#providing dataset to model
history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)

#evaluating model
model.evaluate(x_test, y_test)

#saving the model
model.save("disease_prediction.h5")
