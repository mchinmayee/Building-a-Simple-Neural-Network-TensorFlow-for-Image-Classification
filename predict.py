from tensorflow.keras.preprocessing.image import img_to_array,load_img
from tensorflow.keras.models import load_model
import cv2

from create-model import get_filenames

model_file = '../cats_and_dogs_filtered/cats_vs_dogs_V2.h5'

    
imgDirC = '../cats'
imgDirD = '../dogs'
fileNames =  get_filenames(imgDirC)
fileNames.extend(get_filenames(imgDirD))

from random import shuffle
shuffle(fileNames)

resultList = []
model = load_model(model_file)

for imagePath in fileNames[:10]:
    image = cv2.imread(imagePath)
    # load an image and predict the class
    img = img_to_array(image)
    img = cv2.resize(image,(150,150), interpolation = cv2.INTER_AREA)
    img = img_to_array(img)
    img = img.reshape(1, 150,150, 3)
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]
    result = model.predict([img])

    if np.argmax(result) == 1:
        str_label = 'dog'
    else:
        str_label = 'cat'
            
    if os.path.split(imagePath)[-1][0] == str_label[0]:
        resultList.append(1)
    else:
        resultList.append(0)
        
    print(result[0],': ', str_label)

accuracy = sum(resultList)/len(resultList)*100.0
print('accuracy: {}%'.format(accuracy))