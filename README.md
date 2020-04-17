# Building-a-Simple-Neural-Network-TensorFlow-for-Image-Classification

## V1 Implementation

1. Importing Tensorflow and necessary libraries
2. Preparing the training and testing data
   > > train_dir = '../cats_and_dogs_filtered/train'
   > > test_dir = '../cats_and_dogs_filtered/validation'
   > > model_file = '../cats_and_dogs_filtered/cats_vs_dogs_V2.h5'

> > image_base_dir = '../cats_and_dogs_filtered/'
> > train_data_npz = image_base_dir + '/catdogData/cats_vs_dogs_training_data2.npz'
> > train_label_npz = image_base_dir + '/catdogData/cats_vs_dogs_training_label2.npz'
> > test_data_npz = image_base_dir + '/catdogData/cats_vs_dogs_testing_data2.npz'
> > test_label_npz = image_base_dir + '/catdogData/cats_vs_dogs_testing_label2.npz'

- Creating training and testing sets
  Used rglob() method to find files in a directory. In the program get_filenames( ) returns file names from directory with file extension as ‘jpg’.
- Preprocessing the Image data
  The training and testing datasets are split into two classes with the labels and images stored in different lists. For training and testing image sets used resize_image_train and resize_image_test functions to store images and it’s labels. Used load_data_training_test function to store train and test sets images and labels.
- Initializing the base model
  The base model is the model that is pre-trained. We will create a base model using MobileNet V2 with given input data where the image data is resized to a 150×150 pixel size. We should also prevent the weights of the convolution from being updated before the model is compiled and trained. To do this we set the trainable attribute to false. The function generate_model shows how the MobileNetV2 and Sequential model are used to build a model.
- Compiling the model
  Its time to compile our new model by initializing the right optimizer, loss function and metrics.
  The function generate_model( ) shows the details about the creation of model.
- Training the model
  Finally, it’s time to train the model with the image data we have. Uses some parameters like number of epochs and batch size in model. steps_per_epoch is the number of times the weights are updated for each cycle of training. The ideal value for steps_per_epoch is the number of samples per batch. The model fits with the training and validation data sets with specified parameters.
  On executing the code block train_model( ) function the model actually starts to train. The output of the model shows 90% of test accuracy at 10 number of epochs.

> > Epoch 10/10
> > 2000/2000 [==============================] - 64s 32ms/sample - loss: 0.2978 - acc: 0.8740
> > val_loss: 0.2436 - val_acc: 0.9036
> > 140/140 [==============================] - 3s 23ms/sample - loss: 0.2436 - acc: 0.9036
> > Test loss:0.2435938613223178
> > Test accuracy:0.9035714268684387

### Predicting model for a new image

So far, I have demonstrated how to save models and use them later for prediction, however, this is all boring stuff, the real deal is being able to load a specific image and determine what class it belongs to.
The program predicted for some images of cats and dogs downloaded from Google.

### Conclusion

The above model is able to predict with an accuracy of 60% on unknown data set. Of course, you can try different parameters, architecture, training epochs etc to tune up the model to give a better result. Feel free to do so! Till next time!
