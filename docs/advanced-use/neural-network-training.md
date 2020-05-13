# Neural network training

This section requires to have some basic theoritical knowledge on neural networks but also some practical experience. Although, NeatMS `Neural newtork handler` simplifies greatly the entire training process, it is important that you understand the basic concepts of deep learning so the model that you create will perform well. Keep in mind that having an under performing model could have a big impact on your metabolomics study results and their interpretation.  

## Batch creation

Now that you feel confident with neural network training, let's dive in prepare our batches.
NeatMS provides the necessary functions to do that, all we will have to do is create a `Neural network handler` object and call the batch creation method.

``` python
nn_handler = NN_handler(experiment,  matrice_size=120, margin=1)
```
As you can see we have given two more arguments than we did in the basic tutorial, you can leave them out if you want to use the default values of `120` and `1`. 

The matrix size argument is simple, it represents the number of points used to represent a peak. As for the margin, it represent the size of the surrounding signal that should be kept on the retention time dimesnion. `1` means that the margin size on both side of the peak will be 100% of the peak width. For example, if a peak has a retention time width of 3 seconds, the extracted signal will be of 9 seconds. This is then interpolated to form a matrix of size (1, 120). With the default argument values, the first and last 40 values of the peak will represent the margins (surrounding signal), the middle 40 values represent the peak itself.

A second dimension containing binary values (0 or 1) is then added to represent the margin protion of the signal (0), or the peak signal (1). The resulting matrix size for one peak is therefore (2, 120).

> Note that if you decide to change the matrix size or margin argument, the neural network architecture will be automatically adapted. You will, however, only be able to train your model from scratch, by transfer learning using a preexisting model with the same architecture. Do not change the arguments if you intend to use the default model provided with NeatMS for transfer learning.
 
We can now create our batches of data, 3 batches are required, one for training, one for testing and one for validation. This type of approach is standard will allow us to detect any overfitting, the training and test set will be used during the training process. The validation set remains untouch and is used later on for hyperparameter tuning.

``` python
nn_handler.create_batches(validation_split=0.1, normalise_class=False)
```

By default, the split between training:test:validation batches is 80:10:10, the `validation_split` argument allows you to define the size of test and validation sets. We recommend not to exceed `0.2` as only 60% of the data will be left for training.

The `normalise_class` argument allows you to make sure every class has the same number of peaks for the training, when set to `True`, the number of peaks for each class will be equal to the smallest class.

## Model training

We have 2 options to train the model, training the full model from scratch is the most straightforward but requires a large dataset. Transfer learning, however, may require a few more steps but can be done with a much smaller dataset.

### Full model training

When choosing this option, we recommend that you have at least 500 peaks for each class (or 500 peaks in the smallest class). If you have fewer peaks, you should consider the [transfer learning](#transfer-learning) option instead.

We do not need to load a pre-existing model here, but we can change NeatMS allows changing the learning rate and optimizer. You have the choice between `Adam` (default) and `SGD` optimzers. During development and testing, best performances were achieved using the `Adam` optimizer and a learning rate of `0.00001`, those values are therefore set as default. Make sure to leave the `model` argument as `None`. You are then ready to train the model.

``` python
nn_handler.create_model(lr=0.00001, optimizer='Adam')
nn_handler.train_model(1000)
```

The number of `epochs` can be set when calling the training method (1000 by default). NeatMS does not currently provides callback functions to automatically stop the training. Calling the training method will simply resume the training and train for another specified number of epochs. The network architecture has been built to specificaly prevent overfitting (using dropout and kernel regularizer), however, it is down to you to make sure overfitting does not occur.

Once training done, which will take a few minutes to an hour depending the compouting power you have, you are ready to move on to the [tuning part](hyperparameter-optimization.md). 

### Transfer learning

If you have labelled a few hundreds peaks from your dataset and would like to tune an existing model, this is the right place. Before we dig any further, it is important to understand the archicture of the network we are dealing with. 

The network is made of a convolutional base and a classifier. The convolutional base performs the feature extraction, and the classifier the classification. Here, we will have the option to adjust the entire network, or only train one part of it. 

For example, if I have a very small dataset and I only want to adjust the classification of peaks to what I would manually do, I can make use of the convolutional base which is already trained to extract features and freeze those layers, and fine tune the classifier part only to fit my classification. 

> The part of the network you choose to fine tune is down to you, however, fine tuning the convolutional base will require more data than fine tuning the classifier.

First, we will load the existing model the same way we did in the basic use section.

``` python
nn_handler.create_model(model = "path_to_model.h5")
```

Now that we have our model loaded, we have several options as previously discussed. To train the entire network you can directly call the training function `nn_handler.train_model()`. Otherwise, let see how we can explore the network architecture to select the layers we want to freeze.

The model architecture can be seen like so:

``` python
nn_handler.get_model_summary()

Model: "NeatMS_Model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 2, 120, 1)         0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 2, 120, 32)        832       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 2, 60, 32)         0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 2, 60, 64)         18496     
_________________________________________________________________
flatten_1 (Flatten)          (None, 7680)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               983168    
_________________________________________________________________
dropout_1 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_prediction (Dense)     (None, 3)                 387       
=================================================================
Total params: 1,002,883
Trainable params: 1,002,883
Non-trainable params: 0
_________________________________________________________________
``` 

The model itself is stored in the `nn_handler.class_model` variable, we can select and freeze layers using the following commands (Note that these are Keras functions). 

``` python
# Let's freeze the convolutional base
# We can do that by selecting layers using their names
layer_names = ['conv2d_1','conv2d_2','max_pooling2d_1']
for layer_name in layer_names:
	model.get_layer(layer_name).trainable = False
	
# Or using their position in the network (Here we freeze the first 4 layers)
for layer in nn_handler.class_model.layers[:4]:
	layer.trainable = False
	
# Here is how to make sure that the right layers are still trainable
for layer in nn_handler.class_model.layers:
    print(layer, layer.trainable)
    
<keras.engine.input_layer.InputLayer object at 0x13885b438> False
<keras.layers.convolutional.Conv2D object at 0x138867588> False
<keras.layers.pooling.MaxPooling2D object at 0x138867eb8> False
<keras.layers.convolutional.Conv2D object at 0x1388757f0> False
<keras.layers.core.Flatten object at 0x13ce63ef0> True
<keras.layers.core.Dense object at 0x13ce73080> True
<keras.layers.core.Dropout object at 0x13ce737b8> True
<keras.layers.core.Dense object at 0x13ce73898> True
```

Now, before we start training, we need to compile the model, for that we will have to set the optimizer and the learning rate. Here is how to do that:

``` python
from keras.optimizers import SGD, Adam 
lr = 0.00001
opt = Adam(lr=lr)
nn_handler.class_model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics=['accuracy','mae'])
```
You are now ready to train the model.

``` python
# Remember you can pass the number of epochs as an argument
nn_handler.train_model()
```