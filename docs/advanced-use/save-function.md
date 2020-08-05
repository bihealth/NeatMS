# Saving & loading experiments

## The save function

Labelling manually thousands of peaks is time consuming and will require several hours, losing all your hard work to create a training dataset would be a shame. The saving function is intended for this purpose exclusively and we recommend not to use it otherwise. The reason behind is that it has not been optimzed for saving large project with hundreds of samples.

So saving your project when labelling peaks will give you the freedom to preform this manual task over several days without risking to lose any data.

To save your project, simply call the saving method of the experiment, it will be saved as a .pkl file (python pickle). The file will be created  under the experiment name, in the current directory. You can obviously change the experiment name before saving.

``` python
# Avoid spaces in the experiment name
experiment.name = "my_training_experiment"
experiment.save()
```

> *Note that if you decided to save a large experiment against our recomendation, that's fine, everything should work but may take a lot longer (several minutes to save and load). The pickle file is likely to be very large too! Also, if your experiment contains a lot of samples and peaks, you may hit the system recursion depth limit while creating the pickle object. To circumvent this, you will have to manualy set the recursion limit to a high value using this command `sys.setrecursionlimit(<recursion_limit_value>)`.* 

## Loading an experiment

Loading an experiment is as easy as loading any pickle object. 

``` python
pkl_file = "my_training_experiment.pkl"
with open(pkl_file, 'rb') as f:
    experiment = pickle.load(f)
```

You can then carry on by lauching the annotation tool again, or move on to the next step and train the model. 

> *Important note: the Neural Network module (NN_handler) and Annotation tool of NeatMS are not saved with the experiment object, do not use this function to save your trained model as you will lose it (see next sections).*