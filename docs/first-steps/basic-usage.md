# Basic usage

The code discussed in this section of the documentation is the same as the basic tutorial available on [NeatMS github repository](https://github.com/bihealth/NeatMS). 

## Import data

The main class of our module is the Experiment. Creating an experiment object will allow us to load the input data and access every function from the module.

In order to create an experiment object, we need to set 2 parameters:
 
* The path to the raw data folder
* The path to the feature table (.csv)

``` python
raw_data_folder_path = 'path/to/raw_data/folder'
feature_table_path = 'path/to/feature_table'
```

We can now create an experiment which will automatically load the raw data and the features, and structure the information so we can explore it easily.

``` python
experiment = ntms.Experiment(raw_data_folder_path, feature_table_path)
```

## First data exploration

Now that our experiment object is created, we can make use of the data structure within the module to explore the data before going further and label the peaks. For examples, here is a code snippet to get some closer look at the number of features detected in our dataset, and the number of samples they are present in. 

``` python
from  collections import Counter
exp = experiment
sizes = []
print("# Feature collection:",len(exp.feature_tables[0].feature_collection_list))

for consensus_feature in exp.feature_tables[0].feature_collection_list:
    sizes.append(len(consensus_feature.feature_list))

c = Counter(sizes)
print("Number of consensus features:")
for size, count in c.most_common():
    print("   of size %2d : %6d" % (size, count))
print("        total : %6d" % len(exp.feature_tables[0].feature_collection_list)) 

Feature collection: 13852
Number of consensus features:
   of size  3 :   2311
   of size  6 :   1941
   of size  4 :   1555
   of size  5 :   1275
   of size 20 :   1026
   ...
``` 

A feature collection represent a list features that have been aligned between each other (found in one or more samples).

## Load a Neural network model.

First we need to create an instance of the `NN_handler` (Neural Network Handler) class. This object will allow us to interact with the neural network model, it provides methods to perform actions on and with the neural network and link it to the data through our experiment object.

Let's create an instance of `NN_handler`. The only required argument is the experiment object we want to attach to the neural network handler:

``` python
nn_handler = ntms.NN_handler(experiment)
```

We can now load the neural network model that is provided with the tool in the data folder of [NeatMS github repository](https://github.com/bihealth/NeatMS).

``` python
nn_handler.create_model(model="path_to_model.h5")
```

Information about the model architecture can be found using the summary method. This is particularly useful for advanced usage when training the network through transfer learning, it is also a godd way to check whether network created by a third party is appropriate for us. Make sure that the shapes of the first and last layers of the model correspond to the one below.

``` python
>>> nn_handler.get_model_summary()

Model: "model_1"
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

For example, this summary tells us that the expected shape of the input layer is (2, 120, 1), which is the default shape used by NeatMS. That means that we are good to proceed to the next step as we are using default values for all arguments and parameters.

## Perform the prediction

Before we can perform the prediction, we have to set a threshold value between 0 and 1. This value is model specific and is determined during the training process, hence, there cannot be any default values. The optimal value for the default model that we provide is 0.22, a higher value would return more accurate but less sensitive results while a lower value would have the opposite effect. More information on the tuning of this hyperparameter can be found in the advanced section.

Finally, the prediction function allows us to select the samples on which we want to perform the prediction. In most cases, we will choose to perform the prediction on the full dataset (meaining all peaks from all samples), so we will leave the default parameters.

``` python
# Set the threshold to 0.22
threshold=0.22
# Run the prediction
nn_handler.predict_peaks(threshold)
```

Although, data handling has been optimised, running the prediction may take some time depending on the number and size of the files in the dataset (about 15 to 30 seconds per sample). The prediction is performed by batches of peaks, one batch corresponding to all peaks present in one sample.

## Explore and export results

Now that the prediction is done, every peak present in the dataset has been labeled with one of the 3 default classes: `High_quality`, `Low_quality`, `Noise`. In this section we will learn how to make use of this information to export the data in deferent ways.

Before learning how to use the export method, let's investigate the results. The snippet below will allow you to get a quick overview of the predictions, which gives us some information about the quality of the peak detection.

```
from  collections import Counter
exp = experiment
hq_sizes = []
lq_sizes = []
n_sizes = []
sizes = []
print("# Feature collection:",len(exp.feature_tables[0].feature_collection_list))
for consensus_feature in exp.feature_tables[0].feature_collection_list:
    hq_size = 0
    lq_size = 0
    n_size = 0
    for feature in consensus_feature.feature_list:
        for peak in feature.peak_list:
            if peak.valid:
                if peak.prediction.label == "High_quality":
                    hq_size += 1
                if peak.prediction.label == "Low_quality":
                    lq_size += 1
                if peak.prediction.label == "Noise":
                    n_size += 1
                    
    hq_sizes.append(hq_size)
    lq_sizes.append(lq_size)
    n_sizes.append(n_size)
    sizes.append(len(consensus_feature.feature_list))
    
c = Counter(hq_sizes)
print("\nNumber of consensus features labeled as 'High quality':")
for size, count in c.most_common():
    print("   of size %2d : %6d" % (size, count))
print("        total : %6d" % len(exp.feature_tables[0].feature_collection_list))

c = Counter(lq_sizes)
print("\nNumber of consensus features labeled as 'Low quality':")
for size, count in c.most_common():
    print("   of size %2d : %6d" % (size, count))
print("        total : %6d" % len(exp.feature_tables[0].feature_collection_list))

c = Counter(n_sizes)
print("\nNumber of consensus features labeled as 'Noise':")
for size, count in c.most_common():
    print("   of size %2d : %6d" % (size, count))
print("        total : %6d" % len(exp.feature_tables[0].feature_collection_list))
```

This code should return something similar to this, for every label:

```
# Feature collection: 13852
Number of consensus features labeled as 'High quality':
   of size  3 :   2311
   of size  6 :   1941
   of size  4 :   1555
   
   [...]
```

This tells us that 2311 features have been labeled as high quality in 3 samples, 1941 in 6 samples and 1555 in 4 samples. The number of peaks labeled as `high quality` in all samples rarely exceed a few hundreds, this is totally normal as a peak missing in one sample or simply presenting a poor shape would not qualify and make this number drop. But it does not mean we are facing bad quality data or that the peak picking did not perform well. For example, if you included blank samples in your experiment, it is likely that you will find very few `high quality` peaks present in all samples, and the ones you will find are probably contaminants anyway. So to make sure that we export the right data, filter out the bad quality data, but also structure our output occording to the statistical analysis we want to perform, it is important to understand all options the export method has to offer!

### Export method

That's the last part of the basic usage section but maybe the most important one. Although default values are already set for us, using those blindy is probably not a good idea. The export method can take many arguments to allow high flexibility and fine tuning on what we want to conserve in our exported results, let's take the time to review them.

> Note: *You can call the export method several times to create results with different degrees of filter, different filters or even different structures*.

The first three arguments `filename`, `index` and `na_rep` correspond to the arguments found in the [csv export method from pandas](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html), `filename` is the only required argument.

The rest of the arguments are all optional, they will allow you to structure and select what goes in the output file.

#### Argument list

**Argument:**

``` python
export_classes
```

**Default values:**

``` python
["High_quality", "Low_quality"]
```

This arguments takes the list of classes that you would like to export. Peaks labelled with one of the classes present in that list will be exported, the rest will be set as missing values. By default, `Noise` labelled peaks are exported as missing values.

---

**Argument:**

``` python
min_group_classes
```

**Default values:**

``` python
["High_quality"]
```

This argument works alongside `min_group_size`. `min_group_classes` allows you to specify the list of classes you want to filter using `min_group_size`, a feature will have to be labelled with `min_group_classes` and appear in `min_group_classes` group classes to be conserved. For example, the following parameters would export all peaks labelled with `High_quality` and `Low_quality` if a minimum of 50% of the samples contain the peak labelled as `High_quality`. `Noise` labelled peaks would be reported as missing values.

``` python
export_classes = ["High_quality", "Low_quality"]
min_group_classes = ["High_quality"]
min_group_size = 0.50
```
---

**Argument:**

``` python
min_group_size
```

**Default values:**

``` python
0.75
```

This argument works alongside `min_group_classes`. `min_group_classes` allows you to specify the list of classes you want to filter using `min_group_size`, a feature will have to be labelled with `min_group_classes` and appear in `min_group_classes` group classes to be conserved. `min_group_size` should be set between 0 and 1.

---

**Argument:**

``` python
exclude
```

**Default values:**

``` python
[]
```

This is the sample exclusion list, the method supports either a list of objects of type NeatMS Sample, or a list of strings containing sample names `Sample.name`. Mixed list is supported too but be avoided.

> Note that this exclusion list will affect the argument `min_group_size`. 

---

**Argument:**

``` python
use_annotation
```

**Default values:**

``` python
False
```

This argument enables you to export the annotation set manualy to a peak instead of its predicted label when available.

> This is only possible if you have trained the neural network yourself and used a subset of the data being analysed for training. 

---

**Argument:**

``` python
export_properties
```

**Default values:**

``` python
["rt", "mz", "height"]
```

List of peak properties to be exported. Every property will appear as one column in the final csv file unless otherwise specified below. Here is the list of possible export properties:

`rt` - Feature retention time as given in the input feature table.

`mz` - Feature *m/z* as given in the input feature table

`height` - Peak height as given in the input feature table. One entry per sample.

`area` - Peak area as given in the input feature table. One entry per sample.

`peak_mz` - Peak *m/z* as given in the input feature table. One entry per sample.

`peak_rt` - Peak retention time as given in the input feature table. One entry per sample.

`peak_rt_start` - Peak start retention time as given in the input feature table. One entry per sample.

`peak_rt_end` - Peak end retention time as given in the input feature table. One entry per sample.

`peak_mz_min` - Peak minimum *m/z* as given in the input feature table. One entry per sample.

`peak_mz_max` - Peak maximum *m/z* as given in the input feature table. One entry per sample.

`label` - Peak label as predicted by the model or manually labelled depending on `use_annotation`. One entry per sample.
