# Getting started

## Import NeatMS

After installation, you should be able to import NeatMS.

``` python
import NeatMS as ntms
```

## Getting help

Like any python object, you can inspect NeatMS objects through the `help` function.

``` python
>>> import NeatMS as ntms
>>> help(ntms.Experiment)

Help on class Experiment in module NeatMS.experiment:

class Experiment(builtins.object)
 |  Simple representation of MS experiment.
 |  
 |  Contain the data and meta-data relative to an experiment, 
 |  no metadata specific to the experiment design is stored (such as biological groups).
 |  The general architecture of the data files (location and list of the raw data files and feature file) is stored within this class.
 |  The raw data (spectra, chromatogram and features) is stored in objects of type Sample and 
 |  FeatureTable directly accessible through their respective lists.
 |  
 |  Methods defined here:
 
 [...]
```

For standard usage, we recommend to follow the basic and/or advanced tutorial provided as jupyter notebooks on [NeatMS github repository](https://github.com/bihealth/NeatMS/tree/master/notebook/tutorial). These tutorials cover all necessary commands to import data, run prediction and export results. The advanced tutorial explains how to train the neural network model from scratch or via transfer learning. 

The library also allows changes in the data and neural network structure for users with deep learning experience. For this advanced use of the library, please refer to the advanced documentation section as no tutorial currently covers this. 

