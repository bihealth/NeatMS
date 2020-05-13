# Getting started

## Import NeatMS

After installation, you should be able to import NeatMS as a package

``` python
import NeatMS as ntms
```

## Getting help

Like any python objects, you can inspect NeatMS objects through the `help` function.

``` python
>>> import NeatMS as ntms
>>> help(ntms.Experiment)

Help on class Experiment in module lcms_data_rawdata_support:

class Experiment(builtins.object)
 |  Simple representation of MS experiment.
 |  
 |  Contain the data and meta-data relative to an experiment, no metadata related to the experiment design is stored.
 |  The location and list of the raw data files and feature file are stored within this class.
 |  The raw data (spectra, chromatogram and features) is stored in objects of type Sample and FeatureTable directly accessible through their respective lists.
 |  
 |  Methods defined here:
 
 [...]
```

However, for standard usage, we recommend to follow the basic tutorial and/or advanced tutorial provided as jupyter notebooks on [NeatMS github repository](https://github.com/bihealth/NeatMS). This tutorial cover all necessary commands to import data, run prediction and export results. The advanced tutorial explains how to train the neural network model from scratch or via transfer learning.

