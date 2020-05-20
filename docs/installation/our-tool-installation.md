# NeatMS installation

## Requirements

NeatMS requires python 3.6 or higher and a [TensorFlow compatible system](https://www.tensorflow.org/install/pip#system-requirements). 

## Dependencies

These dependencies will be automatically installed with NeatMS package. However, we recommend to install them manually through pip and verify that they are installed and configured properly.

* pymzml
* numpy
* pandas
* scikit-learn
* tensorflow
* pillow
* h5py
* keras

## NeatMS Installation

### Using pypi

NeatMS can be installed through the pip command, pypi is the reference Python package manager. We also strongly recommend the use of a [vritual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/). Before installing NeatMS, please make sure that you have the latest version of pip installed.

``` bash
pip install --upgrade pip
```

``` bash
pip install NeatMS
```
### Using Bioconda

> Note: *Bioconda supports only 64-bit Linux and Mac OS*

Follow the instructions on the [official bioconda documentation](https://bioconda.github.io/user/install.html) on how to install conda, add bioconda channel and create a conda environment. 

You are now ready to install NeatMS.

``` bash
conda install neatms
```

#### For Mac OS users

Depending on your system settings you may have to add this to your `.bash_profile` or enter it in your terminal (this will only work for the current session).

``` bash
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
```

## Extra dependencies for advanced use

To follow the advanced tutorial and train your own model, you need to manually install these extra dependencies available through pip. 

> *Bioconda users can also install the extra libraries through pip within the conda environment.*

* jupyter notebook
* dash
* jupyter\_plotly\_dash

Just type: 

``` bash
pip install notebook dash jupyter_plotly_dash
```

