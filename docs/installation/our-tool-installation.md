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

NeatMS can be installed through the pip command, pypi is the reference Python package manager. We also strongly recommend the use of a [vritual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) (Cheat sheet available below). Before installing NeatMS, please make sure that you have the latest version of pip installed.

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
* jupyter-dash

Just type: 

``` bash
pip install notebook dash jupyter-dash
```

## Python virtual environment cheat sheet

Here are simple instructions to help you get started with python virtual environments, please refer to the [official documentation](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) if you encounter issues.

We will use the reference Python package manager (pip) which comes with python, and we will first need to make sure that we have the latest version install. Since NeatMS only supports python 3.6 or higher, we will use the `venv` module to create a virtual environment, which also comes by default with python.

Below you will find the instructions for the different operating systems.

### Windows

First, let's upgrade pip (your version will be checked automatically):

```
py -m pip install --upgrade pip
```

Now let's create a virtual environment, we will call it `neatms-env` here but you can give it the name you want.

> *The virtual environment will be created in your current location* 

```
py -m venv neatms-env
```

You should now see a new folder called `neatms-env`.

We can now activate our virtual environment.

```
.\neatms-env\Scripts\activate
```

You are now ready to install NeatMS using the command provided above.

### Linux and macOS

Linux and macOS often have several versions of python installed, we need to make sure that we use python 3 when we upgrade pip and create a virtual environment.

```
python3 -m pip install --user --upgrade pip
```

You can check pip version using the command below, it will also give you the python version, make sure it is 3.6 or above.

```
python3 -m pip --version
```

Now let's create a virtual environment, we will call it `neatms-env` here but you can give it the name you want.

> *The virtual environment will be created in your current location* 
 
```
python3 -m venv neatms-env
```

You should now see a new folder called `neatms-env`.

We can now activate our virtual environment.

```
source neatms-env/bin/activate
```

You are now ready to install NeatMS using the command provided above.