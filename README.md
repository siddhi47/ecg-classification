# ecg-classification

This is a collection of scripts for dataset preparation, training and validation. The following scripts are available:

- dataset preparation
- training
- testing

To run the scripts you need to install few dependencies.

**NOTE: Make sure that you use a virtual environment to install these packages.**


## Install torch

```
pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

Install other requirements

```
pip install -r requirements

```


## Run the script to prepare the dataset.
You may want to write your own data preparation script for this as your dataset might be in a different format than this.

This script assumes that you have a reference file with the following columns:
- file path
- labels

## To run the script just type in

```
python ecg_classification/data_preparation.py -src <dataset location> -dst <output folder> -ref <reference file>

```

## To train the model

```

python ecg_classification/train.py -data <dataset location
```

You may also check what other options are available

```
python ecg_classification/train.py --help

```

## To test the model

```
python ecg_classification/test.py 

```

For further help

```
python ecg_classification/test.py --help
```

