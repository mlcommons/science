# STEMDL classification benchmark

## 1. Introduction

This folder contains the Pytorch-Lightning implementation of STEMDL
classification, developed at STFC.  The STEMDL classification
benchmark represents a machine learning model which can classify CBED
patterns obtained from crystals into one of the 230 crystallographic
space groups. The benchmark uses resnet50, it is written in Pytorch
Lightning and enables distributed learning on multiple GPUs. The
source code is a single Python program "stemdl_classification.py"
which reads the configuration file "stemdlConfig.yaml" containing the
values of parameters. The sequence of steps performed by the benchmark
can be described as follows:

1. read the configuration file (`stemdlConfig.yaml`)
2. read the datasets from the training, validation, testing and
   inference directories as specified in `stemdlConfig.yaml`
3. train the model
4. test the model
4. perform inferencing
5. measure the time of trainig per epoch and time of a single inference operation
6. save the time measurements in `log_file` (see `stemdlConfig.yaml`)
7. save MLCOmmons logging in `mlperf_logfile` (see `stemdlConfig.yaml`)

## 2. Datasets

Before running the benchmark all datasets must be downloaded from the
remore server. The dataset used by the benchmark is about `35` GB
which is split into four folders: training (`28` GB, `148006` files),
validation (`3.8` GB, `20401` files), testing (`1.8` GB, `9374` files)
and inference (`1.8` GB, `9375` files).

The datasets can be downloaded from remote server by using this command:

```bash
$ aws s3 --no-sign-request --endpoint-url https://s3.echo.stfc.ac.uk sync s3://sciml-datasets/ms/stemdl_ds1a ./
```

The target directory in this case is the local one "./", but any other
directory path with a write permission can also be provided.  As a
result of the aws command four directories will be created: training,
testing, validation and inference.

## 3. Installation

It is recommended to run the Stemdl benchmark in the Anaconda environment.

1. If Anaconda is not already installed on the system, it can be
   downloaded from here:
 
   ```bash
   $ wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
   ```

2. Install Anaconda

   ```bash
   $ bash Anaconda3-2021.05-Linux-x86_64.sh`
   ```
   
3. Create conda environment

   ```bash
   $ conda create --name bench python=3.8
   ```

4. Activate environment

   ```bash
   conda activate bench`
   pip install pytorch-lightning
   pip install torchvision
   pip install scikit-learn
   ```

5. For installing the MLCommons logging library please follow the
   instructions at `https://github.com/mlcommons/logging`

   ```bash
   git clone https://github.com/mlperf/logging.git mlperf-logging`
   pip install -e mlperf-logging
   ```

## 4. Running the benchmark

```bash
$ python sytemdl_classification.py --config stemdlConfig.yaml
```

