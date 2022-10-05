# CloudMask benchmark notes

Sea and land surface temperatures (or SST/LST), have a significant
influence on the Earth’s weather.  For instance, large variations of
the SST in the Pacific can cause anything from severe drought, to
heavy rainfall, to tropical cyclones. Estimation of Sea Surface
Temperature (SST) from space-borne sensors, such as satellites, is
crucial for a number of applications in environmental
sciences. Satellites are often equipped with special sensors for this
purpose, such as the Sea and Land Surface Temperature Radiometer on
board the Sentinel-3 satellite, a mission operated jointly by the
European Space Agency (ESA) and by the European Organisation for the
Exploitation of Meteorological Satellites (EUMETSAT). It is possible
to make direct measurements of surface temperature from these
satellites everywhere, except when clouds are present.  Clouds can
really affect the signals measured by satellites so it is then much
harder to retrieve the temperature measurements. One of the aspects
that underpins the derivation of SST is cloud screening, which is a
step that marks each pixel of thousands of satellite images as
containing cloud or clear sky, historically performed using either
thresholding or Bayesian methods.

## Scientific Objective

The scientific objective of CloudMask is to develop a segmentation model 
for classifying the pixels in satellite images. This classification allows 
to determine whether the given pixel belongs to a cloud or to a clear sky. 
The benchmark can be considered as both training and inference focused, 
where the science metric is same as the classification accuracy — number 
of pixels classified correctly. The performance metric, can be inference 
timing and scalability on the training across a number of GPUs.

## Suggestions for improvement
In its present form, the CloudMask benchmark is set as a supervised learning 
problem, with cloud images are treated as inputs. However, like all science 
cases, the “true” ground truth (or labels), are never available for this case. 
Hence, the benchmark uses Bayesian masks, supplied by the provider of satellite 
images, as the ground truth. We believe that in the absence of any ground truth 
this is a valid choice. However, with Bayesian masks not always being accurate 
or not offering a gold-standard for masks, the resulting model is likely to suffer 
from learnability issues, which sets the perfect challenge for an ML-driven case. 
As a further development of the CloudMask benchmark we’d suggest to explore the 
possibility of using unsupervised segmentation techniques.

The current CloudMask reference implementation is variation of the U-Net deep neural network, 
implemented using TensorFlow and Keras, with the support for distributed training using 
TensorFlow’s native library, Distributed Mirrored Strategy. The network consists of 39 
layers with two million trainable parameters. As a further improvement we'd suggest to 
experiment with different networks which can improve not only accuracy but also reduce 
the time required for training and inference.

## Benchmark files

The CloudMask benchmark consists of the following files:
`slstr_cloud.py`,
`data_loader.py`,
`model.py`,
`cloudMaskConfig.yaml`.

The main program is `slstr_cloud.py` which reads
`cloudMaskConfig.yaml` containing configuration details.  The
benchmark also produces a log file with information about the
parameters and runtime of training and inference.


## Installation

It is recommended to run the CloudMask benchmark in the Anaconda
environment where the required packages can be easily installed and
the versioning of libraries can be maintained. For the installation
use these sequence of instructions:

1. If Anaconda is not already installed on the system, it can be
   downloaded from here:

   ```bash
   $ wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
   ```
 

2. Install Anaconda

   ```bash
   $ bash Anaconda3-2021.05-Linux-x86_64.sh
   ```

3. Create conda environment


   ```bash
   $ conda create --name bench python=3.8
   $ conda activate bench
   $ pip install tensorflow-gpu
   $ pip install scikit-learn
   $ pip install h5py
   $ pip install pyyaml
   ```
   
4. For installing the MLCommons logging library please follow the
   instructions at <https://github.com/mlcommons/logging>

   ```bash
   $ git clone https://github.com/mlperf/logging.git mlperf-logging
   $ pip install -e mlperf-logging
   ```

## Clone the Source

```bash
git clone https://github.com/mlcommons/science.git
cd science/benchmarks/cloudmask
```
   
## Data

The dataset is about 180GB and made up of two parts: reflectance and
brightness temperature. The reflectance is captured across six
channels with the resolution of 2400 x 3000 pixels, and the brightness
temperature is captured across three channels with the resolution of
1200 x 1500 pixels. The training files are in the "one-day" folder
(163GB) and the files used for inferencing are in "ssts" folder
(17GB), see `cloudMaskConfig.yaml`.The datasets can be downloaded from
the STFC server by using these commands:

```bash
mkdir -p data/ssts
mkdir -p data/one-day
pip install awscli
aws s3 --no-sign-request --endpoint-url https://s3.echo.stfc.ac.uk sync s3://sciml-datasets/es/cloud_slstr_ds1/one-day ./data/one-day
aws s3 --no-sign-request --endpoint-url https://s3.echo.stfc.ac.uk sync s3://sciml-datasets/es/cloud_slstr_ds1/ssts ./data/ssts
```


## Running the benchmark

### General python command 

This will run the code on the current server

TensorFlow automatically detects the available GPUs and runs the
application in a data parallel mode.  For running the benchmark use
this command:

```bash
$ python slstr_cloud.py --config ./cloudMaskConfig.yaml
```

### Run cloudmask on Rivanna in batch mode

Rivanna is a HPC at University of Virginia. The documentation here serves as an example on how to run it on other machines.

Note: that the program needs more than 150GB data storage, so it is advised to set it up under /projects. 

```bash
$ sbatch target/rivanna.sh
```

The system log files will be located in the directory you started the sbatch command. Other log files are specified in the yaml file.

### Rivanna interactive node 

To run it in an interactive node use

```bash
srun --gres=gpu:1 --pty --mem=64G --time 01:59:00 /bin/bash conda activate BENCH 
```
Once you get the interactive node run the following commands

```bash
module load singularity tensorflow/2.8.0
module load cudatoolkit/11.0.3-py3.8
module load cuda/11.4.2
module load cudnn/8.2.4.15
module load anaconda/2020.11-py3.8

conda create --name MLBENCH python=3.8
source activate MLBENCH
# conda activate MLBENCH

pip install -r requirements.txt
# pip install tensorflow-gpu
# pip install scikit-learn
# pip install h5py
# pip install pyyaml
python slstr_cloud.py --config ./cloudMaskConfig.yaml
```

or

```bash
ijob --gres=gpu:1 --pty --mem=64G --time 01:59:00 /bin/bash conda activate BENCH 
```
Once you get the interactive node run the following commands

```bash
module load singularity tensorflow/2.8.0
module load cudatoolkit/11.0.3-py3.8
module load cuda/11.4.2
module load cudnn/8.2.4.15
module load anaconda/2020.11-py3.8

conda create --name MLBENCH python=3.8
source activate MLBENCH
# conda activate MLBENCH

pip install tensorflow-gpu
pip install scikit-learn
pip install h5py
pip install pyyaml
python slstr_cloud.py --config ./cloudMaskConfig.yaml
```

### Running the code on Pearl

Pearl (at STFC UK) is an NVIDIA DGX-2 machine with 32 NVIDIA V100 GPUS. For running CloudMask in an interactive mode
the following sequence of commands need to be executed:

```bash
Clone github repository
git clone https://github.com/mlcommons/science/
cd ./science/benchmarks/cloudmask

Create conda environment:
conda create --name mlcommons python=3.8
conda activate mlcommons
pip install tensorflow-gpu
pip install scikit-learn
pip install h5py
pip install pyyaml

Install MLCommons logging
git clone https://github.com/mlperf/logging.git mlperf-logging
pip install -e mlperf-logging

Allocate GPU and time
srun --gres=gpu:1 --pty --mem=64G --time 01:59:00 /bin/bash
conda activate mlcommons

Run the code
python slstr_cloud.py --config ./cloudMaskConfig.yaml
```

### Running the code on Summit

The Summit machine at ORNL operates in a batch-mode. For each task we need to write a job-file, for example "cloudMask.job" and submit it by using the: "bsubmit cloudmask.job" command. In the jobfile the "ProjectCode" should be replaced with a valid code which is accepted by the system. The nodes and GPUs are allocated by the -n and -g flags of the jsrun command. For example -g1 allocates 1 GPU and -n 1 node. It is importangt to notice that the number of GPUs and nodes in the "cloudMaskConfig.yaml" file must match the flags in the jsrun command. For example if we requested 24 GPUs in the jsrun we use -n4 -g6 i.e. 4 nodes and 6 GPUs per node.

```bash
#!/bin/bash
#BSUB -W 1:59
#BSUB -nnodes 1
#BSUB -P ProjecCode
#BSUB -o cloud.o%J
#BSUB -J cloudJobx

# Load modules
module load open-ce

# Install libraries
pip install scikit-learn
pip install h5py
pip install pyyaml

# Install MLCommons logging
git clone https://github.com/mlperf/logging.git mlperf-logging
pip install -e mlperf-logging

#This runs on many nodes
echo "Hostname: "
jsrun -n1 -r1 -c1 hostname
echo "Running slsts on GPU=1"
jsrun  -n1 -a1 -r1 -c1 -g1 python slstr_cloud.py --config ./cloudMaskConfig.yaml
```







