# MLCommons Science Earthquake benchmark

## Acknowledgements and References

The following people were instrumental for the development of this
benchmark

* Geoffrey C. Fox
* Gregor von Laszewski
* Robert Knuuti
* Thomas Butler
* Jake Kolesar

A paper about the firs results are published at

* [1] Benchmarking for Science: Efforts from the MLCommons Science
  Working Group Jeyan Thiyagalingam, Gregor von Laszewski, Junqi Yin,
  Murali Emani, Juri Papay, Gregg Barrett, Piotr Luszczek, Aristeidis
  Tsaris, Christine Kirkpatrick, Feiyi Wang, Tom Gibbs, Venkatram
  Vishwanath, Mallikarjun Shankar, Geoffrey Fox, Tony Hey, ISC
  Workshop ISC High Performance 2022 Workshop: HPC on Heterogeneous
  Hardware (H3) <http://www.icl.utk.edu/~luszczek/conf/2022/h3/> May
  29 - June 2, 2022, URL: TBD

A paper about the scientific aspects is published at 

* [2] Fox, G.C.; Rundle, J.B.; Donnellan, A.; Feng, B. Earthquake
  Nowcasting with Deep Learning. GeoHazards 2022, 3,
  199-226. https://doi.org/10.3390/geohazards3020011

The system uses cloudmesh-sbatch for managing hyperparameter sweeps

* [3] Gregor von Laszewski, Robert Knuuti, cloudmesh-sbatch,
  <https://pypi.org/project/cloudmesh-sbatch/>

The system uses cloudmesh-gpu for energy and temperature monitoring of
NVIDIA GPUS

* [4] Gregor von Laszewski, cloudmesh-gpu,
  <https://pypi.org/project/cloudmesh-gpu/>

## Source and Development Version

The develoment version if this benchmark is located at

<https://github.com/laszewsk/mlcommons.git>

Before updating the benchmark here, we first coordinate the new code
in the development github, so that we can test it on various platforms
and computers.  If you have improvement suggestion, please get in
contact with Gregor von Laszewski who coordinates the updates and to
create appropriate branches.


## Earthquake TFT Model


## Background

The background of the science is explained in [2]. The first
performanne results are published in [1]. The code uses cloudmesh
sbatch and gpu [3][4].

## System Setup

This benchmark has predefined experiments for the following system
configurations:

* [NVidia DGX Workstation](./systems/dgxstation/README.md)
* [University of Virginia Rivanna](./systems/rivanna/README.md)

It is advised that you follow the previous instructions if you plan to
run this benchmark on these systems as it may take considerable
resources.

Note that the NVidia DGX Workstation configuration can work on any
Ubuntu workstation that has CUDA, cuDNN, and python 3.8+ installed.

For the gathering of the benchmark running a parameter study, we are
useing cloudmesh-sbatch
(https://github.com/cloudmesh/cloudmesh-sbatch.git), providing us with
an easy to configure system producing experirment results on single
computers, as well as supercomputers while using batch queing
systems. At this time SLURM and regular sh job submission is
supported, while an LSF port is under development.

### Custom Execution

To run the MLCommons Science Earthquake TFT notebook, there are a few
prerequesits we assume about your system:

1. We assume that you are running on Linux-like workstation with posix
   tools availible.  (Git-Bash on windows is untested)
2. We assume you have a modern version of python installed, and python
   3 is exposed as the `python` command.
3. We assume you have installed the NVIDIA CUDA drivers and cuDNN
   libraries.
4. We assume you have cloned the following repositories:
   * the MLCommons-Science repository <url-here>.
   * the earthquake dataset <url-here>.

#### Establishing your python environment

First establish a virtual enviornment and install all the requirements
as defined in the [requirements.txt](./requirements.txt) file.

  
```bash
python -m venv venv.earthquake
source venv.earthquake/bin/activate
python -m pip install -r requirements.txt
```

#### Configure Hyperparameters

Make a copy of the [config.yaml.tmpl](./config.yaml.tmpl) and name it
as `config.yaml`.  This file has defaults for a simple model parameter
specification.  See the inline comments that explain the
purpose of each parameter and for how you can configure them.

Take note of the configurations you set for:

* meta.uuid
* run.workdir
* run.datadir

You will need these values when setting up the data for the model.

#### Setting up the data

Extract the `data.tar.xz` file located from the earthquake dataset
repository so that the files are positioned in the directory of
`run.workdir/<username>/workspace-0`.  This can be done automatically
by running the next script (assuming the data.tar.xz is in the
current directory). The data is automatically retrived from the
repository (https://github.com/laszewsk/mlcommons-data-earthquake).


```bash
META_UUID="0"
RUN_WORKDIR="workspace"
MYUSER="$(whoami)"
#---
RUN_BASE="${RUN_WORKDIR}/${MYUSER}/workspace-${META_UUID}"
mkdir -p $RUN_BASE
tar -xf data.tar.xz -C $RUN_BASE
```

#### Running the notebook

If all the previous seteps have been executed, you should now be able
to run the notebook interactively using jupyterlab or as a batch
execution using papermill.

```bash
papermill "FFFFWNPFEARTHQ_newTFTv29-gregor-parameters-fig.ipynb" "output.ipynb"
```

This will create a new notebook named `output.ipynb` that contains the
resulting Jupyter notebook. Additional output, such as images, logs,
and checkpoints can be obtained in the folder
`$RUN_BASE/data/EarthquakeDec2020/Outputs`.

**Caution**: The papermill command can take **multiple hours**,
even days to execute dependent on which GPU you use.
To get an estimate of the runtime to expect plase see paper [1].

It is advised that you run this command on a system that will not
experience any forced logouts or premature terminations of your
desktop session.  You may use the `nohup` command to launch the
command in the background, but note that all output will be logged to
the file `nohup.out`, so feedback will be limited.

**Optional**: If you are interested in reviewing the lower-level GPU
logging, immediately before running the papermill process, execute

```
cms gpu watch --gpu=0 --delay=1 --dense > ${RUN_BASE}/data/EarthquakeDec2020/Outputs/gpu0.log &
```

This will start a background task to monitor your GPU's compute load
throughout the execution of the notebook. Note that you will need to
kill this process manually after papermill is completed.
Please note that `cms gpu` is an extension to cloudmesh allowing you
to monitor resource usage on the GPU such as energy and temperature.
More deatils can be found at [4].

