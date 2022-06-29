# Set up system tools

* We assume you have python 3.10.4 installed 
* We assume if you like to use the automated report generated (under
  development) you have full latex installed

  
```bash
python3.10 -m venv ~/ENV3
source ~/ENV3/bin/activate
mkdir ~/cm
cd ~/cm
pip install cloudmesh-installer
cloudmesh-installer get sbatch
cms help
```

## Generating experiment configurations

1. Choose a `$PROJECT_DIR` where you wish to store the code and
   output.  The location must have enough storage to run the full
   experiment (average of 15GBs per single experiment).

2. Choose your desired `$EQ_VERSION`.  This must be the name of the
   baseline folder in the code repository.

3. Clone the repository, using the below commands


```bash
export PROJECT_DIR=/project # Update to your desired project path
export EQ_VERSION=latest # Update to your desired baseline of code.
mkdir -p ${PROJECT_DIR}
cd ${PROJECT_DIR}
git clone ssh://git@github.com/laszewsk/mlcommons.git 
cd mlcommons/benchmarks/earthquake/${EQ_VERSION}/experiments/dgxstation
```

## Generate scripts

Note that all script generation is driven by the `dgx-config.yaml`
file.  Most hyperparameters that can be updated are located within
that file as part of the config or experiments section.  This is the
best place to make modifications as these parameters were desiged to
propagate throughout the execution and provides a single configuration
point for tuning these hyperparameters.

### Using Makefile

1. Issue the command `make dgx` to generate the configuration settings
   based on `dgx-config.yaml`'s experiment settings.

   **NOTE**: It is important that this command is run in isolation of
   all subsequent make commands.  Failure to do so will prevent make
   from identifying all the experiment configurations.

2. Review the output located in the `dgx` folder and confirm that
   scripts and configurations appear as they should (that is, the
   correct number of experiments are generated and no unexpanded
   variables are present).

3. Run a specific epoch case by typing `make run-<epoch>
   CUDA_VISIBLE_DEVICES=<gpu>`, where `<epoch>` is the epoch count you
   wish to execute, and `<gpu>` is the CUDA visible GPU specification
   as defined in the
   [CUDA Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars).

4. Note, this command will run in the background, you can monitor the
   progress of the command by typing `make watch` or by inspecting the
   `nohup.out` file that is saved in the
   `dgx/<experiment_configuration>` directory.


### Using manual commands

Note that it is recommended you target the makefile execution route
unless you're looking to have additional control over the experiment
management.

1.  Execute the cms sbatch generation script, providing the input
    script and configuration files as shown below

    ```bash
	cms sbatch generate dgx.in.slurm --setup=dgx-config.yaml --name=dgx
	```

2. Note that the DGX subsystems do not target slurm, so unlike the HPC
   executions, we do not need to generate a submission script.

3. Navigate to your desired configuration in
   `dgx/<experiment_configuration>`.

4. To launch, run the following commands

```bash
# Update to limit what devices the notebook will use with Tensorflow
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
# Replace <experiment_configuration> with your desired experiment to run
cd dgx/<experiment_configuration>
nohup bash slurm.sh &
```

5. Note that the nohup command will not immediately return your input
   prompt.  You can simply press enter after its notification on
   output redirection.

6. All stdout and stderr logging will now be printed to nohup.out.
   Make sure the process spawned by reviewing the output of this file
   using `less nohup.out` or `tail -f nohup.out`


## Results

Once the notebook's execution has completed, the output will be
located in the `dgx/<experiment_configuration>/_output` folder.
