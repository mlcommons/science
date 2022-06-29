#!/usr/bin/env bash
#SBATCH --job-name=mlcommons-eq-{card_name}-{gpu_count}
#SBATCH --output=%u-%j.out
#SBATCH --error=%u-%j.err
#SBATCH --partition={system.partition}
#SBATCH -c {cpu_num}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --gres=gpu:{card_name}:{gpu_count}
#SBATCH --mail-user=%u@virginia.edu
#SBATCH --mail-type=ALL
#SBATCH --account={user.account}

set -uxe

NOTEBOOK_REPO="https://github.com/mlcommons/science.git"
EARTHQUAKE_DATA_REPO="https://github.com/laszewsk/mlcommons-data-earthquake.git"

# This has not yet cms gpu watch into a logfile we want to specify in the config.yaml
# cms gpu watch
# cms gpu watch [--gpu=GPU] [--delay=SECONDS] [--logfile=LOGFILE] [--count=COUNT] [--dense]
# with &

### BEGIN CONFIGURATION

## Specifies the path to save the final output
NB_OUTPUT="_output"

## Unique identifier to isolate executions from each other.
RUN_ID="${RUN_ID:-{meta.uuid}}"

## Set the python version to target
PYTHON_VERSION={system.python}

## Specifies the git branch to use from the repository
## by default using {run.branch}
## The SAFE version cleans the name of any path-like characters
BRANCH=${BRANCH:-{run.branch}}
BRANCH_SAFE=${BRANCH/\//-}

## Specifies a github fork to use when running the build
FORK=${FORK:-"laszewsk"}

## Specifies the primary directory to use when running the
## benchmark.  When not specifies, it assumes the username
## on the sytem as the directory.
MLCOMMONS_HOME=${MLCOMMONS_HOME:-{run.workdir}}

## Specifies a specific revision of the Earthquake notebook
## Used to select the different folder names in the earthquake
## directory
REV="{revision}"

## Specifies the path that contains utilities on the HPC
## cluster, such as the system scripts.
RESOURCE_DIR="{run.resourcedir}"


### END CONFIGURATION


### FUNCTION BEGIN
# Finds the current user's common name from the current operating
# environment.
### FUNCTION END
function DETECT_RUN_USER() {
   # Gets the current username from the current system password database
   # will be an empty string if the command fails or a bad username is
   # supplied

   #local MY_NAME="$(getent passwd $(whoami) | cut -d: -f5 || true)"
   ## When the username is not known or the password database cannot run
   ## set the value to the current system username
   #if [ x"${MY_NAME}" == x"" ]; then
   #     echo $(whoami)
   ## Otherwise, use the human friendly username
   #else
   #     echo ${MY_NAME}
   #fi
   echo $(whoami)
}



## Specifies the username to apply when running on a
## clustrered environment
RUN_USER="$(DETECT_RUN_USER)"

## Sets up the base directory for running the experiment from the
## previously defined vars.
RUN_BASE=${MLCOMMONS_HOME}/${RUN_USER}/workspace-${RUN_ID}
VENV_PATH={run.venvpath}/${RUN_USER}/venv-${RUN_ID}-${PYTHON_VERSION}
GIT_BASEPATH=${MLCOMMONS_HOME}/${RUN_USER}/git-${RUN_ID}
LOCKDB=${MLCOMMONS_HOME}/${RUN_USER}/lockdb-${RUN_ID}

mkdir -p ${LOCKDB}

# Cleanup locks and temporary files on failed execution
trap "rm -f ${MLCOMMONS_HOME}/${RUN_USER}/.*.lock ; exit" 1 2 3 6 15


### FUNCTION BEGIN
# Performs a git checkout
# GLOBALS:
#   MLCOMMONS_HOME
#   GIT_REV
# ARGUMENTS:
#   $1 - The url to use with the git command
#   $2 - The folder to place the cloned repo into
#   $3 - The branch to checkout after cloning.  By default, this is "main".
# OUTPUTS:
#   Git command line output
### FUNCTION END
function git_checkout() {
    local repo_url=$1
    local target=${2:-$(basename ${repo_url} .git)}
    local branch=${3:-main}

    # Make sure the base directory we're cloning into exists.
    local target_lead=$(dirname $target)
    mkdir -p ${target_lead}

    # Clone the repo if it doesn't already exist and switch to the
    # targeted branch.
    if [ ! -e "$target" ]; then
        git clone $repo_url $target
        (cd $target && \
            git checkout $branch)
    # Ensure that the current checkout doesn't have any local edits
    # and checks out the named branch.
    else
        (cd $target && \
            git fetch origin  && \
            git checkout $branch && \
            git reset --hard origin/$branch && \
            git clean -d --force)
    fi
    GIT_REV="$(cd "$target" && git rev-parse --short=8 HEAD)"
}


### FUNCTION BEGIN
# Creates a file lock handle.  Creates an empty file in
# $MLCOMMONS_HOME with the passed name as a dot file.
# GLOBALS:
#   LOCKDB
# ARGUMENTS:
#   $1 - The lock name to register in MLCOMMONS_HOME
### FUNCTION END
function make_lock() {
    local lockname="${LOCKDB}/.$1.lock"
    touch "${lockname}"
}

### FUNCTION BEGIN
# Releases a file lock handle.  Deletes an empty file in
# $MLCOMMONS_HOME with the passed name as a dot file.
# GLOBALS:
#   LOCKDB
# ARGUMENTS:
#   $1 - The lock name to register in MLCOMMONS_HOME
### FUNCTION END
function release_lock() {
    local lockname="${LOCKDB}/.$1.lock"
    if [ -e "${lockname}" ] ; then
      rm -f "${lockname}"
    fi
}

### FUNCTION BEGIN
# Deadwait function that stops execution while the named
# lock is present on the filesystem.
# GLOBALS:
#   LOCKDB
# ARGUMENTS:
#   $1 - The lock name to register in MLCOMMONS_HOME
#   $2 - The duration to wait in seconds.  Defaults to 5 seconds.
# OUTPUTS:
#   Prints a dot every increment that the file lock is present.
### FUNCTION END
function wait_lock() {
    local lockname="${LOCKDB}/.$1.lock"
    local delay=${2:-5}
    printf "Checking for lock ${lockname}"
    while [ -e ${lockname} ] ; do
      printf "."
      sleep $delay
    done
    printf ". Done\n"
}

### FUNCTION BEGIN
# Utility function that waits until the named lock has been released
# and then immediately claims the lock.
# CALLTREE:
#   wait_lock
#   make_lock
# ARGUMENTS:
#   $1 - The lock name to use for wait_lock and make_lock
#   $2 - The duration to wait in seconds.  Defaults to 5 seconds.
# OUTPUTS:
#   Prints a dot every increment that the file lock is present.
### FUNCTION END
function waitmake_lock() {
    wait_lock $1 ${2:-5}
    make_lock $1
}


### FUNCTION BEGIN
# Outputs the total runtime accoring to the script.  Only functions
# when running on a slurm system
# GLOBALS
#   SLURM_JOBID
# OUTPUTS
#   a CSV prefixed with '# slurmjob,' with the jobid, username, submittion time
#   start time, end time, and status.
### FUNCTION END
function slurm_runtime() {
    # Run only if done in a slurm job
    if [[ ! -z "${SLURM_JOBID}" ]] ; then
      # Query current job details and substitute end time with the current time.
      sacct -j ${SLURM_JOBID} -P --delimiter=, -o jobid,user,submit,start,end,state \
        | sed -e 's/^/# slurmjob,/g' \
              -e 's/,Unknown,RUNNING$/,'$(date +%Y-%m-%dT%H:%m:%S -d @${END_TIME})'DONE/g'
    fi
}

### FUNCTION BEGIN
# Creates a virtual environment in a specified directory using a
# specific requirements.txt file.
# GLOBALS
#   RUN_BASE
# CALLTREE
#   waitmake_lock
# ARGUMENTS
#   $1 - The folder to establish the python virtual environment into
#   $2 - The path to a python requirements.txt file
### FUNCTION END
function setup_venv(){
    local target=$1
    local requirements_path=$2
    # Prevent parallel updates to pip
    waitmake_lock pip.lock
        python -m venv --upgrade-deps ${target}
	case "$(uname -s)" in
	  MINGW*) source ${target}/Scripts/activate ;;
	  *) source ${target}/bin/activate ;;
        esac
        python -m pip install -r ${requirements_path} \
            --progress-bar off
        python -m pip install cloudmesh-gpu --progress-bar off
        python -m pip freeze > ${RUN_BASE}/pip-freeze.txt
    release_lock pip.lock
}

### FUNCTION BEGIN
# Outputs slurm-aware environment variables
# OUTPUT
#   A fenced listing of all current environment variables
#   beginning with SLURM_.  This is only displayed if the
#   current environment has the sbatch command installed.
### FUNCTION END
function slurm_env() {
  if command -v sbatch ; then
    echo "Slurm Environment Details"
    echo "===start[env]==========="
    printenv | grep "SLURM_"
    echo "===end[env]============="
  fi
}


mkdir -p ${RUN_BASE}
RUN_BASE_ABS=$(realpath ${RUN_BASE})

echo "Working in <$(pwd)>"
echo "Base directory in <${RUN_BASE}>"
echo "MLCommons home in <${MLCOMMONS_HOME}>"
echo "Revision: <${REV}>"
echo "Script: <{script}>"
echo "Python: <${PYTHON_VERSION}>"
echo "Run ID: <${RUN_ID}>"

# Load cuda/python on HPCs if module is present.
if command -v module ; then
  module purge
  module use ${RESOURCE_DIR}/modulefiles
  # BUG need to reconfigure script to support python on other platforms.
  module load python-rivanna/${PYTHON_VERSION} cuda cudnn
fi

if [ -e "$(pwd)/config.yaml" ]; then
	cp $(pwd)/config.yaml ${RUN_BASE_ABS}
fi



# Checkout git repo for mlcommons earthquake data, locking the repo
# to prevent concurrent git operations
waitmake_lock mlcommons-data-earthquake.git
    git_checkout ${EARTHQUAKE_DATA_REPO} \
                 "${GIT_BASEPATH}/mlcommons-data-earthquake" \
                 main
release_lock mlcommons-data-earthquake.git

# Checkout git repo for mlcommons main repository, locking the repo
# to prevent concurrent git operations
waitmake_lock mlcommons.git
    git_checkout ${NOTEBOOK_REPO} \
                 "${GIT_BASEPATH}/mlcommons" \
                 $BRANCH
    # Extract the mlcommons eearthquake data to the required path.
    if [ ! -e ${RUN_BASE}/data/EarthquakeDec2020 ]; then
        tar Jxvf ${GIT_BASEPATH}/mlcommons-data-earthquake/data.tar.xz \
            -C ${RUN_BASE}
        # BUG; should be in zip file (or created as part of the python file
        mkdir -p ${RUN_BASE}/data/EarthquakeDec2020/Outputs
    fi

    # Create a python virtual environment based on the current benchmark's requirements.txt file
    setup_venv ${VENV_PATH} \
               ${GIT_BASEPATH}/mlcommons/benchmarks/earthquake/${REV}/requirements.txt

    # Copy the earthquake notebook into our working directory to prevent any modifications to the git repo.
    mkdir -p ${NB_OUTPUT}
    cp ${GIT_BASEPATH}/mlcommons/benchmarks/earthquake/${REV}/{script} \
       ${RUN_BASE}/FFFFWNPFEARTHQ_newTFTv29-${RUN_USER}-${GIT_REV}.ipynb
release_lock mlcommons.git

cms gpu watch --gpu=0 --delay=1 --dense > ${RUN_BASE}/data/EarthquakeDec2020/Outputs/gpu0.log &

# Execute the notebook using papermill
papermill "${RUN_BASE}/FFFFWNPFEARTHQ_newTFTv29-${RUN_USER}-${GIT_REV}.ipynb" \
          "${RUN_BASE}/FFFFWNPFEARTHQ_newTFTv29-${RUN_USER}-${GIT_REV}_output.ipynb" \
          --no-progress-bar --log-output --log-level INFO

# Save the final notebook after everything's done
echo "Saving final notebook to ${NB_OUTPUT}"
cp "${RUN_BASE}/FFFFWNPFEARTHQ_newTFTv29-${RUN_USER}-${GIT_REV}_output.ipynb" \
   "${NB_OUTPUT}/FFFFWNPFEARTHQ_newTFTv29-${GIT_REV}_output.ipynb"

echo "Saving outputs to ${NB_OUTPUT}"
mkdir -p ${NB_OUTPUT}
cp -R "${RUN_BASE}/data/EarthquakeDec2020/Outputs" ${NB_OUTPUT}

echo "Execution Complete"
echo "Cleaning up workspace"
# perform a safe delete; if variables become unset, this will delete ./$USERNAME/./, which should fall
# through.  This is a safety measure to prevent unbounded deletion errors.
# Workspace
rm -rf --preserve-root ${MLCOMMONS_HOME:-.}/${RUN_USER:-$(whoami)}/workspace-${RUN_ID}
# Venv
rm -rf --preserve-root {run.venvpath}/${RUN_USER:-$(whoami)}/venv-${RUN_ID}-${PYTHON_VERSION}
# Git repos
rm -rf --preserve-root ${MLCOMMONS_HOME:-.}/${RUN_USER:-$(whoami)}/git-${RUN_ID}
# Lock database
rm -rf --preserve-root ${MLCOMMONS_HOME:-.}/${RUN_USER:-$(whoami)}/lockdb-${RUN_ID}

echo "Cleanup complete"

exit 0
