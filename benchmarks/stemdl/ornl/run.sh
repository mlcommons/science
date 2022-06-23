#BSUB -P stf218
#BSUB -W 0:10
#BSUB -nnodes 1
#BSUB -q killable 
#BSUB -J mlperf-science-stemdl
#BSUB -o stemdl-classification%J.o
#BSUB -e stemdl-classification%J.e

module load open-ce
DATA_DIR=/gpfs/alpine/world-shared/stf011/junqi/stemdl/classification/data


jsrun -n1 -a6 -g6 -c42 -r1 -b none python sgcl_runner.py --train-dir $DATA_DIR/train --val-dir $DATA_DIR/test
