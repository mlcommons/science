This folder contains the original implementation of STEMDL developed at ORNL.

## Software Requirements:

### Classification Task

- Python >= 3.6
- PyTorch >= 1.2 
- Horovod >= 0.19 
- opencv-python 
- h5py 
- mlperf-logging 

## Data  

Download the classificatoin dataset from [10.13139/OLCF/1510313](https://doi.ccs.ornl.gov/ui/doi/70) via Globus and pre-process into numpy files via 
```bash
python preprocess/preprocessor_parallel.py <path-to-downloaded-data> <path-to-processed-data>
```
Note that the train-test data split is with respect to different materials. 

If you have access to OLCF machines, the preprocessed data are available at 
```bash
classification: /gpfs/alpine/world-shared/stf011/junqi/stemdl/classification/data
```

## Quickstart

- Clone the repo.

- Run the space-group classification: 

__run.sh__ is the working job script on Summit, and the training is launched via   

```bash
jsrun -n<NODES> -a6 -g6 -c42 -r1 -b none python sgcl_runner.py --epochs 10 
                                                         --batch-size 32 
                                                         --train-dir <path-to-train-dataset> 
                                                         --val-dir <path-to-val_dataset>  
```

## Scientific Goal 

### Classification 

- A universal classifier for the space group of solid-state materials. 
 
## Metrics 

### Classification

- Due to the intrinsic imbalance of the crystal space group distribution in nature, the classes in the dataset are also imbalanced. We use both __top1 accuracy__ and __F1 score (Macro)__ to measure the model performance and the example implementation provides a baseline validation top1 accuracy ~0.57 and F1 score ~0.43. Note that the train-test data split is with respect to different materials, i.e. the validation is on materials hasn't been seen by the model.   
 
- Considering the application of the pre-trained model at the edge, other metrics of interest are the __model size__ and __inference time__. Generally, for the same performance metric as above, the smaller the model size, the better.    

