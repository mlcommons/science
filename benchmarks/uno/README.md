# CANDLE UNO

CANDLE (Exascale Deep Learning and Simulation Enabled Precision Medicine
for Cancer) project  aims to implement deep learning architectures that
are relevant to problems in cancer. These architectures address problems
at three biological scales: cellular (Pilot1 P1), molecular (Pilot2 P2)
and population (Pilot3 P3).

Pilot1 (P1) benchmarks are formed out of problems and data at the
cellular level. The high level goal of the problem behind the P1
benchmarks is to predict drug response based on molecular features of
tumor cells and drug descriptors. Pilot2 (P2) benchmarks are formed out
of problems and data at the molecular level. The high level goal of the
problem behind the P2 benchmarks is molecular dynamic simulations of
proteins involved in cancer, specifically the RAS protein. Pilot3 (P3)
benchmarks are formed out of problems and data at the population level.
The high level goal of the problem behind the P3 benchmarks is to
predict cancer recurrence in patients based on patient related data.

## Benchmark objectives

Uno application from Pilot1 (P1): The goal of Uno is to predict tumor
response to single and paired drugs, based on molecular features of
tumor cells across multiple data sources. It aims to accelerate the 
science goal of effective drugs can be developed to cure the tumor cells. 

## Data description

Combined dose response data contains sources: CCLE, CTRP, gCSI,
GDSC, NCI60, SCL, SCLC, ALMANAC.FG, ALMANAC.FF, and ALMANAC.1A.
For this benchmark, we used the AUC configuration of Uno that 
utilizes a single data source, CCLE. 

## Steps to run

A static dataset is prebuilt and available 
[here](http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/uno/top_21_auc_1fold.uno.h5)

```bash
$ wget http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/uno/top_21_auc_1fold.uno.h5
````

The training can be initiated with the following command:

```bash
python uno_baseline_keras2.py --config_file uno_auc_model.txt --use_exported_data top_21_auc_1fold.uno.h5 --es True
```

Link to the original instructions is available 
[here](https://github.com/ECP-CANDLE/Benchmarks/tree/develop/Pilot1/Uno)
and [here](https://github.com/ECP-CANDLE/Benchmarks/blob/develop/Pilot1/Uno/README.AUC.md)
