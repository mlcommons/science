:toc:
:toclevels: 4

:sectnums:

# MLCommons® Science Working Group  Benchmark Submission Rules

:TOC:

## Difference in the General Submission Rules

The general MLCommons® submission rules are summarized in

* https://github.com/mlcommons/policies/edit/master/submission_rules.adoc

This document lists the differences between the MLCommons Science Group Submissions. The differences are summarized by

* Rolling submission with review
* The directory structure for the coordinated storing of results as required by MLCommons
* Augmentation of the code to produce valid mllog files

All other requirements are the same as discussed in the General Submission Rules Document

## Overview of the Rolling Submission Process

The following goals are to be captured by the submission process

* Submissions can be made to the MLCommons® Science GitHub at any time for any benchmark that has been released.
* Submissions will be automatically checked and then reviewed by the working group which has a review committee for each benchmark.
* Depending on the number of scientific innovations in the submission, the review time will vary.
* The submitters will get an automatic acknowledgment of the submission and a customized response from the committee within a week of the submittal date.
* This second response will indicate the estimated time for a committee review to be completed.
* On completion of the committee review, all submissions that are considered in scope will be posted on the working group GitHub which includes a scientific discovery "leaderboard" for each benchmark.
    Updates will be summarized quarterly
* The innovations will be described and can include aspects other than final accuracy
    e.g. the submission might need a smaller dataset to achieve an interesting accuracy. It is expected that benchmarks will be posted for at least a year to gather a rich set of input.

## Timeline of Deployment

The submission process will be in place for release at SC22, Dallas November 16.

== Logging Libraries

Augmentation of codes for consideration into the inclusion of the
science benchmarks must use the

* https://github.com/mlcommons/logging[MLCommons® Logging Library]

An alternative library that internally produces MLCommons® events for
logging is the

* https://github.com/cloudmesh/cloudmesh-common/blob/main/cloudmesh/common/StopWatch.py[StopWatch] from https://github.com/cloudmesh/cloudmesh-common[cloudmesh-common]
* https://github.com/cloudmesh/cloudmesh-common/blob/main/README-mlcommons.md[Quickstart for using Cloudmesh StopWatch for MLcommons]

This library has the advantage of generating a human-readable summary
table in addition to the MLCommons® log events.



## Directory Structure

---
> Note: The original directory structure rule Section 13 will be removed from the document
>
> * https://github.com/mlcommons/science/blob/main/policy.adoc#directory-structure-for-submission
>
> and placed here.

---


In this section, we document the directory structure for submissions. We introduce the following variables denoted by `{ }` around the Variable name. The brackets `[ ]` are used to donate a list

`{organization}` ::= The organization submitting the benchmark

`{application}` ::= The application, a value from [cloudmask,earthquake,uno,stemdl]

`{system}` ::= Defines the system used for this benchmark

`{descriptor}` ::= A unique descriptor of the experiment, as described in the sientific_contribution. For example `experiment-1`, `faultzone-3`.

`{n}` ::= number of repeated experiments

All results are stored in a directory such as

`{organization}/{application}/{system}/{descriptor}/`

Within this directory, all parameters for that experiment are stored, so that all information for the experiment is self-contained within the experiment.

This includes

1. A number of scripts that are used to run the particular benchmark on the
   specified system to allow reproducibility.


2. `result-n.txt` ::= The result logs for the `n`-th run with the parameters
   defined by `config.yaml`

3. `config.yaml` ::= A configuration file that contains all hyperparameters and
   other parameters to define a run. This configuration file contains an entry
   that uniquely describes the version of the code that is run. The version
   must be included in the MLCommons benchmark repository. This also includes all
   hyperparameters including new ones particular to this approach.
   The configuration file should include enough details to replicate the
   experiment with the locations of the program and the data. If the data does not fit in a GitHub repository it can be placed in a publicly accessible data store. Its location needs to be specified as an endpoint in the YAML file, with a command line example on how to retrieve it. As multiple files could be needed a list of commands can be specified.
  Examples of the configuration format in the YAML file are:

   github:
     discription: EarthQuake Prediction
     repo: https://mlcommons.github.com/...
     branch: main
     version: 1.0
     tag: 1.0
  data:
  - aws s3 rsync ....


4. `sientific_contribution.pdf` ::= A detailed description of the scientific
   contribution and the algorithms and associated hyperparameters used in the
   benchmark.

5. A `README.md` file that describes how to run it.
   The `README.md` must have
   sufficient information to create such runs. In some cases, a program may be
   used to run multiple experiments and create
   such a directory automatically. Enough information must be included in the
   directory, so such parameterized runs can be conducted, while also
   replicating the appropriate directory structure. The reason we require for
   each result its own subdirectory is to allow output notebooks and comments to be
   submitted for each of the results if needed. This is especially the case
   when jupyter notebooks are used as the benchmark to be executed, allowing
   the notebook with all its cells to be submitted along the `results.txt`
   file.

6. Log File requirements.

   1. The log file must have am Organization Records in Mllog entry format. This includes mllog entries for `POINT_IN_TIME` with the values

      * submission_benchmark
      * submission_org
      * submission_division
      * submission_status
      * submission_platform

   2. Scientific Result. Each benchmark must have an mllog entry  POINT_IN_TIME with the key "result" and the value of a dict
      describing the result format and meaning.
      The result must be documented in detail in the `sientific_contribution.pdf` file.


== References

We included here a list of supporting and related documents

* [1] https://github.com/laszewsk/mlcommons/raw/main/pub/Science-WG-of-MLCommons®-presentation.pdf[Overview presentation of the MLScience Group]  Barrett,
Wahid Bhimji,
Bala Desinghu,
Murali Emani,
Geoffrey Fox,
Grigori Fursin,
Tony Hey,
David Kanter,
Christine Kirkpatrick,Hai Ah Nam,
Juri Papay,
Amit  Ruhela,
Mallikarjun Shankar,
Jeyan Thiyagalingam
Aristeidis Tsaris,
Gregor von Laszewski,
Feiyi Wang,
Junqi Yin
, MLCommons® Community Meeting, (also available in
https://docs.google.com/presentation/d/1xo_M3dEV1BS7OcXjvjyOUOLkHh8WyHuawqj1OR2iJw4/edit#slide=id.g10e8f04304c_1_73[Google docs]), December 9 2021.

* [2] https://github.com/laszewsk/mlcommons/raw/main/pub/mlcommons_science_wg_paper_2022.pdf[AI Benchmarking for Science: Efforts from the
MLCommons® Science Working Group], Jeyan Thiyagalingam, Gregor von Laszewski, Junqi Yin, Murali Emani,
Juri Papay, Gregg Barrett, Piotr Luszczek, Aristeidis Tsaris,
Christine Kirkpatrick, Feiyi Wang, Tom Gibbs, Venkatram Vishwanath,
Mallikarjun Shankar, Geoffrey Fox, Tony Hey, June 2022

* [3] https://mlcommons.org/en/policies/[MLCommons® Policies]

* [4] https://github.com/mlcommons/training_policies[MLCommons® Training policies]

* [4] https://github.com/mlcommons/inference_policies[MLCommons® Interference Policies]

* [6] https://github.com/mlcommons/policies[MLCommons® submission Rules for training and inference]

* [7] https://github.com/mlcommons/science[MLCommons® Science GitHub Repository]

* [8] https://github.com/laszewsk/mlcommons[Science Development GitHub Repository to prepare release candidates for the MLCommons® repository]
