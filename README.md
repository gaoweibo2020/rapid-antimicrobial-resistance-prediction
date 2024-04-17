###  The implementation of "Machine learning assisted MALDI mass spectrometry for rapid antimicrobial resistance prediction in clinicals"

------

#### Prepare Environment

```shell
conda create -n medical python==3.9
pip install jupyter notebook
pip install numpy
pip install scipy
pip install pandas
pip install sklearn
pip install scipy
```

#### Prepare Dataset

1. put the `txt` file of MALDI mass spectrometry into `data` folder

2. make data.csv

   | ID                                      | Class                                              | Homology                | Date                  | Patient        |
   | --------------------------------------- | -------------------------------------------------- | ----------------------- | --------------------- | -------------- |
   | Data representation and data file names | ‘S’ stands for sensitive; ‘R’ stands for resistant | The homology of patient | Data measurement time | patient number |
   |                                         |                                                    |                         |                       |                |
   |                                         |                                                    |                         |                       |                |

#### Run Algorithms

- `machine_learning.ipynb` is about running a machine learning algorithm on the patient-wise.
- `machine_learning_time_wise.ipynb` is about running a machine learning algorithm on the time-wise.
- `feature_selection.ipynb` is about running peak feature screening algorithms.

