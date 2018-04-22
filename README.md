# e-NABLE multi-class classifier 

This repo contains a TensorFlow multi-class classifier which classifies Google+ posts from e-NABLE communitites into the following classes.
* Reporting
* Device
* Delivery 
* Progress
* Becoming member
* Attempt Action
* Activity
* Other

## Setting up the environment

### Prequisites
* TensorFlow
* spaCy
* pandas
* numpy

To create the environment with the above prequisites run
```bash
conda env create -f environment.yml
```

**Note**: This installs the GPU version of TensorFlow. If it's the CPU version that's desired change '_tensorfow-gpu_' to '_tensorflow_' in the environment.yml file.

## Running the code

```bash
# Launch the environment
$ source activate enable-multiclassifier-env

# Train the model
$ python main.py train

# Test the model
$ python main.py test
```

## Summary

```
Class - Report
-----------------------
Precision: 0
Recall: 0.0
F1 Score: 0
-----------------------

Class - Device
-----------------------
Precision: 1.0
Recall: 0.6386554621848739
F1 Score: 0.7794871794871795
-----------------------

Class - Delivery
-----------------------
Precision: 0
Recall: 0.0
F1 Score: 0
-----------------------

Class - Progress
-----------------------
Precision: 1.0
Recall: 0.3755274261603376
F1 Score: 0.5460122699386503
-----------------------

Class - Becoming_Member
-----------------------
Precision: 0
Recall: 0.0
F1 Score: 0
-----------------------

Class - Attempt_Action
-----------------------
Precision: 0
Recall: 0.0
F1 Score: 0
-----------------------

Class - Activity
-----------------------
Precision: 0
Recall: 0.0
F1 Score: 0
-----------------------

Class - Other
-----------------------
Precision: 0
Recall: 0.0
F1 Score: 0
-----------------------
```
