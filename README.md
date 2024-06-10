# README

## Introduction

This repo contains all the code needed to reproduce the results of this paper - "Micro-expression Recognition by Fusing AU Detection and Spatio-temporal Features" - [[paper]](https://ieeexplore.ieee.org/abstract/document/10446702). It mainly includes two parts of code, one is Matlab code, and the other is Python code.

## Installation for Python

### Requirements

```command
# Install requirement
$ pip install -r requirements.txt

# Download landmarks weight for DLIB
$ mkdir -p PythonCodes/dataloader/weight
$ wget https://github.com/davisking/dlib-models/raw/master/mmod_human_face_detector.dat.bz2 -P dataloader/weight
$ bzip2 -d PythonCodes/dataloader/weight/mmod_human_face_detector.dat.bz2
$ wget https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2 -P dataloader/weight
$ bzip2 -d PythonCodes/dataloader/weight/shape_predictor_68_face_landmarks.dat.bz2
```

## Instructions for use

In general, this paper conducts end-to-end MER model training on the public datasets, which is divided into the following steps.

1. Video interpolation is performed for each ME sample (with Python codes).
2. Motion magnification on interpolated frames (with Matlab codes).
3. Optical flow extraction between amplified motion differences (with Matlab codes).
4. Preprocessing (with Python codes): Extract a specified number of ROIs for the optical flow map; and process the original ME sample, convert it to the corresponding data format for subsequent AU detection.
5. End-to-end training is based on the above ROIs data and AU data (with Python codes).

Since there are many data files involved, the data path needs to be carefully modified according to the comments.

## Training

```
usage: train.py [-h] --csv_path CSV_PATH --mat_dir mat_dir --au_dir AU_FILE
                --catego CATEGO [--num_classes NUM_CLASSES]
                [--batch_size BATCH_SIZE]
                [--weight_save_path WEIGHT_SAVE_PATH] [--epochs EPOCHS]
                [--learning_rate LEARNING_RATE]

optional arguments:
  -h, --help            show this help message and exit
  --csv_path CSV_PATH   Path for the csv file for training data
  --mat_dir mat_dir		Root for the training images
  --au_dir AU_FILE   	Files root for npz
  --catego CATEGO       SAMM or CASME dataset
  --num_classes NUM_CLASSES
                        Classes to be trained
  --batch_size BATCH_SIZE
                        Training batch size
  --weight_save_path WEIGHT_SAVE_PATH
                        Path for the saving weight
  --epochs EPOCHS       Epochs for training the model
  --learning_rate LEARNING_RATE
                        Learning rate for training the model
```

