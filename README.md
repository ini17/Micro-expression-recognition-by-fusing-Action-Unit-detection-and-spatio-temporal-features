# README
## Introduction
The source code was trying to reproduce the paper - "Micro expression recognition by fusing Action Unit detection and spatio-temporal features". [[paper]](https://ieeexplore.ieee.org/abstract/document/10446702)
## Installation

### Requirements
```command
# Install requirement
$ pip install -r requirements.txt

# Download landmarks weight for DLIB
$ mkdir -p dataloader/weight
$ wget https://github.com/davisking/dlib-models/raw/master/mmod_human_face_detector.dat.bz2 -P dataloader/weight
$ bzip2 -d dataloader/weight/mmod_human_face_detector.dat.bz2
$ wget https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2 -P dataloader/weight
$ bzip2 -d dataloader/weight/shape_predictor_68_face_landmarks.dat.bz2
```

### MagNet
The structure of MagNet was adapted from [here](https://github.com/ZhengPeng7/motion_magnification_learning-based). Please download the pretrained weight from their release and place in `dataloader/weight/`.

### DLIB with GPU (not necessary)
```command
# Remove the cpu version first
$ pip uninstall dlib
# Install cudnn and its toolkit
$ conda install cudnn cudatoolkit
# Build from source
$ git clone https://github.com/davisking/dlib.git
$ cd dlib
$ mkdir build & cd build
$ cmake .. \
    -DDLIB_USE_CUDA=1 \
    -DUSE_AVX_INSTRUCTIONS=1 \
    -DCMAKE_PREFIX_PATH=<path to  conda env>\
    -DCMAKE_C_COMPILER=gcc-6 -DCMAKE_CXX_COMPILER=g++-6
$ cmake --build .
$ cd ..
$ python setup.py install \
    --set USE_AVX_INSTRUCTIONS=1 \
    --set DLIB_USE_CUDA=1 \
    --set CMAKE_PREFIX_PATH=<path to  conda env>  \
    --set CMAKE_C_COMPILER=gcc-6 \
    --set CMAKE_CXX_COMPILER=g++-
```

## Dataset
* [CASME II](http://fu.psych.ac.cn/CASME/casme2-en.php)
* [SAMM](https://personalpages.manchester.ac.uk/staff/adrian.davison/SAMM.html)

## Training
```
usage: train.py [-h] --csv_path CSV_PATH --image_root IMAGE_ROOT --npz_file
                NPZ_FILE --catego CATEGO [--num_classes NUM_CLASSES]
                [--batch_size BATCH_SIZE]
                [--weight_save_path WEIGHT_SAVE_PATH] [--epochs EPOCHS]
                [--learning_rate LEARNING_RATE]

optional arguments:
  -h, --help            show this help message and exit
  --csv_path CSV_PATH   Path for the csv file for training data
  --image_root IMAGE_ROOT
                        Root for the training images
  --npz_file NPZ_FILE   Files root for npz
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

## Citation
```bibtex
@inproceedings{wang2024micro,
  title={Micro-expression recognition by fusing action unit detection and Spatio-temporal features},
  author={Wang, Lei and Huang, Pinyi and Cai, Wangyang and Liu, Xiyao},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={5595--5599},
  year={2024},
  organization={IEEE}
}
```
