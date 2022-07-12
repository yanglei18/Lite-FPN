## Requirements
All codes are tested under the following environment:
*   Ubuntu 18.04
*   Python 3.7
*   Pytorch 1.3.1
*   CUDA 10.1

## Dataset
We train and test our model on official [KITTI 3D Object Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). 
Please first download the dataset and organize it as following structure:
```
kitti
│──training
│    ├──calib 
│    ├──label_2 
│    ├──image_2
│    └──ImageSets
└──testing
     ├──calib 
     ├──image_2
     └──ImageSets
```  

## Setup
1. We use `conda` to manage the environment:
```
conda create -n Lite-FPN python=3.7

conda install pytorch=1.3 torchvision -c pytorch
conda install yacs scikit-image tqdm numba fire pybind11

pip install mmcv-full==1.2.5
pip install mmdet==2.11.0

git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.9.0
pip install -v -e .  # or "python setup.py develop"
```

2. Build codes:
```
cd Lite-FPN
python setup.py build develop
```

3. Link to dataset directory:
```
mkdir datasets
ln -s /path_to_kitti_dataset datasets/kitti
```

## Getting started
First check the config file under `configs/`. 

Training :
```
python tools/plain_train_net.py --config-file "configs/smoke_gn_vector.yaml"
```

Evaluation :
```
python tools/evaluate_script.py --config-file "configs/smoke_gn_vector.yaml"
```

# Acknowledgement

Many thanks to these excellent open source projects:
- [SMOKE](https://github.com/lzccccc/SMOKE) 
