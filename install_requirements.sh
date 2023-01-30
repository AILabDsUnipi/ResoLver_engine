#!/usr/bin/bash

conda install -y python=3.6
conda create -y --name DGN_conda_env python=3.6
conda install -y pandas=1.1.5
conda install -y numba=0.50
conda install -y pyproj=2.6.1.post1
conda install -y cython=0.29.24
yes | pip install geo-py==0.4
conda install -y shapely=1.7.1
conda install -y matplotlib=3.1.1
conda install -y scipy=1.5
conda install -y pymongo=3.12.0
conda install -y tensorflow-gpu=1.14
yes | pip install Keras==2.1.2
conda install -y h5py=2.10.0
conda install -y numpy==1.19.2
conda install -y psutil==5.8.0