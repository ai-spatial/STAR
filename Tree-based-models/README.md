# Spatial Transformation Framework for Random Forest
The deep learning version is at: [Deep-learning-models](/Deep-learning-models): Fully-connected, convolutional, recurrent, ...

## Paper information
**[RSE'25]** Yiqun Xie, Anh Nhu, Xiao-Peng Song, Xiaowei Jia, Sergii Skakun, Haijun Li, Zhihao Wang. Accounting for Spatial Variability with Geo-aware Random Forest: A Case Study for US Major Crop Mapping. Remote Sensing of Environment, 2024.

The code uploaded is a temporary version: It was developed building on the deep learning version of the code so the code includes references to tensorflow packages, where some fucntions were still used for convenience and left unchanged. Later developments will clean those and replace them with numpy or sklearn functions to simplify the package imports in certain environments.

It is recommended to run the model on multi-core machines to maximize the efficiency for random forest models. There is a parameter setting in paras.py that allows customized number of cores.
