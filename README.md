# IMIGUE Classification

This repository contains code for classification for the [Micro-Gesture Understanding and Emotion Analysis dataset](https://github.com/linuxsino/iMiGUE). We have notebooks for both attempted approaches here (Movenet+GRU and finetuned ViT).

# Install
```
pip install -r requirements.txt
```

# Methods
We host two methods, ViT (best results) and GRU. In order to run ViT, all you have to do is to run `./notebooks/inference_vit.ipynb`. 
**Keep in mind** that you will need appropriate GPU hardware to do inference, as the ViT model is quite compute heavy.

In order to run Movenet+GRU, navigate to `notebooks/gru_approach.ipynb`, change `dataset_dir` to appropriate test directory. It is expected to have 32 folders, the same as provided train dataset. Run the notebook end to end.

# Note

All code is executed on powerful machine with RTX4090 GPU, wih `pytorch_2.2.1-cuda12.1-cudnn8-devel/jupyter` image. 
Please follow official installation guides for `cudatoolkit` and `pytorch` to be able to use this code.