# Running on test data

## MoveNet+GRU

Requirements for the notebook is listed at `notebooks/gru-requirements.txt`
- Navigate to `notebooks/gru_approach.ipynb`
- Change `dataset_dir` to appropriate test directory. It is expected to have 32 folders, the same as provided train dataset.
- Fully run the notebook

# IMIGUE Classification with ViT

All code is executed on powerful machine with RTX4090 GPU, wih `pytorch_2.2.1-cuda12.1-cudnn8-devel/jupyter` image. 
Please follow official installation guides for `cudatoolkit` and `pytorch` to be able to use this code.


# Install (Assuming you have torch and cudatoolkit)

```
pip install -r requirements.txt
```
