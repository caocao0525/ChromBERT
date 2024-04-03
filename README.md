# ChromBERT: Uncovering Chromatin State Motifs in the Human Genome using a BERT-based Approach

This repository contains the code for 'ChromBERT: Uncovering Chromatin State Motifs in the Human Genome using a BERT-based Approach'. 
If you utilize our models or code, please reference our paper. We are continuously developing this repo, and welcome any issue reports.

This package offers the source codes for the ChromBERT model, which draws significant inspiration from [DNABERT](https://doi.org/10.1093/bioinformatics/btab083) 
<sub>(Y. Ji et al., "DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome", Bioinformatics, 2021.)</sub> 
It includes pre-trained models for promoter regions, 
fine-tuned models, and a motif visualization tool. Following the approach of DNABERT, training encompasses general-purpose pre-training 
and task-specific fine-tuning. We supply pre-trained models for both 2k upstream and 4k downstream, as well as 4k upstream and 4k downstream configurations. 
Additionally, we provide a fine-tuning example for the 2k upstream and 4k downstream setup. For data preprocessing, utility functions are available in `utility.py`, 
which users can customize to meet their requirements.

## Citation

## 1. System Requirements and Optimal Configurations
### Software
- Operating System: Linux (Ubuntu 22.04 LTS recommended)
- Python: 3.6 or higher
- PyTorch: 1.4.0 (Check with `conda list pytorch` to see your installed version)

#### Verified Configurations
We have tested and confirmed that the following configurations work well for running our model:

| Configuration    | CUDA Version | cuDNN Version | NVIDIA Driver Version | GPU               |
|------------------|--------------|---------------|-----------------------|-------------------|
| **Configuration 1** | 10.1         | 7.6.5         | 515.65.01             | NVIDIA A40        |
| **Configuration 2** | 10.0.130     | 7.6.5         | 450.119.04            | NVIDIA RTX 2080Ti |


### Hardware
- Recommended GPU: NVIDIA A40 or higher with appropriate CUDA compatibility
- Memory: 64 GB RAM recommended. This recommendation is based on the memory requirements observed during testing, which includes processing large datasets and maintaining efficient model operations.


Select a configuration that best matches your available resources. Ensuring compatibility among these components is crucial for optimal performance.


## 2. Installation with Environment Setup
We strongly suggest creating a Python virtual environment using [Anaconda](https://docs.anaconda.com/free/anaconda/install/linux/). Below, you'll find a detailed, step-by-step guide that covers everything from establishing a new conda environment to installing ChromBERT along with the necessary packages.


#### 2-1. Create and activate a new conda environment
```
$ conda create -n chrombert python=3.6
$ conda activate chrombert
```

#### 2-2. Install Pytorch 
```
$ conda install pytorch torchvision torchaudio cudatoolkit=11.7 -c pytorch
```
If you encounter any compatibility issues or if your setup requires a different version of PyTorch or CUDA, 
please refer to the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) 
for detailed instructions and compatibility information.

#### 2-3. Clone the DNABERT repository to download the source code to your local machine.
```
$ git clone https://github.com/caocao0525/ChromBERT
```

#### 2-4. Install ChromBERT in editable mode to allow for dynamic updates to the code without needing reinstallation.
```
$ cd ChromBERT
$ python3 -m pip install --editable .
```

#### 2-5. Install required packages
```
$ cd examples
$ python3 -m pip install -r requirements.txt
```

For the environment setup, including the Python version and other settings, you can refer to the configurations used in the [DNABERT GitHub repository](https://github.com/jerryji1993/DNABERT). These guidelines will help ensure compatibility with the foundational aspects we utilize from DNABERT.

## 2. Installation

To get a local copy of the repository, use the following command:
```bash
git clone https://github.com/caocao0525/ChromBERT.git


