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

<br>

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

<br>

## 2. Installation with Environment Setup
We strongly suggest creating a Python virtual environment using [Anaconda](https://docs.anaconda.com/free/anaconda/install/linux/). Below, you'll find a detailed, step-by-step guide that covers everything from establishing a new conda environment to installing ChromBERT along with the necessary packages.


#### 2-1. Create and activate a new conda environment
```bash
$ conda create -n chrombert python=3.6
$ conda activate chrombert
```

#### 2-2. Install Pytorch 
```bash
$ conda install pytorch torchvision cudatoolkit=11.7 -c pytorch
```
If you encounter any compatibility issues or if your setup requires a different version of PyTorch or CUDA, 
please refer to the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) 
for detailed instructions and compatibility information.

#### 2-3. Clone the DNABERT repository to download the source code to your local machine.
```bash
$ git clone https://github.com/caocao0525/ChromBERT
```

#### 2-4. Install ChromBERT in editable mode to allow for dynamic updates to the code without needing reinstallation.
```bash
$ cd ChromBERT
$ python3 -m pip install --editable .
```

#### 2-5. Install required packages
```bash
$ cd examples
$ python3 -m pip install -r requirements.txt
```

For the environment setup, including the Python version and other settings, you can refer to the configurations used in the [DNABERT GitHub repository](https://github.com/jerryji1993/DNABERT). These guidelines will help ensure compatibility with the foundational aspects we utilize from DNABERT.


<br> 

## 3. Chromatin state data pre-processing
In this tutorial, we presume that users have a `.bed` file of chromatin states labeled numerically according to 15 different chromatin states classes offered by [ROADMAP](https://egg2.wustl.edu/roadmap/web_portal/chr_state_learning.html) (Roadmap Epigenomics Consortium et al., "Integrative analysis of 111 reference human epigenomes," Nature, 2015). 

#### 3-1. Convert `.bed` to a string

*Step 1*. Convert  `.bed` to DataFrame

Begin by using the `bed2df_expanded` function, which transforms a .bed file into a DataFrame. This function expects the path to your bed file as its argument. The resulting DataFrame features columns such as Chromosome, Start, End, State (numerical representation of chromatin states), Length, Unit (the length divided by 200 base pairs for normalization), State_Seq (a sequence of alphabets representing chromatin states), and State_Seq_Full (the State_Seq extended according to the Unit length).

Example: 

```python
from css_utility import bed2df_expanded
dataframe = bed2df_expanded('path/to/your_bed_file.bed')
```

*[Optional]* Save `.bed` DataFrames Cell-Wise

For batch processing of .bed files stored
in a directory, employ the unzipped_to_df function. This function processes each .bed file in the specified directory (bed_file_dir), converting them into DataFrames as outlined in Step 1, and handles them in a manner conducive to your analysis needs (e.g., storing each DataFrame separately for cell-wise analysis).

Ensure your .bed files are located in the bed_file_dir before executing this function. The function iteratively reads each .bed file, converting it into a DataFrame with the structure and content detailed previously.

Example:

```python
from css_utility import unzipped_to_df
unzipped_to_df('path/to/bed_file_dir', output_path='path/to/your/output_dir')
```

*Step 2*. Convert DataFrame to string

The DataFrame created from the .bed file can be converted into a string of alphabets, where each letter represents a chromatin state. This allows users to treat the data as raw sequences. The function `df2unitcss` (recommended) compresses these sequences by reducing the genomic representation to units of 200 bps, reflecting the labeling of chromatin states at this resolution. For users who wish to retain the original genomic length in their analyses, we provide the `df2longcss` function. The output from both functions is a chromosome-wise list (excluding the Mitochondrial chromosome) of alphabet strings, with each string corresponding to a chromosome.

```python
from css_utility import df2unitcss
unit_length_string_list = df2unitcss(your_dataframe)
```

*Step 3*. Pretraining data preparation

In this section, we provide a guide for extracting promoter regions and preparing pretraining data. 
We use the `RefSeq_WholeGene.bed` file,which includes comprehensive gene annotations from the RefSeq database, aligned with the hg19 human genome assembly (GRCh37). The term "DataFrame" in `input_path` below refers to the data format you should have obtained from *Step 1*. 

```python
from css_utility import save_TSS_by_loc

save_TSS_by_loc('path/to/RefSeq_WholeGene.bed', input_path='path/to/your/dataframe', output_path='path/to/your/output', file_name='your_filename_suffix', up_num=upstream_distance, down_num=downstream_distance, unit=200)

```
This function enables users to extract and save specific regions of interest (e.g., user-defined promoter regions) as a pickle file. 
You can define these regions by setting `up_num` and `down_num`,  which represent the distances upstream and downstream from the Transcription Start Site (TSS), respectively.


<br>

## 4. Pre-train

<br>

## 5. Fine-tuning

<br>

## 6. Prediction and Visualization

<br>

## 7. Motif Detection

<br>

## 8. Motif Clustering



