
# ChromBERT: Uncovering Chromatin State Motifs in the Human Genome Using a BERT-based Approach


🚧 Work in progress: This branch is being developed to support Python 3.11 and expand ChromBERT’s features.


This repository contains the code for '**ChromBERT: Uncovering Chromatin State Motifs in the Human Genome using a BERT-based Approach**'. 
If you utilize our models or code, please reference our paper. We are continuously developing this repo, and welcome any issue reports.

<p align="center">
  <img src="./abs_fig.png" alt="ChromBERT in a nutshell" width="680">
</p>

This package offers the source codes for the ChromBERT model, which draws significant inspiration from [DNABERT](https://doi.org/10.1093/bioinformatics/btab083) 
<sub>(Y. Ji et al., "DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome", Bioinformatics, 2021.)</sub> 
ChromBERT includes pre-trained models for promoter regions (2 kb upstream to 4 kb downstream of TSS) and whole-genome regions, covering both the 15-chromatin state system (127 cell types from the ROADMAP database) and the 18-chromatin state system (1699 cell types from the IHEC database). Fine-tuned models for gene expression classification and regression (15-chromatin state system) are also provided. For downstream analysis, ChromBERT offers a DTW-based motif clustering and visualization tool.

Utility functions for data preprocessing and analysis are available in the `processing/chrombert_utils` directory. A Google Colab tutorial is provided for dataset preparation and curation, which we recommend completing before proceeding to the training stage in the `training/examples` directory.
## Citation

If you use this repository in your research, please cite our paper:

**ChromBERT: Uncovering Chromatin State Motifs in the Human Genome Using a BERT-based Approach**  
Authors: Seohyun Lee, Che Lin, Chien-Yu Chen, and Ryuichiro Nakato

bioRxiv, July 26, 2024.
**DOI:** [10.1101/2024.07.25.605219](https://doi.org/10.1101/2024.07.25.605219)



<br>

## 1. System Requirements and Optimal Configurations
### Software
- Operating System: Linux (Ubuntu 22.04 LTS recommended)
- Python: 3.11
- PyTorch: 2.6.0 (Check with `conda list | grep torch` to see your installed version)

#### Verified Configurations
We have tested and confirmed that the following configuration works well for running our model:

| Component               | Version / Info                                 |
|-------------------------|------------------------------------------------|
| CUDA Version            | 12.4                                           |
| cuDNN Version           | 9.1.0                                          |
| NVIDIA Driver Version   | 550.78                                         |
| GPU                     | Tested with NVIDIA RTX 6000 Ada Generation     |



### Hardware
- Recommended GPU: NVIDIA RTX 6000 Ada Generation or higher with appropriate CUDA compatibility
- Memory: 251 GB RAM recommended. This recommendation is based on the memory requirements observed during testing, which includes processing large datasets and maintaining efficient model operations. 


<br>

## 2. Installation with Environment Setup
To ensure optimal performance and avoid dependency conflicts, we recommend setting up separate environments for data preprocessing and model training. For each environment, an `environment.yml` file is provided for easy setup using Conda (or [Mamba](https://github.com/mamba-org/mamba) for faster resolution). Follow the instructions below to clone the repository and create the environment.

#### 2-1. Clone the ChromBERT repository
To download the source code to your local machine, execute:

```bash
$ git clone https://github.com/caocao0525/ChromBERT
$ cd ChromBERT
```

#### 2-2. Setting up the data processing environment
Follow these steps to create and activate an environment for data processing and analysis:

Using Conda:
```bash
$ conda env create -f environment.yml
$ conda activate chrombert # Activate the environment

# Note: The prompt will change to reflect the current environment name, shown as (chrombert)$
(chrombert)$ conda deactivate # Deactivate current environment
```

Or, using Mamba (if installed):
```
$ mamba env create -f environment.yml # Create environment from file
$ mamba activate chrombert
```

The `chrombert_utils` package is essential for data preprocessing and downstream analysis related to Chromatin State Sequences. To install this package, ensure you are operating within the data processing environment by following these steps:

```bash
$ conda activate chrombert
(chrombert)$ cd processing
(chrombert)$ pip install -e .
```

#### 2-3. Setting up the training environment
Follow these steps to create and activate an environment specifically for training: 

```bash
$ cd training
$ mamba env create -f environment.yml 
$ conda activate chrombert_training # Activate the environment

# Note: The prompt will change to reflect the current environment name, shown as (chrombert_training)$
(chrombert_training)$ conda deactivate # Deactivate current environment
```
<!--Mamba enhances the setup process by speeding up dependency resolution and package installation compared to Conda.-->

Next, in the `chrombert_training` environment, install the packages for training as follows:

```bash
$ conda activate chrombert_training
(chrombert_training)$ cd training
(chrombert_training)$ pip install -e . --config-settings editable_mode=compat
```


<br> 

## 3. Chromatin state data pre-processing

<!--In this tutorial, we presume that users have a `.bed` file of chromatin states labeled numerically according to 15 different chromatin states classes offered by [ROADMAP](https://egg2.wustl.edu/roadmap/web_portal/chr_state_learning.html) (Roadmap Epigenomics Consortium et al., "Integrative analysis of 111 reference human epigenomes," Nature, 2015). -->

We highly recommend using the Colab tutorial for preparing your pretraining and fine-tuning data:  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/caocao0525/ChromBERT/blob/chrombert-py311-extended/colab/colab_data/test_files/your_notebook.ipynb?flush_cache=true)

<br>

## 4. Training

For pre-training, fine-tuning, and to replicate our results, we recommend users download the `ChromBERT.zip` file from the Zenodo link: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10907412.svg)](https://doi.org/10.5281/zenodo.10907412)

For organized access, please store the downloaded file in an appropriate directory, such as `training/examples/prom/pretrain_data`. 
In this section, we provide procedures for the 4-mer dataset. However, users have the flexibility to change the value of `k` by modifying the line `export KMER=4` in each script to suit their specific requirements.

#### 4-1. Pre-training
The pre-training script is located in the `training/examples/prom/script_pre/` directory. Users can adjust the file names within the script should they alter the directory or the name of the training data files. 

```bash
(chrombert_training) $ cd training/examples/prom/script_pre
(chrombert_training) $ bash run_4mer_pretrain.sh
```

#### 4-2. Fine-tuning
Following pre-training, the parameters are saved in the `training/examples/prom/pretrain_result/` directory. To replicate our fine-tuning results, users should place the files `train.tsv` and `dev.tsv` directly in the `examples/prom/ft_data/` directory. This location includes data for classifying promoter regions between genes that are highly expressed (RPKM > 50) and those that are not expressed (RPKM = 0).Note that our `ChromBERT.zip` file offers 15 different types of promoter region fine-tuning data under the `promoter_finetune_data` directory. Users are encouraged to properly place the required file.

```bash
(chrombert_training) $ cd training/examples/prom/script_ft
(chrombert_training) $ bash run_4mer_finetune.sh
```

#### 4-3. Prediction
To obtain an attention matrix for the prediction result, execute the scripts in the following order: First, run `run_4mer_pred1.sh`, followed by `run_4mer_pred2.sh`. It is essential to ensure that `run_4mer_pred1.sh` is executed before `run_4mer_pred2.sh`.

```bash
(chrombert_training) $ cd training/examples/prom/script_pred
(chrombert_training) $ bash run_4mer_pred1.sh

# After you get the result in the `training/examples/prom/prediction`

(chrombert_training) $ bash run_4mer_pred2.sh
```

<br>

## 5. Motif Detection and Clustering

The identification of chromatin state motifs can be categorized into two phases: Motif Detection and Motif Clustering. During the Motif Detection phase, chromatin state sequences that have high attention scores and are uniquely associated with the class of interest (for example, the promoter region) are identified and organized into a dataframe. Subsequently, these sequences undergo clustering through Dynamic Time Warping (DTW) in the Motif Clustering phase, leading to the identification of the definitive chromatin state motifs.

#### 5-1. Motif Detection

```bash
(chrombert) $ cd training/motif/prom
(chrombert) $ bash ./motif_prom.sh 
```

Executing the script as described above allows users to generate a `init.csv` file in the `./result` directory. This file includes a comprehensive list of chromatin state sequences. To adjust settings such as the window size, minimum sequence length, and the minimum occurrence threshold, users can modify the script's arguments as demonstrated below:

```bash
(chrombert) $ bash ./motif_prom.sh --window_size 12 --min_len 5 --min_n_motif 2
```

For further assistance, the `--help` option provides a detailed explanation of all available arguments, their default settings, and an illustrative example of how to use them:

```bash
(chrombert) $ bash ./motif_prom.sh --help
```

#### 5-2. Motif Clustering

First, users can create a matrix to serve as the foundational data structure for motif clustering by executing the following code:

```python
df_sequences=crb.motif_init2df(input_path='path/to/your/init.csv')
```

To generate the predicted classes for each motif in the `init.csv` file by employing Dynamic Time Warping (DTW) along with agglomerative clustering, execute the code below.
The `categorical` option is a boolean where `True` means that the user considers the distance between each chromatin state equal, while `False` means that the chromatin states A to O are numerically converted to 1 to 15.
The default is False. 

```python
y_pred=crb.motif_init2pred(input_path='path/to/your/init.csv',
                           categorical=False,
                           fillna_method='ffill', # Method to fill NaN padding for shorter sequences ('ffill' or 'O')
                           n_clusters=number_of_clusters
                           linkage_method='complete' # Linkage method for agglomerative clustering. See the documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html 
                           )
```

*[Optional]* We provide a function to create an dendrogram, which aids in determining the optimal number of clusters for usability.

```python
crb.motif_init2pred_with_dendrogram(input_path='path/to/your/init.csv',
                                    categorical=False,
                                    n_cluster=None,
                                    fillna_method='ffill', # Method to fill NaN padding for shorter sequences ('ffill' or 'O')
                                    linkage_method='complete', # Linkage method for agglomerative clustering. See the documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html 
                                    threshold=<int>) # To estimate the initial number of cluster. Users can adjust it according to the shape of dendrogram
```
Note that with `n_cluster=None`, the number of clusters is estimated based on the specified threshold.

To obtain the clustered motifs in a DataFrame format, users can execute the following function:

```python
clustered_sequence=crb.motif_init2class(input_path='path/to/your/init.csv', n_clusters=number_of_clusters)
```

For visualization purposes, users can understand the overall characteristics of clustered motifs by using the following function:

```python
crb.motif_init2cluster_vis(input_path='path/to/your/init.csv',
                           categorical=False,
                           n_clusters=number_of_clusters,
                           fillna_method='ffill', # Method to fill NaN padding for shorter sequences ('ffill' or 'O')
                           linkage_method='complete', # Linkage method for agglomerative clustering. See the documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
                           random_state=<int>, # Random seed for reproducibility of the generated figure
                           font_scale=0.04, # Adjust the text size balance
                           font_v_scale=9, # Adjust the text vertical size ratio
                           fig_w=12, # Figure width
                           fig_h=8, # Figure height
                           node_size=1000, # Size of bubbles
                           node_dist=0.05   # Distance between nodes
                           )
```
Note that the generated image file is saved at the same directory with a name `cluster_result.png`


*[Optional]* We provide an optional feature that facilitates the generation of a UMAP, designed to help users intuitively grasp the essential features of clustered motifs. 
It's important to note that users have the flexibility to configure the `n_neighbors` and `min_dist` parameters to suit their specific needs.

```python
crb.motif_init2umap(input_path='path/to/your/init.csv',
                    categorical=False,
                    n_clusters=number_of_clusters,
                    fillna_method='ffill', # Method to fill NaN padding for shorter sequences ('ffill' or 'O')
                    linkage_method='complete', # Linkage method for agglomerative clustering. See the documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
                    n_neighbors=size_you_want,
                    min_dist=minimum_distance_you_want,
                    random_state=<int>) # Random seed for reproducibility of the generated figure

```
<br>


<!--## Contributing

We welcome contributions to [Project Name]. If you're interested in helping, please take a look at our [Contributing Guidelines](CONTRIBUTING.md) for more information on how to get started.-->

<br>

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.


<br>

<!-- ## Acknowledgements

Thanks to [Contributor Name] for [contribution], and to everyone who has submitted issues or pull requests.

<br>

## Contact

For any questions or feedback, feel free to [open an issue](link-to-issues) or contact me directly at [email@example.com](mailto:email@example.com). -->

---

Thank you for checking out **ChromBERT**. If you found this project useful, please consider starring it on GitHub to help it gain more visibility.


<br>







