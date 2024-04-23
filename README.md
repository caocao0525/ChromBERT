# ChromBERT: Uncovering Chromatin State Motifs in the Human Genome Using a BERT-based Approach

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
We recommend setting up a Python virtual environment using [Mamba](https://github.com/mamba-org/mamba), a fast and efficient Conda-compatible package manager. Below is a concise guide on how to create a new environment and install ChromBERT along with all required dependencies.

#### 2-1. Clone the ChromBERT repository to download the source code to your local machine.
To download the source code to your local machine, execute:

```bash
$ git clone https://github.com/caocao0525/ChromBERT
```

#### 2-2. Create and activate a new environment
Using Mamba, you can quickly create and activate a new environment for the ChromBERT project. Follow these steps:
<!--
```bash
$ conda create -n chrombert python=3.6
$ conda activate chrombert
```
-->
<!--
---- Testing ... merging env of training and analysis -->

```bash
$ conda install mamba -n base -c conda-forge # Install Mamba using Conda
$ cd examples
$ mamba env create -f environment.yml # Create environment from file
$ conda activate chrombert # Activate the environment
```
Mamba enhances the setup process by speeding up dependency resolution and package installation compared to Conda.

<!--
```bash
$ cd ChromBERT/examples
$ conda env create -f environment.yml
$ conda activate chrombert
```
-->

<!--(chrombert) $ conda install pytorch torchvision cudatoolkit=11.7 -c pytorch-->

#### 2-3. Install `css_utility` package
For data pre-processing and downstream analysis, install the utility package.

```bash
(chrombert)$ cd css_utility
(chrombert)$ pip install .
```

#### 2-4. Environment Details

ChromBERT uses a specific set of packages and versions to ensure compatibility and performance. The environment is configured with the following key components:

- PyTorch 1.4.0
- Torchvision 0.2.1
- CUDA Toolkit 10.1.243
- cuDNN 7.6.5

Ensure these versions are used to avoid compatibility issues. The environment can be set up using the provided `environment.yml` file which includes these specific versions.

If you encounter any compatibility issues or if your setup requires a different version of PyTorch or CUDA, 
please refer to the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) 
for detailed instructions and compatibility information.

<!--
```bash
(chrombert) $ conda install pytorch torchvision cudatoolkit=10.1.243 -c pytorch
```

#### 2-4. Install ChromBERT in editable mode to allow for dynamic updates to the code without needing reinstallation.
```bash
(chrombert) $ cd ChromBERT
(chrombert) $ python3 -m pip install --editable .
```

#### 2-5. Install required packages
```bash
(chrombert) $ cd examples
(chrombert) $ python3 -m pip install -r requirements.txt
```

For the environment setup, including the Python version and other settings, you can refer to the configurations used in the [DNABERT GitHub repository](https://github.com/jerryji1993/DNABERT). These guidelines will help ensure compatibility with the foundational aspects we utilize from DNABERT.
-->




<br> 

## 3. Chromatin state data pre-processing
In this tutorial, we presume that users have a `.bed` file of chromatin states labeled numerically according to 15 different chromatin states classes offered by [ROADMAP](https://egg2.wustl.edu/roadmap/web_portal/chr_state_learning.html) (Roadmap Epigenomics Consortium et al., "Integrative analysis of 111 reference human epigenomes," Nature, 2015). 

Before you start, ensure your files are named according to the "E###" format, where "E" is a fixed prefix and "###" represents an integer, such as "E001" or "E127".

#### 3-1. Convert `.bed` to a string

*Step 1*. Convert  `.bed` to DataFrame

Begin by using the `bed2df_expanded` function, which transforms a `.bed` file into a DataFrame. This function expects the path to your bed file as its argument. The resulting DataFrame features columns such as `Chromosome`, `Start`, `End`, `State` (numerical representation of chromatin states), `Length`, `Unit` (the length divided by 200 base pairs for normalization), `State_Seq` (a sequence of alphabets representing chromatin states), and `State_Seq_Full` (the `State_Seq` extended according to the Unit length).

Example: 

```python
from css_utility import *
dataframe = bed2df_expanded('path/to/your_bed_file.bed')
```

*[Optional]* Save `.bed` DataFrames Cell-Wise

For batch processing of `.bed` files stored
in a directory, employ the `unzipped_to_df` function. This function processes each `.bed` file in the specified directory (`bed_file_dir`), converting them into DataFrames as outlined in Step 1, 
and handles them in a manner conducive to your analysis needs (e.g., storing each DataFrame separately for cell-wise analysis).

Ensure your `.bed` files are located in the `bed_file_dir` before executing this function. The function iteratively reads each `.bed` file, converting it into a DataFrame with the structure and content detailed previously.

Example:

```python
from css_utility import *
unzipped_to_df('path/to/bed_file_dir', output_path='path/to/your/output_dir')
```

*Step 2*. Convert DataFrame to string

The DataFrame created from the .bed file can be converted into a string of alphabets, where each letter represents a chromatin state. This allows users to treat the data as raw sequences. The function `df2unitcss` (recommended) compresses these sequences by reducing the genomic representation to units of 200 bps, reflecting the labeling of chromatin states at this resolution. For users who wish to retain the original genomic length in their analyses, we provide the `df2longcss` function. The output from both functions is a chromosome-wise list (excluding the Mitochondrial chromosome) of alphabet strings, with each string corresponding to a chromosome.

```python
from css_utility import *
unit_length_string_list = df2unitcss(your_dataframe)
```

*Step 3*. Pre-training data preparation

In this section, we provide a guide for extracting promoter regions and preparing pretraining data. 
We use the `RefSeq_WholeGene.bed` file, which includes comprehensive gene annotations from the RefSeq database, aligned with the hg19 human genome assembly (GRCh37). The term "DataFrame" in `input_path` below refers to the data format you should have obtained from *Step 1*. 

```python
from css_utility import *
save_TSS_by_loc('path/to/RefSeq_WholeGene.bed', input_path='path/to/your/dataframe', output_path='path/to/your/output', file_name='your_filename_suffix', up_num=upstream_distance, down_num=downstream_distance, unit=200)
```
This function enables users to extract and save specific regions of interest (e.g., user-defined promoter regions) as a pickle file. 
You can define these regions by setting `up_num` and `down_num`,  which represent the distances upstream and downstream from the Transcription Start Site (TSS), respectively.

After extraction, users can segment the data into k-mers using the function below, adjusting the `k` value to the desired number. 
For optimal computational efficiency, we recommend using 4-mers.

Note: In this context, 'css' refers to a chromatin state sequence.

```python
from css_utility import *
prom_css_Kmer_by_cell(path='path/to/your/pickled/css', output_path='path/to/your/output', k=4)  # Replace '4' with your desired k-mer length
```

*Step 4*. Fine-tuning data preparation

Suppose users wish to compare promoter regions associated with varying expression levels of the nearest gene. In that case, we offer the following function to assist in preparing the data for promoter regions with the desired RPKM levels. It is assumed that users have organized an input directory containing subdirectories, each of which includes `.refFlat` files.

```python
from css_utility import *

# Function call to extract and save promoter regions with specified gene expression levels
extNsaveProm_g_exp(
    exp_gene_dir='path/to/your/parent_dir/of/refFlat',  # Parent directory containing subdirectories with refFlat files
    df_pickle_dir='path/to/your/pickled/css',  # Directory for pickled chromatin state sequences (CSS)
    output_path='path/to/your/output',  # Directory to save output files
    file_name='your_filename_suffix',  # Suffix for the output file names
    rpkm_val=50,  # RPKM threshold value, replace '50' with your desired RPKM level
    up_num=upstream_distance,  # Upstream distance from TSS, replace 'upstream_distance' with a numerical value
    down_num=downstream_distance,  # Downstream distance from TSS, replace 'downstream_distance' with a numerical value
    unit=200  # Unit size for chromatin states, usually 200 base pairs (bps)
)
```

Ensure your custom `.refFlat` file is formatted with tab-separated values including gene and transcript names, chromosome, strand, transcription and coding region positions, exon count, and exon start/end positions. 

Use the following function for promoters nearest to genes with an RPKM value of 0. 
Execute this code after running `extNsaveProm_g_exp` with `rpkm=0`.

```python
from css_utility import *

extNsaveNOTexp_by_compare(
    whole_gene_ref_path='path/to/gene/reference/file',  # Path to the reference file with whole gene annotations
    exp_ref_path='path/to/refFlat/for/rpkm0',  # Path to the refFlat files generated by extNsaveProm_g_exp with rpkm=0
    df_pickle_dir='path/to/your/pickled/css',  # Path to the directory containing pickled chromatin state sequences (CSS)
    output_path_ref='path/to/your/output/reference/file',  # Output path for the reference file of non-expressed (RPKM=0) genes
    output_path_prom='path/to/your/output',  # Output directory for promoter regions of non-expressed (RPKM=0) genes
    up_num=upstream_distance,  # Numerical value for the upstream distance from TSS
    down_num=downstream_distance,  # Numerical value for the downstream distance from TSS
    unit=200  # Chromatin state unit size, typically 200 base pairs (bps)
)

```

This function generates and saves reference files for genes not expressed (RPKM=0) and their associated promoter regions in the specified output directories.

Similarly to pre-training data, users can segment the data into k-mers using the function below, adjusting the `k` value to the desired number after extraction. 
For optimal computational efficiency, we recommend using 4-mers.

```python
from css_utility import *
prom_css_Kmer_by_cell(path='path/to/your/pickled/css', output_path='path/to/your/output', k=4)  # Replace '4' with your desired k-mer length
```


<br>

## 4. Training

For pre-training, fine-tuning, and to replicate our results, we recommend users download the `ChromBERT.zip` file from [Zenodo](URL) 
For organized access, please store the downloaded file in an appropriate directory, such as `examples/prom/pretrain_data`. 
In this section, we provide procedures for the 4-mer dataset. However, users have the flexibility to change the value of `k` by modifying the line `export KMER=4` in each script to suit their specific requirements.

#### 4-1. Pre-training
The pre-training script is located in the `examples/prom/script_pre/` directory. Users can adjust the file names within the script should they alter the directory or the name of the training data files. 

```bash
(chrombert) $ cd examples/prom/script_pre
(chrombert) $ bash run_4mer_pretrain.sh
```

#### 4-2. Fine-tuning
Following pre-training, the parameters are saved in the `examples/prom/pretrain_result/` directory. To replicate our fine-tuning results, users should place the files `train.tsv` and `dev.tsv` in the `examples/prom/ft_data/` directory. This location includes data for classifying promoter regions between genes that are highly expressed (RPKM > 50) and those that are not expressed (RPKM = 0).

```bash
(chrombert) $ cd examples/prom/script_ft
(chrombert) $ bash run_4mer_finetune.sh
```

#### 4-3. Prediction
To obtain an attention matrix for the prediction result, execute the scripts in the following order: First, run `run_4mer_pred1.sh`, followed by `run_4mer_pred2.sh`. It is essential to ensure that `run_4mer_pred1.sh` is executed before `run_4mer_pred2.sh`.

```bash
(chrombert) $ cd examples/prom/script_pred
(chrombert) $ bash run_4mer_pred1.sh

# After you get the result in the `examples/prom/prediction`

(chrombert) $ bash run_4mer_pred2.sh
```

<br>

## 5. Motif Detection and Clustering

The identification of chromatin state motifs can be categorized into two phases: Motif Detection and Motif Clustering. During the Motif Detection phase, chromatin state sequences that have high attention scores and are uniquely associated with the class of interest (for example, the promoter region) are identified and organized into a dataframe. Subsequently, these sequences undergo clustering through Dynamic Time Warping (DTW) in the Motif Clustering phase, leading to the identification of the definitive chromatin state motifs.

#### 5-1. Motif Detection

```bash
(chrombert) $ cd motif/prom
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
from css_utility import *
df_sequences=motif_init2df(input_path='path/to/your/init.csv')
```

To generate the predicted classes for each motif in the `init.csv` file by employing Dynamic Time Warping (DTW) along with k-means clustering, execute the code below:

```python
from css_utility import *
y_pred=motif_init2pred(input_path='path/to/your/init.csv', n_clusters=number_of_clusters)
```

*[Optional]* We provide a function to create an elbow plot, which aids in determining the optimal number of clusters for usability.

```python
from css_utility import *
motif_init2elbow(input_path='path/to/your/init.csv', n_start=1, n_end=25)
```
Note that `n_start` and `n_end` specify the range of cluster numbers to be tested. 

To obtain the clustered motifs in a DataFrame format, users can execute the following function:

```python
from css_utility import *
clustered_sequence=motif_init2class_df(input_path='path/to/your/init.csv', n_clusters=number_of_clusters)
```

For visualization purposes, users can understand the overall characteristics of clustered motifs by using the following function:

```python
from css_utility import *
motif_init2class_vis(input_path='path/to/your/init.csv', n_clusters=number_of_clusters)
```

*[Optional]* We provide an optional feature that facilitates the generation of a UMAP, designed to help users intuitively grasp the essential features of clustered motifs. 
It's important to note that users have the flexibility to configure the `n_neighbors` and `min_dist` parameters to suit their specific needs.

```python
from css_utility import *

motif_init2umap(input_path='path/to/your/init.csv',
                n_clusters=number_of_clusters,
                n_neighbors=size_you_want,
                min_dist=min_dist_you_want,
                random_state=random_seed)

```
<br>

## Contributing

We welcome contributions to [Project Name]. If you're interested in helping, please take a look at our [Contributing Guidelines](CONTRIBUTING.md) for more information on how to get started.

<br>

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details.

<br>

## Acknowledgements

Thanks to [Contributor Name] for [contribution], and to everyone who has submitted issues or pull requests.

<br>

## Contact

For any questions or feedback, feel free to [open an issue](link-to-issues) or contact me directly at [email@example.com](mailto:email@example.com).

---

Thank you for checking out [Project Name]. If you found this project useful, please consider starring it on GitHub to help it gain more visibility.







