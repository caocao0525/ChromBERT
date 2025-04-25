#!/usr/bin/env python
# coding: utf-8

# # CSS utility
# 
# Functions that can be exploited for data pre-processing and downstream analysis
# 
# ChromBERT has been expanded to include support for IHEC data

# In[94]:


# ### To convert the file into .py
# !jupyter nbconvert --to script css_utility_colab_dev_working.ipynb


# In[1]:


import os
import re
import sys
import random
import operator
import itertools
from itertools import cycle
import argparse
import time

import pickle
import glob
import ast
import collections
from collections import defaultdict, OrderedDict, Counter
import datetime
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
import matplotlib.transforms as transforms
import networkx as nx

import seaborn as sns
import scipy
from scipy.stats import norm
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.cluster.hierarchy import ClusterWarning
import warnings
from sklearn.metrics import confusion_matrix, classification_report
from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from tslearn.metrics import dtw

from tqdm import tqdm, notebook
from tqdm.notebook import tqdm_notebook

import logomaker
from wordcloud import WordCloud
# import stylecloud


# ### Useful Dictionaries

# In[2]:


state_dict={1:"A", 2:"B", 3:"C", 4:"D", 5:"E",6:"F",7:"G",8:"H" ,
                9:"I" ,10:"J",11:"K", 12:"L", 13:"M", 14:"N", 15:"O"}


# In[3]:


state_dict_18={1:"A", 2:"B", 3:"C", 4:"D", 5:"E",6:"F",7:"G",8:"H" ,
                9:"I" ,10:"J",11:"K", 12:"L", 13:"M", 14:"N", 15:"O", 16:"P", 17:"Q", 18:"R"}


# In[4]:


css_name=['TssA','TssAFlnk','TxFlnk','Tx','TxWk','EnhG','Enh','ZNF/Rpts',
          'Het','TssBiv','BivFlnk','EnhBiv','ReprPC','ReprPcWk','Quies']


# In[5]:


css_name_18=['TssA','TssAFlnk','TssFlnkU','TssFlnkD','Tx','TxWk','EnhG1','EnhG2','EnhA1','EnhA2','EnhWk','ZNF/Rpts',
          'Het','TssBiv','EnhBiv','ReprPC','ReprPcWk','Quies']


# In[6]:


css_dict=dict(zip(list(state_dict.values()), css_name))  # css_dict={"A":"TssA", "B":"TssAFlnk", ... }


# In[7]:


css_dict_18=dict(zip(list(state_dict_18.values()), css_name_18))  # css_dict={"A":"TssA", "B":"TssAFlnk", ... }


# In[8]:


# Color dict update using the info from https://egg2.wustl.edu/roadmap/web_portal/chr_state_learning.html
css_color_dict={'TssA':(255,0,0), # Red
                'TssAFlnk': (255,69,0), # OrangeRed
                'TxFlnk': (50,205,50), # LimeGreen
                'Tx': (0,128,0), # Green
                'TxWk': (0,100,0), # DarkGreen
                'EnhG': (194,225,5), # GreenYellow 
                'Enh': (255,255,0),# Yellow
                'ZNF/Rpts': (102,205,170), # Medium Aquamarine
                'Het': (138,145,208), # PaleTurquoise
                'TssBiv': (205,92,92), # IndianRed
                'BivFlnk': (233,150,122), # DarkSalmon
                'EnhBiv': (189,183,107), # DarkKhaki
                'ReprPC': (128,128,128), # Silver
                'ReprPCWk': (192,192,192), # Gainsboro
                'Quies': (240, 240, 240)}  # White -> bright gray 


# In[9]:


css_color_dict_18={ 'TssA':(255,0,0), # Red
                 'TssAFlnk': (255,69,0), # OrangeRed
                 'TssFlnkU': (255,69,0), # OrangeRed
                 'TssFlnkD': (255,69,0), # OrangeRed
                 'Tx': (0,128,0), # Green
                 'TxWk': (0,100,0), # DarkGreen
                 'EnhG1': (194,225,5), # GreenYellow 
                 'EnhG2': (194,225,5), # GreenYellow 
                 'EnhA1': (255,195,77), # Orange
                 'EnhA2': (255,195,77), # Orange
                 'EnhWk': (255,255,0),# Yellow
                 'ZNF/Rpts': (102,205,170), # Medium Aquamarine
                 'Het': (138,145,208), # PaleTurquoise
                 'TssBiv': (205,92,92), # IndianRed
                 'EnhBiv': (189,183,107), # DarkKhaki
                 'ReprPC': (128,128,128), # Silver
                 'ReprPcWk': (192,192,192), # Gainsboro
                 'Quies': (240, 240, 240)}  # White -> bright gray 


# In[10]:


state_col_dict_num={'A': (1.0, 0.0, 0.0),
 'B': (1.0, 0.271, 0.0),
 'C': (0.196, 0.804, 0.196),
 'D': (0.0, 0.502, 0.0),
 'E': (0.0, 0.392, 0.0),
 'F': (0.761, 0.882, 0.02),
 'G': (1.0, 1.0, 0.0),
 'H': (0.4, 0.804, 0.667),
 'I': (0.541, 0.569, 0.816),
 'J': (0.804, 0.361, 0.361),
 'K': (0.914, 0.588, 0.478),
 'L': (0.741, 0.718, 0.42),
 'M': (0.502, 0.502, 0.502),
 'N': (0.753, 0.753, 0.753),
 'O': (0.941, 0.941, 0.941)}


# In[11]:


state_col_dict_num_18={'A': (1.0, 0.0, 0.0),
 'B': (1.0, 0.271, 0.0),
 'C': (1.0, 0.271, 0.0),
 'D': (1.0, 0.271, 0.0),
 'E': (0, 0.502, 0),
 'F': (0, 0.392, 0),
 'G': (0.761, 0.882, 0.02),
 'H': (0.761, 0.882, 0.02),
 'I': (1.0, 0.765, 0.302),
 'J': (1.0, 0.765, 0.302),
 'K': (1.0, 1.0, 0),
 'L': (0.4, 0.804, 0.667),
 'M': (0.541, 0.569, 0.816),
 'N': (0.804, 0.361, 0.361),
 'O': (0.741, 0.718, 0.42),
 'P': (0.502, 0.502, 0.502),
 'Q': (0.753, 0.753, 0.753),
 'R': (0.941, 0.941, 0.941) }


# In[12]:


def colors2color_dec(css_color_dict):
    colors=list(css_color_dict.values())
    color_dec_list=[]
    for color in colors:
        color_dec=tuple(rgb_elm/255 for rgb_elm in color)
        color_dec_list.append(color_dec)        
    return color_dec_list


# **scale 0 to 1**

# In[13]:


state_col_dict=dict(zip(list(state_dict.values()),colors2color_dec(css_color_dict)))


# In[14]:


state_col_dict_18=dict(zip(list(state_dict_18.values()),colors2color_dec(css_color_dict_18)))


# **scale 0 to 255**

# In[15]:


state_col_255_dict=dict(zip(list(state_dict.values()),list(css_color_dict.values())))


# In[16]:


state_col_255_dict_18=dict(zip(list(state_dict_18.values()),list(css_color_dict_18.values())))


# **hexacode**

# In[17]:


hexa_state_col_dict={letter: "#{:02x}{:02x}{:02x}".format(*rgb) for letter,rgb in state_col_255_dict.items()}


# In[18]:


hexa_state_col_dict_18={letter: "#{:02x}{:02x}{:02x}".format(*rgb) for letter,rgb in state_col_255_dict_18.items()}


# **name instead of alphabets**

# In[19]:


css_name_col_dict=dict(zip(css_name,state_col_dict.values()))


# In[20]:


css_name_col_dict_18=dict(zip(css_name_18,state_col_dict_18.values()))


# ### Helper functions

# In[21]:


def flatLst(lst):
    flatten_lst=[elm for sublst in lst for elm in sublst]
    return flatten_lst


# In[22]:


### Produce colorful letter-represented chromatin state sequences
def colored_css_str_as_is(sub_str):   # convert space into space
    col_str=""
    for letter in sub_str:
        if letter==" ":
            col_str+=" "
        else:                
            for state in list(state_col_255_dict.keys()):
                if letter==state:
                    r=state_col_255_dict[letter][0]
                    g=state_col_255_dict[letter][1]
                    b=state_col_255_dict[letter][2]
                    col_letter="\033[38;2;{};{};{}m{}\033[38;2;255;255;255m".format(r,g,b,letter)
                    col_str+=col_letter
    return print("\033[1m"+col_str+"\033[0;0m") 


# In[23]:


def seq2kmer(seq, k):
    """
    Convert original sequence to kmers
    """
    kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
    kmers = " ".join(kmer)
    return kmers


# In[24]:


def kmer2seq(kmers):
    """
    Convert kmers to original sequence
    """
    kmers_list = kmers.split(" ")
    bases = [kmer[0] for kmer in kmers_list[0:-1]]
    bases.append(kmers_list[-1])
    seq = "".join(bases)
    assert len(seq) == len(kmers_list) + len(kmers_list[0]) - 1
    return seq


# In[25]:


# create dataframe from bed file
# bed file here means: EXXX_15_coreMarks_stateno.bed

def bed2df_as_is(filename):    

    """Create dataframe from the .bed file, as is.
    Dataframe contains following columns:
    chromosome |  start |  end  | state """

    df_raw=pd.read_csv(filename, sep='\t', lineterminator='\n', header=None, low_memory=False)
    df=df_raw.rename(columns={0:"chromosome",1:"start",2:"end",3:"state"})
    df=df[:-1]
    df["start"]=pd.to_numeric(df["start"])
    df["end"]=pd.to_numeric(df["end"])

    return df


# ### Main functions

# In[27]:


def bed2df_expanded(filename, state_num=15):
    """Create an expanded dataframe from the .bed file.
    Dataframe contains following columns:
    chromosome |  start |  end  | state | length | unit | state_seq | state_seq_full"""
    if not os.path.exists(filename):
        raise FileNotFoundError("Please provide a valid file path.")

    df_raw=pd.read_csv(filename, sep='\t', lineterminator='\n', header=None, low_memory=False)
    df=df_raw.rename(columns={0:"chromosome",1:"start",2:"end",3:"state"})
    df=df[:-1]
    df["start"]=pd.to_numeric(df["start"])
    df["end"]=pd.to_numeric(df["end"])
    df["state"]=pd.to_numeric(df["state"])
    df["length"]=df["end"]-df["start"]
    df["unit"]=(df["length"]/200).astype(int)  # chromatin state is annotated every 200 bp (18th May 2022)
    if state_num==18:
        df["state_seq"]=df["state"].map(state_dict_18)
        df["state_seq_full"]=df["unit"]*df["state_seq"]
    else:
        df["state_seq"]=df["state"].map(state_dict)
        df["state_seq_full"]=df["unit"]*df["state_seq"]

    return df 


# In[28]:


def unzipped_to_df(path_unzipped, output_path="./", state_num=15):
    """
    Store the DataFrame converted from .bed file, cell-wise
    - path_unzipped: the directory of your .bed files
    - output_path: the directory where the file will be saved. Dafaults to the current working directory.
    """
    unzipped_epi=sorted(os.listdir(path_unzipped))
    unzipped_epi_files=[os.path.join(path_unzipped,file) for file in unzipped_epi]
    for file in unzipped_epi_files:
        cell_id=file.split("/")[-1][:4]
        # print(cell_id) ###### for test

        output_name=os.path.join(output_path,cell_id+"_df_pickled.pkl")
        df=bed2df_expanded(file, state_num=state_num)
        df.to_pickle(output_name)
        # if cell_id=="E002":  ###### for test
        #     break
    return print("Files saved to {}".format(output_path))
# unzipped_to_df(unzipped_epi_files, output_path="../database/roadmap/df_pickled/")


# In[30]:


# # test for unzipped_to_df
# path_unzipped='../database/bed/unzipped'
# test_unzipped_to_df=unzipped_to_df(path_unzipped,output_path="../database/final_test")
# # test passed


# In[29]:


# first, learn where one chromosome ends in the df
# this is just a prerequisite function for df2css_chr

def df2chr_index(df):

    """Create a list of smaller piece of string of the state_seq_full per chromosome
    This function generates a list of chromatin state sequence strings chromosome-wise"""

    total_row=len(df)
    chr_len=[]
    chr_check=[]
    chr_index=[]

    for i in range(total_row):
        if (df["start"].iloc[i]==0) & (i >0):
            chr_len.append(df["end"].iloc[i-1]) # chr_len stores the end position of each chromosome
            chr_check.append(df["start"].iloc[i]) # for assertion : later check chr_check are all zero
            chr_index.append(i-1) # the index (row number)

    end_len=df["end"].iloc[-1] # add the final end position
    end_index=total_row-1 # add the final end index (row number)

    chr_len.append(end_len)
    chr_index.append(end_index)

    assert len(chr_len)==df["chromosome"].nunique() #assert the length of the list corresponds to no. of chromosome
    assert len(chr_index)==df["chromosome"].nunique()

    return chr_index


# In[30]:


def df2chr_df(df):

    """Create a list of dataframes, each of which containing 
    the the whole expanded type of dataframe per chromosome"""

    start=0
    df_chr_list=[]
    chr_index=df2chr_index(df)

    for index in chr_index:
        df_chr=df[start:index+1] # note that python [i:j] means from i to j-1
        chr_name=df["chromosome"].iloc[start] # string, such as chr1, chr2, ...
        df_name='df_'+chr_name  # the chromosome-wise data stored like df_chr1, df_chr2, ...
        locals()[df_name]=df_chr # make a string into a variable name
        df_chr_list.append(df_chr)
        start=index+1

    return df_chr_list   # elm is the df of each chromosome


# In[31]:


# make a long string of the css (unit length, not the real length)
def df2unitcss(df):
    """
    Create a list of 24 lists of chromatin states in string, reduced per 200 bps
    """
    df_lst_chr=df2chr_df(df)
    # remove the microchondria DNA from df_lst_chr
    if df_lst_chr[-3]["chromosome"].iloc[0]=="chrM":
        del df_lst_chr[-3]
#         assert df_lst_chr[-3]["chromosome"].iloc[0]=="chr22"
#     else:   
#         assert df_lst_chr[-3]["chromosome"].iloc[0]=="chr22"
    all_unit_css=[]
    for i in range(len(df_lst_chr)):
        df_chr=df_lst_chr[i]
        css_chr=''
        for j in range(len(df_chr)):
            css_chr+=df_chr["unit"].iloc[j]*df_chr["state_seq"].iloc[j]
        all_unit_css.append(css_chr)  
    return all_unit_css


# In[32]:


# # test for df2unitcss
# with open("../database/final_test/E001_df_pickled.pkl","rb") as f:
#     test_df=pickle.load(f)
# all_unit_css=df2unitcss(test_df)
# print(len(all_unit_css))
# print(type(all_unit_css))
# # test passed


# In[33]:


# #### To save css_unit from df_pickled in bulk
# df_path = "../database/df_pickled"
# output_path = "../database/css_unit_pickled"

# for file_name in os.listdir(df_path):
#     file_path = os.path.join(df_path, file_name)  # Full path to the file
#     with open(file_path, "rb") as f:
#         df = pickle.load(f)

#         # Extract file_id only for files starting with 'IHEC'
#         if file_name[:4] == 'IHEC':
#             file_id = file_name[:14]
#         else:
#             continue  # Skip files that do not match the condition

#         # Process the dataframe and save the output
#         unit_css = df2unitcss(df)
#         output_file_name = os.path.join(output_path, f"{file_id}_unitcss.pkl")
#         with open(output_file_name, "wb") as g:
#             pickle.dump(unit_css, g)
# ## this has been conducted, no need to execute again


# In[36]:


def shorten_string(s, factor):
    # This regular expression matches groups of the same character.
    pattern = re.compile(r'(.)\1*')

    # This function will be used to replace each match.
    def replacer(match):
        # The group that was matched.
        group = match.group()

        # Calculate the new length, rounding as necessary.
        new_length = round(len(group) / factor)

        # Return the character repeated the new number of times.
        return group[0] * new_length

    # Use re.sub to replace each match in the string.
    return pattern.sub(replacer, s)


# In[37]:


def Convert2unitCSS_main_new(css_lst_all, unit=200):# should be either css_gene_lst_all or css_Ngene_lst_all
    """
    Input: css_gene_lst_all or css_Ngene_lst_all, the list of chromosome-wise list of the css in genic, intergenic regions.
    Output: css_gene_unit_lst_all or css_Ngene_unit_lst_all
    """
    reduced_all=[]
    for i in range(len(css_lst_all)):
        reduced_chr=[]
        for j in range(len(css_lst_all[i])):
            reduced=shorten_string(css_lst_all[i][j], unit)
            reduced_chr.append(reduced)
        reduced_all.append(reduced_chr)
    return reduced_all


# In[38]:


# make a long string of the css (not using unit, but the real length)
def df2longcss(df):
    """
    Create a list of 24 lists of chromatin states in string, in real length
    """
    df_lst_chr=df2chr_df(df)
    # remove the microchondria DNA from df_lst_chr
    if df_lst_chr[-3]["chromosome"].iloc[0]=="chrM":
        del df_lst_chr[-3]
#         assert df_lst_chr[-3]["chromosome"].iloc[0]=="chr22"
    elif df_lst_chr[-2]["chromosome"].iloc[0]=="chrM":
        del df_lst_chr[-2]
#         assert df_lst_chr[-3]["chromosome"].iloc[0]=="chr22"

    all_css=[]
    for i in range(len(df_lst_chr)):
        df_chr=df_lst_chr[i]
        css_chr=''
        for j in range(len(df_chr)):
            css_chr+=df_chr["length"].iloc[j]*df_chr["state_seq"].iloc[j]
        all_css.append(css_chr)  
    return all_css


# In[39]:


# function for preprocess the whole gene data and produce chromosome-wise gene lists
# each element is dataframe

# def whGene2GLChr(whole_gene_file='../database/RefSeq/RefSeq.WholeGene.bed'):
def whGene2GLChr(whole_gene_file):
    """
    For pre-processing the whole gene data and produce chromosome-wise gene lists
    """
    print("Extracting the gene file ...")
    g_fn=whole_gene_file
    g_df_raw=pd.read_csv(g_fn, sep='\t', lineterminator='\n', header=None, low_memory=False)
    g_df_int=g_df_raw.rename(columns={0:"chromosome",1:"TxStart",2:"TxEnd",3:"name",4:"unk0",
                                  5:'strand', 6:'cdsStart', 7:'cdsEnd',8:"unk1",9:"exonCount",
                                  10:"unk2",11:"unk3"})
    g_df=g_df_int[["chromosome","TxStart","TxEnd","name"]]

    # Remove other than regular chromosomes
    chr_lst=['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10',
             'chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19',
             'chr20','chr21','chr22','chrX','chrY']
    g_df=g_df.loc[g_df["chromosome"].isin(chr_lst)]

    # Create a list of chromosome-wise dataframe 
    g_df_chr_lst=[]
    for num in range(len(chr_lst)):
        chr_num=chr_lst[num]
        g_chr_df='g_'+chr_num
        locals()[g_chr_df]=g_df[g_df["chromosome"]==chr_num]
        g_chr_df=locals()[g_chr_df]
        g_chr_df=g_chr_df.sort_values("TxStart")
        g_df_chr_lst.append(g_chr_df)
    print("Done!")

    return g_df_chr_lst


# In[40]:


#### Merging the gene table #### modified June. 29. 2023

def merge_intervals(df_list):
    merged_list = []  # List to hold merged DataFrames

    for df in df_list:
        # Sort by 'TxStart'
        df = df.sort_values(by='TxStart')

        # Initialize an empty list to store the merged intervals
        merged = []

        # Iterate through the rows in the DataFrame
        for _, row in df.iterrows():
            # If the list of merged intervals is empty, or the current interval does not overlap with the previous one,
            # append it to the list
            if not merged or merged[-1]['TxEnd'] < row['TxStart']:
                merged.append({'TxStart': row['TxStart'], 'TxEnd': row['TxEnd']})  # Only keep 'TxStart' and 'TxEnd'
            else:
                # Otherwise, there is an overlap, so we merge the current and previous intervals
                merged[-1]['TxEnd'] = max(merged[-1]['TxEnd'], row['TxEnd'])

        # Convert the merged intervals back into a DataFrame and append it to the list
        merged_list.append(pd.DataFrame(merged))

    return merged_list  # a list of DF, containing only TxStart and TxEnd


# In[41]:


def remove_chrM_and_trim_gene_file_accordingly(whole_gene_file,df):

    ########### Gene without overlap ###########
    g_df_chr_lst=whGene2GLChr(whole_gene_file)  ##### fixed June 29. 2023
    new_gene_lst_all=merge_intervals(g_df_chr_lst) ##### fixed June 29. 2023
    ############################################################

    #### Remove chrM ###########################################
    contains_chrM = df['chromosome'].str.contains('chrM').any()  #check whether it contains M
    if contains_chrM:
        df= df[~df['chromosome'].str.contains('chrM')]

    contains_chrY = df['chromosome'].str.contains('chrY').any()

    ##### if the target file does not contain Y, remove Y in the gene list file
    if not contains_chrY:
        new_gene_lst_all=new_gene_lst_all[:-1] ## the final element is for Y
    ############################################################

    assert len(df["chromosome"].unique())==len(new_gene_lst_all)
    return new_gene_lst_all, df


# In[42]:


def save_TSS_by_loc(whole_gene_file, input_path="./",output_path="./",file_name="upNkdownNk", up_num=2000, down_num=4000, unit=200):
    """
    extract TSS region by location estimation. 
    input: (1) whole_gene_file: the raw gene bed file (e.g. RefSeq.WholeGene.bed)
           (2) input_path: pickled df per cell
    output: save tss_by_loc_css_unit_all at the output path
    """
    file_lst=os.listdir(input_path)
    all_files=[os.path.join(input_path,file) for file in file_lst]
    for file in all_files:
        cell_num=file.split("/")[-1][:4]
#         if cell_num=="E002": break  # for test 
        with open(file,"rb") as f:
            df_pickled=pickle.load(f)
        # align the gene file and the df file according to their availability(some cells does not have chr Y)
        new_gene_lst_all, trimmed_df=remove_chrM_and_trim_gene_file_accordingly(whole_gene_file,df_pickled)
        css_lst_chr = df2longcss(trimmed_df) # list of long css per chromosome
        total_chr = len(new_gene_lst_all)       
        tss_by_loc_css_all = []
        for i in range(total_chr):
            gene_start_lst = new_gene_lst_all[i]["TxStart"]
            css_lst = css_lst_chr[i]
            tss_by_loc_css_chr = []
            for j in range(len(gene_start_lst)):
                gene_start = gene_start_lst[j]
                win_start = max(0, gene_start - up_num)  # use max to prevent negative index
                win_end = min(len(css_lst), gene_start + down_num)  # use min to prevent index out of range
                tss_by_loc_css = css_lst[win_start:win_end]
                tss_by_loc_css_chr.append(tss_by_loc_css)               
            tss_by_loc_css_all.append(tss_by_loc_css_chr)
        tss_by_loc_css_unit_all=Convert2unitCSS_main_new(tss_by_loc_css_all, unit=unit)  
        output_file_name=os.path.join(output_path,cell_num+"_prom_"+file_name+".pkl")
        with open(output_file_name,"wb") as g:
            pickle.dump(tss_by_loc_css_unit_all,g)

    return print("All done!") #tss_by_loc_css_unit_all


# In[43]:


# with open ("../database/roadmap/df_pickled/E003_df_pickled.pkl", "rb") as f:
#     df_test=pickle.load(f)


# In[44]:


################## for Colab test


# In[45]:


def save_prom_by_loc_per_cell(whole_gene_file, df, up_num=2000, down_num=4000, unit=200):
    """
    Extract promoter regions around TSS by location estimation.

    Parameters:
        whole_gene_file: The raw gene BED file (e.g., RefSeq.WholeGene.bed)
        df: DataFrame containing the required data (instead of loading a pickled file)
        up_num: Number of bases upstream from TSS to extract (default: 2000)
        down_num: Number of bases downstream from TSS to extract (default: 4000)
        unit: Window size for conversion (default: 200)

    Returns:
        tss_by_loc_css_unit_all: Extracted promoter region data
    """
    # Align the gene file and the df file according to their availability (some cells do not have chr Y)
    new_gene_lst_all, trimmed_df = remove_chrM_and_trim_gene_file_accordingly(whole_gene_file, df)
    css_lst_chr = df2longcss(trimmed_df)  # list of long css per chromosome
    total_chr = len(new_gene_lst_all)       
    tss_by_loc_css_all = []

    for i in range(total_chr):
        gene_start_lst = new_gene_lst_all[i]["TxStart"]
        css_lst = css_lst_chr[i]
        tss_by_loc_css_chr = []

        for j in range(len(gene_start_lst)):
            gene_start = gene_start_lst[j]
            win_start = max(0, gene_start - up_num)  # use max to prevent negative index
            win_end = min(len(css_lst), gene_start + down_num)  # use min to prevent index out of range
            tss_by_loc_css = css_lst[win_start:win_end]
            tss_by_loc_css_chr.append(tss_by_loc_css)               

        tss_by_loc_css_all.append(tss_by_loc_css_chr)

    prom_by_loc_per_cell_css_unit_all = Convert2unitCSS_main_new(tss_by_loc_css_all, unit=unit)  
    return prom_by_loc_per_cell_css_unit_all


# In[46]:


# whole_gene_file='../database/RefSeq/RefSeq.WholeGene.bed'
# prom_by_loc_per_cell_css_unit_all=save_prom_by_loc_per_cell(whole_gene_file, df_test, up_num=2000, down_num=4000, unit=200)


# In[89]:


def process_prom_list(prom_by_loc_per_cell_css_unit_all, k, low_signal):
    """
    Process TSS list by flattening, converting to k-mers, removing all-O k-mers, and filtering empty entries.
    Parameters:
        prom_by_loc_per_cell_css_unit_all: List of lists of sequences (strings) from TSS extraction
        k: K-mer length   
    Returns:
        Processed list of k-mers
    """
    print("Finalizing the list ...")
    flat_list = flatLst(prom_by_loc_per_cell_css_unit_all)  # Flatten the list
    kmer_list = [seq2kmer(seq, k) for seq in flat_list if len(seq) >= k]  # Convert to k-mers
    filtered_kmers = [" ".join([kmer for kmer in kmers.split() if str(low_signal) * k not in kmer]) 
                      for kmers in kmer_list]  # Remove k-mers with all-O letters   
    final_prom_list=[entry for entry in filtered_kmers if entry]  # Remove empty entries
    print("Done!")
    return final_prom_list


######## so this can be used as Colab data for promoter pretraining.. but now how to get the fine-tuning?
#### dont forget to upload the gene file


# In[48]:


# final_prom_list=process_prom_list(prom_by_loc_per_cell_css_unit_all, 4, "O")
# ### then later save 'final_prom_list' to /content/.. directory on colab



# In[ ]:





# In[ ]:


# with open("../database/final_test/E001_df_pickled.pkl","rb") as f:
#     test_df=pickle.load(f)


# In[ ]:





# In[ ]:





# In[49]:


# # test for save_TSS_by_loc
# whole_gene_file='../database/RefSeq/RefSeq.WholeGene.bed'
# save_TSS_by_loc(whole_gene_file=whole_gene_file, input_path="../database/roadmap/df_pickled/",output_path="../database/final_test/",file_name="up2kdown4k", up_num=2000, down_num=4000, unit=200)
# # test passed


# In[34]:


#######################################################################
# [IHEC] Pretrain data preprocessing and storing (for 18-state version)
#######################################################################

# For promoter region extraction for IHEC data

def save_TSS_by_loc_IHEC(whole_gene_file, input_path="./",output_path="./",file_name="upNkdownNk", up_num=2000, down_num=4000, unit=200):
    """
    extract TSS region by location estimation. 
    input: (1) whole_gene_file: the raw gene bed file (e.g. RefSeq.WholeGene.bed)
           (2) input_path: pickled df per cell
    output: save tss_by_loc_css_unit_all at the output path
    """
    file_lst=os.listdir(input_path)
    all_files=[os.path.join(input_path,file) for file in file_lst]
    for file in all_files:
        cell_num=file.split("/")[-1][:14]
        # if cell_num=="IHECRE00000002": break  # for test 
        with open(file,"rb") as f:
            df_pickled=pickle.load(f)
        # align the gene file and the df file according to their availability(some cells does not have chr Y)
        new_gene_lst_all, trimmed_df=remove_chrM_and_trim_gene_file_accordingly(whole_gene_file,df_pickled)
        css_lst_chr = df2longcss(trimmed_df) # list of long css per chromosome
        total_chr = len(new_gene_lst_all)       
        tss_by_loc_css_all = []
        for i in range(total_chr):
            gene_start_lst = new_gene_lst_all[i]["TxStart"]
            css_lst = css_lst_chr[i]
            tss_by_loc_css_chr = []
            for j in range(len(gene_start_lst)):
                gene_start = gene_start_lst[j]
                win_start = max(0, gene_start - up_num)  # use max to prevent negative index
                win_end = min(len(css_lst), gene_start + down_num)  # use min to prevent index out of range
                tss_by_loc_css = css_lst[win_start:win_end]
                tss_by_loc_css_chr.append(tss_by_loc_css)               
            tss_by_loc_css_all.append(tss_by_loc_css_chr)
        tss_by_loc_css_unit_all=Convert2unitCSS_main_new(tss_by_loc_css_all, unit=unit)  
        tss_by_loc_css_unit_all_flat=flatLst(tss_by_loc_css_unit_all)
        output_file_name=os.path.join(output_path,cell_num+"_prom_"+file_name+".pkl")
        with open(output_file_name,"wb") as g:
            pickle.dump(tss_by_loc_css_unit_all_flat,g)

    return print("All done!") #tss_by_loc_css_unit_all_flat


# In[51]:


# save_TSS_by_loc_IHEC(whole_gene_file="../database/RefSeq/hg38/RefSeq.WholeGene.bed", 
#                 input_path="../database/df_pickled/",
#                 output_path="../database/prom_IHEC/prom_css_pickled/up2kdown4k",
#                 file_name="up2kdown4k", 
#                 up_num=2000, down_num=4000, unit=200)


# In[ ]:





# In[ ]:





# In[ ]:





# In[52]:


# Pretrain data preprocessing and storing

# Preprocessing for removing continuous O state for pretrain dataset
# 1. Save the CSS per cell, per chromosome
def save_css_by_cell_wo_continuous_15state(path_to_css_unit_pickled, output_path,k=4):
    # read files from css_unit_pickled
    files=os.listdir(path_to_css_unit_pickled)
    file_path_lst=[os.path.join(path_to_css_unit_pickled,file) for file in files]
    for file_path in file_path_lst:
        file_name=os.path.basename(file_path)
        if file_name[0] == 'E' and file_name[1:4].isdigit():
            file_id = file_name[:4]
        else:
            pass
        # ##########################
        # if str(file_id)=="E003":
        #     break  # for test
        # ##########################
        with open(file_path,"rb") as f:
            css=pickle.load(f)
        css_kmer=[]
        for css_chr in css:
            css_chr_kmer=seq2kmer(css_chr,k)
            target_to_remove="O"*k   # get rid of the word with continuous 15th state "o"
            css_chr_kmer_trim = css_chr_kmer.replace(target_to_remove, "")
            # clean up extra spaces
            css_chr_kmer_trim = ' '.join(css_chr_kmer_trim.split())
            css_kmer.append(css_chr_kmer_trim)
        output_file_name=os.path.join(output_path,file_id+"_unitcss_wo_all"+str(k)+"O_state.pkl")    
        with open(output_file_name, "wb") as g:
            pickle.dump(css_kmer, g)  # note that it is chromosome-wise list (each element corresponds to each chromosome)

        print("trimmed css by cell saved: ",file_id)
    return 


# In[53]:


# Preprocessing for removing continuous O state for pretrain dataset
# 2-2. Concatenate all the cells and create one .txt file (IHEC file version)
# (Note. new line joining chromosome-wise and cell-wise)
def kmerCSS_to_pretrain_data_ihec(path_to_kmer_css_unit_pickled,output_path):
    files=os.listdir(path_to_kmer_css_unit_pickled)
    file_path_lst=[os.path.join(path_to_kmer_css_unit_pickled,file) for file in files]

    css_all=[]
    for file_path in file_path_lst:
        file_name=os.path.basename(file_path)
        if file_name[:4] == 'IHEC':
            file_id = file_name[:14]
        else:
            pass
        # ##########################
        # if str(file_id)=="E003":
        #     break  # for test
        # # ##########################
        with open(file_path,"rb") as f:
            css=pickle.load(f)

        css_per_cell='\n'.join(css)   # join the chromosome by new line

        css_all.append(css_per_cell)   

    css_all_cell='\n'.join(css_all)  # join the cell by new line

    output_name=os.path.join(output_path,"pretrain_genome_all.txt") 
    with open(output_name, "w") as g:
        g.write(css_all_cell)

    return 


# #### Expansion to IHEC data 
# (1) Pretraining data preparation for all genomic regions

# In[54]:


#######################################################################
# [IHEC] Pretrain data preprocessing and storing (for 18-state version)
#######################################################################

# Preprocessing for removing continuous R state for pretrain dataset
# 1-1. Save the CSS per cell, per chromosome

# def save_css_by_cell_wo_continuous_18state(path_to_css_unit_pickled, output_path, k=4, max_tokens=510):
#     """
#     Preprocesses and saves CSS per cell, per chromosome, removing continuous R states 
#     and splitting excess tokens into new lines, saving output as .txt.

#     Args:
#         path_to_css_unit_pickled (str): Path to the pickled CSS unit files.
#         output_path (str): Path to save the processed files.
#         k (int): Length of the k-mer. Default is 4.
#         max_tokens (int): Maximum number of tokens per line. Default is 510.
#     """
#     # Read files from css_unit_pickled
#     files = os.listdir(path_to_css_unit_pickled)
#     file_path_lst = [os.path.join(path_to_css_unit_pickled, file) for file in files]

#     for file_path in file_path_lst:
#         file_name = os.path.basename(file_path)
#         if file_name[:4] == 'IHEC':
#             file_id = file_name[:14]
#         else:
#             continue

#         with open(file_path, "rb") as f:
#             css = pickle.load(f)

#         css_kmer = []
#         for css_chr in css:
#             css_chr_kmer = seq2kmer(css_chr, k)
#             target_to_remove = "R" * k  # Get rid of the word with continuous R state
#             css_chr_kmer_trim = css_chr_kmer.replace(target_to_remove, "")
#             # Clean up extra spaces
#             css_chr_kmer_trim = ' '.join(css_chr_kmer_trim.split())
#             # Split into chunks of max_tokens
#             tokens = css_chr_kmer_trim.split()
#             for i in range(0, len(tokens), max_tokens):
#                 chunk = tokens[i:i + max_tokens]
#                 css_kmer.append(' '.join(chunk))

#         # Save output as .txt file
#         output_file_name = os.path.join(output_path, file_id + "_test_" + str(k) + "R_state.txt")
#         with open(output_file_name, "w") as g:
#             for line in css_kmer:
#                 g.write(line + "\n")  # Write each chunk as a line in the file

#         print("Trimmed and chunked CSS by cell saved as .txt: ", file_id)

#     return


#######################################################################
# [IHEC] Pretrain data preprocessing and storing (for 18-state version)
#######################################################################

# For promoter region extraction for IHEC data

################### This is a revised version to handle the list of list  (what we used for whole genome area) 
################### And modified to handle the faltten list (for promoter area) and an empty string element
################### When replace the original one, remain the original as commented 
def save_css_by_cell_wo_continuous_18state(path_to_css_unit_pickled, output_file, k=4, max_tokens=510):
    """
    Preprocesses CSS data and saves all processed content into a single text file, 
    removing continuous R states and splitting excess tokens into new lines.

    Args:
        path_to_css_unit_pickled (str): Path to the pickled CSS unit files.
        output_file (str): Path to save the consolidated processed file.
        k (int): Length of the k-mer. Default is 4.
        max_tokens (int): Maximum number of tokens per line. Default is 510.
    """
    # Read files from css_unit_pickled
    files = os.listdir(path_to_css_unit_pickled)
    file_path_lst = [os.path.join(path_to_css_unit_pickled, file) for file in files]

    all_css_kmer = []  # Collect all processed data here

    for file_path in file_path_lst:
        file_name = os.path.basename(file_path)
        if file_name[:4] == 'IHEC':
            file_id = file_name[:14]
        else:
            continue

        with open(file_path, "rb") as f:
            css = pickle.load(f)

        # Handle flat list input
        if isinstance(css[0], str):  # If `css` is a flat list
            css = [css]  # Wrap into a single list to process as one "chromosome"

        for css_chr_list in css:  # Iterate over sublists
            for css_chr in css_chr_list:  # Iterate over strings
                if not css_chr.strip():  # Skip empty strings
                    continue
                css_chr_kmer = seq2kmer(css_chr, k)
                target_to_remove = "R" * k  # Get rid of the word with continuous R state
                css_chr_kmer_trim = css_chr_kmer.replace(target_to_remove, "")
                # Clean up extra spaces
                css_chr_kmer_trim = ' '.join(css_chr_kmer_trim.split())
                # Split into chunks of max_tokens
                tokens = css_chr_kmer_trim.split()
                for i in range(0, len(tokens), max_tokens):
                    chunk = tokens[i:i + max_tokens]
                    all_css_kmer.append(' '.join(chunk))

    # Save all processed data into a single text file
    with open(output_file, "w") as g:
        for line in all_css_kmer:
            g.write(line + "\n")  # Write each chunk as a line in the file

    print("All CSS data saved into a single file: ", output_file)

    return


# In[55]:


# # css_unit_path="../database/css_unit_pickled_test/"
# # wo_R_output_path="../database"
# # save_css_by_cell_wo_continuous_18state(css_unit_path, wo_R_output_path, k=4, max_tokens=510)


##### latest check (In this case no need to conduct save_and_concatenate_css)
# # to save the promoter regions only
# save_css_by_cell_wo_continuous_18state(path_to_css_unit_pickled="../database/prom_IHEC/prom_css_pickled/up2kdown4k", 
#                                        output_file="../database/prom_IHEC/prom_kmer_wo_cont_R/up2kdown4k/promoter_ihec_all_4mer_wo_4R.txt", 
#                                        k=4, 
#                                        max_tokens=510)


# In[56]:


#######################################################################
# [IHEC] Pretrain data preprocessing and storing (for 18-state version)
#######################################################################

# 1-2. Save the CSS of all cells by concatenating them
# no need to execute if save_css_by_cell_wo_continuous_18state produced a single text file

def save_and_concatenate_css(path_to_css_unit_pickled, concatenated_output_file, k=4, max_tokens=510):
    """
    Preprocesses CSS per cell, per chromosome, removing continuous R states,
    splitting excess tokens into new lines, and saving all processed CSS into a single file.

    Args:
        path_to_css_unit_pickled (str): Path to the pickled CSS unit files.
        concatenated_output_file (str): Path to save the concatenated .txt file.
        k (int): Length of the k-mer. Default is 4.
        max_tokens (int): Maximum number of tokens per line. Default is 510.
    """
    # Read files from css_unit_pickled
    files = os.listdir(path_to_css_unit_pickled)
    file_path_lst = [os.path.join(path_to_css_unit_pickled, file) for file in files]

    all_kmers = []  # List to store all processed css_kmers for concatenation

    for file_path in file_path_lst:
        file_name = os.path.basename(file_path)
        if file_name[:4] == 'IHEC':
            file_id = file_name[:14]
        else:
            continue

        with open(file_path, "rb") as f:
            css = pickle.load(f)

        for css_chr in css:
            css_chr_kmer = seq2kmer(css_chr, k)
            target_to_remove = "R" * k  # Get rid of the word with continuous R state
            css_chr_kmer_trim = css_chr_kmer.replace(target_to_remove, "")
            # Clean up extra spaces
            css_chr_kmer_trim = ' '.join(css_chr_kmer_trim.split())
            # Split into chunks of max_tokens
            tokens = css_chr_kmer_trim.split()
            for i in range(0, len(tokens), max_tokens):
                chunk = tokens[i:i + max_tokens]
                all_kmers.append(' '.join(chunk))  # Add each chunk directly to all_kmers

    # Save concatenated output to a single .txt file
    with open(concatenated_output_file, "w") as concat_file:
        for line in all_kmers:
            concat_file.write(line + "\n")

    print(f"All processed CSS concatenated and saved to {concatenated_output_file}")



# In[57]:


# save_and_concatenate_css(
#     path_to_css_unit_pickled="../database/css_unit_pickled",
#     concatenated_output_file="../database/pretrain/IHEC_all_genome_cut_4mer_wo_R_all_concatenated_css.txt",
#     k=4,
#     max_tokens=510
# )


# In[58]:


######################################################
#   Gene expression prediction dataset preparation   #
######################################################


# In[59]:


# 1-1. Extract the sequence and rpkm value as a dataframe (for a single cell)
# [Note] This is for the TxStart to TxEnd regions!


def create_css_with_rpkm_df(file_path, css_unit_path, unit=200):
    """
    Processes chromatin state data and extracts chromatin state sequences per chromosome.
    1. Read the css_unit_path (the whole css_unit from genome)
    2. Read the EXXX_processed.tsv file which contains gene loci with RPKM value
    3. Extract the gene loci in css_unit with RPKM value
    4. Remove all low-signal entries, like "OOOOOO"
    4. Create dataframe with columns "sequence" and "RPKM"

    Args:
        file_path (str): Path to the input EXXX_processed.tsv file.
        css_unit_path (str): Path to the css_unit pickle file.
        unit (int): Division factor for TxStart and TxEnd.

    Returns:
        pd.DataFrame: A DataFrame with "sequence" and "RPKM" columns.
    """
    # Step 1: Read the input file
    if file_path.endswith('.pkl'):
        with open(file_path, 'rb') as f:
            df = pickle.load(f)
    else:
        col_name = ["gene_id", "chromosome", "strand", "TxStart", "TxEnd", "RPKM"]
        df = pd.read_csv(file_path, names=col_name, sep="\t")

    # Step 2: Filter out unwanted rows
    df_filtered = df[df['gene_id'] != 'name'].copy()
    df_filtered = df_filtered[df_filtered['chromosome'] != 'chrM'].copy()

    # Step 3: Convert TxStart and TxEnd from strings to integers
    df_filtered['TxStart'] = pd.to_numeric(df_filtered['TxStart'], errors='coerce')
    df_filtered['TxEnd'] = pd.to_numeric(df_filtered['TxEnd'], errors='coerce')

    # Divide by 'unit', round, and convert to integers
    df_filtered['TxStart'] = (df_filtered['TxStart'] / unit).round().astype(int)
    df_filtered['TxEnd'] = (df_filtered['TxEnd'] / unit).round().astype(int)

    # Step 4: Load css_unit
    with open(css_unit_path, "rb") as gg:
        css_unit = pickle.load(gg)

    # Step 5: Extract chromatin state sequences and keep RPKM
    chromatin_sequences = []
    rpkm_values = []
    chromosome_map = {f"chr{i+1}" if i < 22 else ("chrX" if i == 22 else "chrY"): i for i in range(len(css_unit))}

    for _, row in df_filtered.iterrows():
        chromosome_index = chromosome_map.get(row['chromosome'])
        if chromosome_index is not None:
            start = row['TxStart'] - 1  # Adjust for Python's 0-based indexing
            end = row['TxEnd']  # Include end index
            sequence = css_unit[chromosome_index][start:end]
            chromatin_sequences.append(sequence)
            rpkm_values.append(row['RPKM'])

    # Step 6: Create the DataFrame
    result_df = pd.DataFrame({
        "sequence": chromatin_sequences,
        "RPKM": rpkm_values
    })

    # Step 7: Remove low-signal entries (e.g., sequences with only "O")
    result_df = result_df[result_df["sequence"].apply(lambda seq: not all(char == "O" for char in seq))]

    return result_df


# In[60]:


# # Example usage
# file_path = "../database_exp/RNA/hg19/gene_expression/sorted_output/E003_processed.tsv"
# css_unit_path = "../chromatin_state/database/roadmap/css_unit_pickled/E003_unitcss_woChrM.pkl"
# result_df = create_css_with_rpkm_df(file_path, css_unit_path)

# # Display the first few rows
# result_df.head()


# In[61]:


# 1-2. Prepare cell-wise dataframe of chromatin state sequence and rpkm pairs

def process_all_css_unit_rpkm_pair(file_dir, css_unit_dir, output_dir, unit=200):
    """
    Processes matching files from file_dir and css_unit_dir.

    Args:
        file_dir (str): Directory containing EXXX_processed.tsv files.
        css_unit_dir (str): Directory containing EXXX_unitcss_woChrM.pkl files.
        output_dir (str): Directory to save the processed DataFrames.
        unit (int): Division factor for TxStart and TxEnd.

    Returns:
        None
    """
    # List all files in the directories
    file_paths = [f for f in os.listdir(file_dir) if f.endswith("_processed.tsv")]
    css_unit_paths = [f for f in os.listdir(css_unit_dir) if f.endswith("_unitcss_woChrM.pkl")]

    # Match files by the first four characters (e.g., E003)
    matched_files = [
        (os.path.join(file_dir, f), os.path.join(css_unit_dir, u))
        for f in file_paths for u in css_unit_paths
        if f[:4] == u[:4]
    ]

    # Process each matched file pair
    for file_path, css_unit_path in matched_files:
        print(f"Processing: {file_path} with {css_unit_path}")
        # Load and process the data
        result_df = create_css_with_rpkm_df(file_path, css_unit_path, unit)

        # Save the resulting DataFrame
        output_file = os.path.join(output_dir, f"{os.path.basename(file_path)[:-4]}_css_unit_and_rpkm.csv")
        result_df.to_csv(output_file, index=False)
        print(f"Saved result to {output_file}")


# In[62]:


# file_dir="../database_exp/RNA/hg19/gene_expression/sorted_output/"
# css_unit_dir= "../chromatin_state/database/roadmap/css_unit_pickled/"
# output_dir="../database_exp/RNA/hg19/gene_expression/css_unit_and_rpkm/"
# process_all_css_unit_rpkm_pair(file_dir, css_unit_dir, output_dir, unit=200)


# In[63]:


# 1-3. Extract the sequence and rpkm value as a dataframe (for a single cell)
# [Note] This is for the Promoter regions!

def create_promoter_css_with_rpkm_df(file_path, css_unit_path, unit=200, upstream=2000, downstream=4000, remove_o=True):
    """
    Processes chromatin state data and extracts promoter chromatin state sequences per chromosome.
    The promoter region is defined as `upstream` bases upstream and `downstream` bases downstream from TSS.

    Args:
        file_path (str): Path to the input EXXX_processed.tsv file.
        css_unit_path (str): Path to the css_unit pickle file.
        unit (int): Division factor for TxStart and TxEnd.
        upstream (int): Number of bases upstream of TSS for the promoter region.
        downstream (int): Number of bases downstream of TSS for the promoter region.
        remove_o (bool): Whether to remove sequences dominated entirely by the "O" chromatin state
                         (low signal/quiescent regions). Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame with "sequence" and "RPKM" columns for promoter regions.
    """
    # Step 1: Read the input file
    if file_path.endswith('.pkl'):
        with open(file_path, 'rb') as f:
            df = pickle.load(f)
    else:
        col_name = ["gene_id", "chromosome", "strand", "TxStart", "TxEnd", "RPKM"]
        df = pd.read_csv(file_path, names=col_name, sep="\t")

    # Step 2: Filter out unwanted rows
    df_filtered = df[df['gene_id'] != 'name'].copy()
    df_filtered = df_filtered[df_filtered['chromosome'] != 'chrM'].copy()

    # Step 3: Convert TxStart and TxEnd from strings to integers
    df_filtered['TxStart'] = pd.to_numeric(df_filtered['TxStart'], errors='coerce')
    df_filtered['TxEnd'] = pd.to_numeric(df_filtered['TxEnd'], errors='coerce')

    # Divide by 'unit', round, and convert to integers
    df_filtered['TxStart'] = (df_filtered['TxStart'] / unit).round().astype(int)
    df_filtered['TxEnd'] = (df_filtered['TxEnd'] / unit).round().astype(int)

    # Step 4: Load css_unit
    with open(css_unit_path, "rb") as gg:
        css_unit = pickle.load(gg)

    # Step 5: Extract promoter chromatin state sequences and keep RPKM
    chromatin_sequences = []
    rpkm_values = []
    chromosome_map = {f"chr{i+1}" if i < 22 else ("chrX" if i == 22 else "chrY"): i for i in range(len(css_unit))}

    for _, row in df_filtered.iterrows():
        chromosome_index = chromosome_map.get(row['chromosome'])
        if chromosome_index is not None:
            # Adjust start and end based on promoter region
            if row['strand'] == '+':
                start = row['TxStart'] - (upstream // unit) - 1  # 2k upstream
                end = row['TxStart'] + (downstream // unit)  # 4k downstream
            elif row['strand'] == '-':
                start = row['TxEnd'] - (downstream // unit) - 1  # 4k downstream
                end = row['TxEnd'] + (upstream // unit)  # 2k upstream
            else:
                continue

            # Ensure valid indices
            start = max(0, int(start))
            end = min(len(css_unit[chromosome_index]), int(end))

            # Extract sequence
            sequence = css_unit[chromosome_index][start:end]
            chromatin_sequences.append(sequence)
            rpkm_values.append(row['RPKM'])

    # Step 6: Create the DataFrame
    result_df = pd.DataFrame({
        "sequence": chromatin_sequences,
        "RPKM": rpkm_values
    })

    # Step 7: Remove low-signal entries (e.g., sequences with only "O")
    if remove_o:
        result_df = result_df[result_df["sequence"].apply(lambda seq: not all(char == "O" for char in seq))]

    return result_df



# In[64]:


def process_all_promoter_css_unit_rpkm_pairs(file_dir, css_unit_dir, output_dir, unit=200, upstream=2000, downstream=4000, remove_o=True):
    """
    Processes the first matched file pair from file_dir and css_unit_dir for promoter regions.

    Args:
        file_dir (str): Directory containing EXXX_processed.tsv files.
        css_unit_dir (str): Directory containing EXXX_unitcss_woChrM.pkl files.
        output_dir (str): Directory to save the processed DataFrames.
        unit (int): Division factor for coordinates.
        upstream (int): Number of bases upstream of TSS for promoter regions.
        downstream (int): Number of bases downstream of TSS for promoter regions.
        remove_o (bool): Whether to remove sequences dominated entirely by the "O" chromatin state
                         (low signal/quiescent regions). Defaults to True.

    Returns:
        None
    """
    import os

    # List all files in the directories
    file_paths = [f for f in os.listdir(file_dir) if f.endswith("_processed.tsv")]
    css_unit_paths = [f for f in os.listdir(css_unit_dir) if f.endswith("_unitcss_woChrM.pkl")]

    # Match files by the first four characters (e.g., E003)
    matched_files = [
        (os.path.join(file_dir, f), os.path.join(css_unit_dir, u))
        for f in file_paths for u in css_unit_paths
        if f[:4] == u[:4]
    ]

    # Process only the first matched file pair
    for file_path, css_unit_path in matched_files:
        print(f"Processing: {file_path} with {css_unit_path}")

        # Load and process the data using promoter-specific function
        result_df = create_promoter_css_with_rpkm_df(file_path, css_unit_path, unit, upstream, downstream, remove_o=remove_o)

        # Save the resulting DataFrame
        output_file = os.path.join(output_dir, f"{os.path.basename(file_path)[:-4]}_promoter_css_unit_and_rpkm.csv")
        result_df.to_csv(output_file, index=False)
        print(f"Saved result to {output_file}")

        # # Break after processing the first file (for test)
        # break


# In[65]:


# file_dir="../database_exp/RNA/hg19/gene_expression/sorted_output/"
# # css_unit_dir= "../chromatin_state/database/roadmap/css_unit_pickled/"

# # # ####### with entire "O" state in promoter 
# # # output_dir="../database_exp/RNA/hg19/gene_expression/promoter_region/up2kdown4k/with_entire_O_state_prom_css_unit_and_rpkm/"
# # # process_all_promoter_css_unit_rpkm_pairs(file_dir, css_unit_dir, output_dir, unit=200, upstream=2000, downstream=4000,remove_o=False)

# # ####### without entire "O" state in promoter 
# # output_dir="../database_exp/RNA/hg19/gene_expression/promoter_region/up2kdown4k/without_entire_O_state_prom_css_unit_and_rpkm/"
# # process_all_promoter_css_unit_rpkm_pairs(file_dir, css_unit_dir, output_dir, unit=200, upstream=2000, downstream=4000,remove_o=True)


# In[66]:


import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_and_split_dataset(file_path, k=4, test_size=0.2, random_state=42):
    """
    Loads a dataset, k-merizes the sequence column using an existing seq2kmer function, 
    shuffles it, and splits it into training and validation sets.

    Parameters:
    - file_path (str): Path to the CSV file.
    - k (int): K-mer size (default=4).
    - test_size (float): Proportion of data to be used for validation (default=0.2, meaning 80/20 split).
    - random_state (int): Random seed for reproducibility (default=42).

    Returns:
    - train_data (pd.DataFrame): Shuffled training dataset with k-merized sequences.
    - dev_data (pd.DataFrame): Shuffled validation dataset with k-merized sequences.
    """
    # Load the dataset
    data = pd.read_csv(file_path)

    # Apply k-merization using the provided seq2kmer function
    data["sequence"] = data["sequence"].apply(lambda seq: seq2kmer(seq, k))

    # Shuffle the dataset for randomness
    data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Split into training and validation sets
    train_data, dev_data = train_test_split(data, test_size=test_size, random_state=random_state)

    return train_data, dev_data


# In[67]:


# # Example usage
# file_without_entire_O = "../database_exp/RNA/hg19/gene_expression/promoter_region/up2kdown4k/without_entire_O_state_prom_css_unit_and_rpkm_concatenated.csv"
# train_data, dev_data = preprocess_and_split_dataset(file_without_entire_O, k=4)

# # Save the output (optional)
# train_data.to_csv("../database_exp/RNA/hg19/gene_expression/promoter_region/up2kdown4k/without_entire_O_state_prom_training_data/4mer/train.tsv", sep="\t", index=False)
# dev_data.to_csv("../database_exp/RNA/hg19/gene_expression/promoter_region/up2kdown4k/without_entire_O_state_prom_training_data/4mer/dev.tsv", sep="\t", index=False)


# print("Dataset successfully k-merized and split into train.tsv and dev.tsv!")


# In[ ]:





# In[ ]:





# In[68]:


def prom_css_Kmer_by_cell(path="./", output_path="./",k=4):
    output_dir=str(k)+"mer/"
    output_path_fin=os.path.join(output_path, output_dir)
    os.makedirs(output_path_fin, exist_ok=True)

    all_files=sorted([os.path.join(path, file) for file in os.listdir(path)]) 

    for file in all_files:
        prom_kmer_all=[]
        cell_id=file.split("/")[-1][:4]

        # if cell_id=="E003": break # for test use

        with open(file, "rb") as f:
            prom=pickle.load(f)
        prom_css=flatLst(prom)  # make a list from list of a list
        prom_kmer=[seq2kmer(item,k) for item in prom_css]
        prom_kmer_all.append(prom_kmer)
        prom_kmer_all_flt=flatLst(prom_kmer_all)
        prom_kmer_all_flt_not_zero=[item for item in prom_kmer_all_flt if item!=""]
        output_name=cell_id+"_all_genes_prom_"+str(k)+"merized.txt"
        with open(output_path_fin+output_name, "w") as g:
            g.write("\n".join(prom_kmer_all_flt_not_zero))
    return 


# In[69]:


# test for prom_css_Kmer_by_cell
# path="../database/roadmap/prom/up2kdown4k/all_genes/"
# output_path="../database/final_test/"
# prom_css_Kmer_by_cell(path=path, output_path=output_path, k=4)
# test passed


# #### Pipeline 
# 
# (1) `prom_expGene2css` : cut the prom regions of long css <br>
# (2) `extProm_wrt_g_exp` : transform css into unit length css <br>
# (3) `extNsaveProm_g_exp` : load the required file and process all, and save

# #### Function: `Gexp_Gene2GLChr`
# 
# * This function only checks a single file.
# * Usage: After the gene expression files such as `gene_highlyexpressed.refFlat` are acquired by `/database/bed/gene_expression/classifygenes_ROADMAP_RPKM.py`, apply this function to obtain the list of dataframe per chromosome contains the transcription start and end indices.
# * Input: gene expression (high/low/not) file
# * Output: a chromosome-wise list of dataframe containing `TxStart` and `TxEnd`

# In[70]:


# function for preprocess the whole gene data and produce chromosome-wise gene lists
# each element is dataframe

### this function is not essential, but just to check by create df from .refFlat
def Gexp_Gene2GLChr(exp_gene_file='../database/bed/gene_expression/E050/gene_highlyexpressed.refFlat'):
    print("Extracting the gene file ...")
    g_fn=exp_gene_file
    g_df_raw=pd.read_csv(g_fn, sep='\t', index_col=False, header=0)
    g_df=g_df_raw
    g_df=g_df.iloc[:,1:]
    g_df.rename(columns={"name":"gene_id"}, inplace=True)
    g_df.rename(columns={"#geneName":"geneName"}, inplace=True)
    g_df.rename(columns={"txStart":"TxStart"}, inplace=True) # to make it coherent to my previous codes
    g_df.rename(columns={"txEnd":"TxEnd"}, inplace=True)
#     g_df=g_df_raw.rename(columns={0:"geneName",1:"gene_id",2:"chrom",3:"strand",4:"txStart",5:"txEnd",
#                                       6:"cdsStart",7:"cdsEnd",8:"exonCount",9:"exonStart",10:"exonEnds",
#                                       11:"gene type",12:"transcript type",13:"reference transcript name",
#                                       14:"reference transcription id"})
    ## string to the list of "int", for exon start/end ##
    g_df_temp=g_df # copy for processing
    exon_start_int_lst=[]
    for i, str_lst in enumerate(g_df_temp["exonStarts"]):
        int_lst=[int(elm) for elm in str_lst.replace("[","").replace("]","").split(",")]
        assert g_df_temp["exonCount"][i]==len(int_lst) # make sure the no. element in exon st count
        exon_start_int_lst.append(int_lst)    
    g_df_temp["exonStarts"]=exon_start_int_lst

    exon_end_int_lst=[]
    for i, str_lst in enumerate(g_df_temp["exonEnds"]):
        int_lst=[int(elm) for elm in str_lst.replace("[","").replace("]","").split(",")]
        assert g_df_temp["exonCount"][i]==len(int_lst) # make sure the no. element in exon start = count
        exon_end_int_lst.append(int_lst)    
    g_df_temp["exonEnds"]=exon_end_int_lst    
    g_df=g_df_temp # and make it back the original name

    g_df=g_df[["geneName","gene_id","chrom","TxStart","TxEnd"]] # extract these only

    # Remove other than regular chromosomes
    chr_lst=['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10',
             'chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19',
             'chr20','chr21','chr22','chrX','chrY']
    g_df=g_df.loc[g_df["chrom"].isin(chr_lst)]

    # Create a list of chromosome-wise dataframe 
    g_df_chr_lst=[]
    for num in range(len(chr_lst)):
        chr_num=chr_lst[num]
        g_chr_df='g_'+chr_num  # name it as "g_"
        locals()[g_chr_df]=g_df[g_df["chrom"]==chr_num]
        g_chr_df=locals()[g_chr_df]
        g_chr_df=g_chr_df.sort_values("TxStart")
        g_df_chr_lst.append(g_chr_df)

    # Remove the overlapped area (using removeOverlapDF function in css_utility.py)
    g_df_chr_collapsed_lst=[]
    for g_df_chr in g_df_chr_lst:
        g_df_chr_collapsed=removeOverlapDF(g_df_chr)
        assert len(g_df_chr)>=len(g_df_chr_collapsed)
        g_df_chr_collapsed_lst.append(g_df_chr_collapsed)
    print("Done!")

    return g_df_chr_collapsed_lst  # list of dataframe


# #### Function `prom_expGene2css`
# * This function produces a long list (not unit length) of css according to the gene expression table, per cell.

# In[71]:


def prom_expGene2css(g_lst_chr_merged,df, up_num=2000, down_num=4000):   # df indicates css, created by bed2df_expanded
    """
    modified from `compGene2css`
    Input: Reference gene file trimmed for gene expresseion level, df (CSS)
    Output: list of chromosome-wise list that contains the css at (expressed) genic area with prom only.
    """
    g_lst_chr=g_lst_chr_merged
    df = df[df['chromosome'] != 'chrM']
    css_lst_chr=df2longcss(df) # list of long css per chromosome

    g_lst_chr = g_lst_chr[:len(css_lst_chr)]  # adjust the length of list according to length of df (might not have chrY)
    total_chr=len(css_lst_chr)

    print("Matching to the chromatin state sequence data ...")
    css_prom_lst_all=[]
    # for i in tqdm_notebook(range(total_chr)):
    for i in range(total_chr):
        css=css_lst_chr[i]   # long css of i-th chromosome
        gene_df=g_lst_chr[i] # gene df of i-th chromosome

        css_prom_lst_chr=[]
        for j in range(len(gene_df)):
            prom_start=gene_df["TxStart"].iloc[j]-1-up_num  # python counts form 0
            prom_end=prom_start+up_num+down_num+1      # python excludes the end
            if gene_df["TxEnd"].iloc[j]<prom_end:  # if longer than gene body, then just gene body
                prom_end=gene_df["TxEnd"].iloc[j]+1

            css_prom=css[prom_start:prom_end]           # cut the gene area only
            css_prom_lst_chr.append(css_prom)     # store in the list

        css_prom_lst_all.append(css_prom_lst_chr)  # list of list

    assert len(css_prom_lst_all)==total_chr

    # remove chromosome if it is empty (e.g. chrY for female)
    css_prom_lst_all=[elm for elm in css_prom_lst_all if elm!=[]] 

    print("Done!")
    return css_prom_lst_all 


# In[72]:


def extProm_wrt_g_exp(exp_gene_file, df, up_num=2000, down_num=4000,unit=200):
    """
    extract promoter regions of genes according to gene expression level
    """
    df = df[df['chromosome'] != 'chrM']
    g_lst_chr=Gexp_Gene2GLChr(exp_gene_file)
    g_lst_chr_merged=merge_intervals(g_lst_chr)

    css_prom_lst_all=prom_expGene2css(g_lst_chr_merged,df, up_num=up_num, down_num=down_num)
    css_prom_lst_unit_all=Convert2unitCSS_main_new(css_prom_lst_all, unit=unit)
    return css_prom_lst_unit_all


# #### Function: `removeOverlapDF` and `gene_removeDupl`
# 
# * Main function: `gene_removeDupl`
# * `removeOverlapDF`: function used inside the main function.
# * To acquire final collapsed gene table, run `gene_removeDupl`

# In[73]:


def removeOverlapDF(test_df):    
    new_lst=[]
    for i in range(len(test_df)):
        start=test_df["TxStart"].iloc[i]
        end=test_df["TxEnd"].iloc[i]

        exist_pair=(start,end)

        if i==0:
            new_pair=exist_pair
            new_lst.append(new_pair)        
        else:
            start_pre=test_df["TxStart"].iloc[i-1]
            end_pre=test_df["TxEnd"].iloc[i-1]

            # first, concatenate all the shared start
            if start==start_pre:
                new_end=max(end, end_pre)
                new_pair=(start, new_end)
            # second, concatenate all the shared end
            elif end==end_pre:
                new_start=min(start, start_pre)
                new_pair=(new_start, end)
            else:    
                new_pair=exist_pair

        new_lst.append(new_pair) 
    new_lst=list(dict.fromkeys(new_lst))

    mod_lst=[[start, end] for (start, end) in new_lst] # as a list element

    for j, elm in enumerate(mod_lst):
        start, end = elm[0], elm[1]

        if j==0:
            continue
        else:
            start_pre=mod_lst[j-1][0]
            end_pre=mod_lst[j-1][1]

            if end_pre>=end:
                mod_lst[j][0]=mod_lst[j-1][0]  # if end_pre is larger than end, replace start as start_pre
                mod_lst[j][1]=mod_lst[j-1][1]  # if end_pre is larger than end, replace end as end_pre

            elif start <=end_pre:
                mod_lst[j][0]=mod_lst[j-1][0]  # current start=start_pre
                mod_lst[j-1][1]=max(mod_lst[j][1],mod_lst[j-1][1])  # end_pre = end

            else:
                continue

    mod_lst=[tuple(elm) for elm in mod_lst]
    fin_lst=list(dict.fromkeys(mod_lst))
    gene_collapsed_df=pd.DataFrame(fin_lst, columns=["TxStart", "TxEnd"])

    return gene_collapsed_df


# In[74]:


def gene_removeDupl(whole_gene_file='../database/RefSeq/RefSeq.WholeGene.bed'):
    g_df_chr_lst=whGene2GLChr(whole_gene_file)
    new_gene_lst_all=[]
    for chr_no in range(len(g_df_chr_lst)):
        gene_df=g_df_chr_lst[chr_no]
        gene_collapsed_df=removeOverlapDF(gene_df)
        new_gene_lst_all.append(gene_collapsed_df)
    return new_gene_lst_all # list of chromosome-wise dataframe for collapsed gene table


# #### Function `extNsaveProm_g_exp`
# * Fine-tuning data preparation
# * This function processes the above works (cut the prom region and make it unit length css) per cell
# * Input
#     * `exp_gene_dir`: directory where refFlat for each cell (subdir means the sub directory for different gene expression level)
#     * `df_pickle_dir`: dataframe of each cell
#     * `rpkm_val`: RPKM value, 10, 20, 30, or 50
#     * `up_num`: upstream of gene
#     * `down_num`: from TSS (gene initial part) to cut
#     * `unit`: because chromatin states are annotated by 200 bps
# * Output: save the file according to the `rpkm_val` at the output path

# In[75]:


def extNsaveProm_g_exp(exp_gene_dir="./", df_pickle_dir="./",output_path="./",file_name="up2kdown4k",rpkm_val=50, up_num=2000, down_num=4000,unit=200):
    exp_gene_subdir=os.listdir(exp_gene_dir)
    exp_gene_tardir=[os.path.join(exp_gene_dir, subdir) for subdir in exp_gene_subdir if str(rpkm_val) in subdir][0]    

    if rpkm_val==0:
        exp_gene_tardir=os.path.join(exp_gene_dir, "rpkm0")

    exp_gene_files=sorted([os.path.join(exp_gene_tardir,file) for file in os.listdir(exp_gene_tardir)])

    for exp_gene_file in exp_gene_files:
        cell_id=exp_gene_file.split("/")[-1][:4]

        # print(cell_id)   ## for test
        # if cell_id=="E004":break ## for test

        df_name=[file for file in os.listdir(df_pickle_dir) if cell_id in file][0]
        df_path=os.path.join(df_pickle_dir,df_name)
        with open(df_path,"rb") as f:
            df=pickle.load(f)
        css_prom_lst_unit_all=extProm_wrt_g_exp(exp_gene_file, df, up_num=up_num, down_num=down_num,unit=unit)

        output_name=output_path+"rpkm"+str(rpkm_val)+"/"+cell_id+"_prom_"+file_name+".pkl"
        output_dir = os.path.dirname(output_name)

        # print(output_name) ### test

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        with open(output_name, "wb") as g:
            pickle.dump(css_prom_lst_unit_all,g)
    return print("Saved at ",output_path)


# In[76]:


# test for extNsaveProm_g_exp
# extNsaveProm_g_exp(exp_gene_dir="../database/roadmap/gene_exp/refFlat_byCellType/", df_pickle_dir="../database/roadmap/df_pickled/",output_path="../database/final_test/",file_name="up2kdown4k",rpkm_val=50, up_num=2000, down_num=4000,unit=200)
# test passed


# In[ ]:


###### colab fine-tuning



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Extract Promoter regions from not expressed genes

# #### Pipeline 
# 
# (1) `extWholeGeneRef` : Just extract the whole gene location files from `chr.gene.refFlat` <br>
# (2) `extNOTexp_by_compare` : Extract the not expressed genes by comparing with whole gene with rpkm>0 <br>
# (3) `extNsaveNOTexp_by_compare` : load the required file and process all, and save refFlat (.pkl) and prom-region css (.pkl)

# In[77]:


def extWholeGeneRef(whole_gene_ref):
    ###### modified from Gexp_Gene2GLChr, this function provides the df of whole genes
    ###### note that this file contains Y chromosome
    g_fn=whole_gene_ref
    g_df=pd.read_csv(g_fn, sep='\t', index_col=False, header=0)
    g_df=g_df.iloc[:,1:]
    g_df.rename(columns={"name":"gene_id"}, inplace=True)
    g_df.rename(columns={"#geneName":"geneName"}, inplace=True)
    g_df.rename(columns={"txStart":"TxStart"}, inplace=True) # to make it coherent to my previous codes
    g_df.rename(columns={"txEnd":"TxEnd"}, inplace=True)     
    g_df=g_df[["chrom","TxStart","TxEnd"]] # extract these only
    # Remove other than regular chromosomes
    chr_lst=['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10',
             'chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19',
             'chr20','chr21','chr22','chrX','chrY']
    g_df=g_df.loc[g_df["chrom"].isin(chr_lst)]

    # Create a list of chromosome-wise dataframe 
    g_df_chr_lst=[]
    for num in range(len(chr_lst)):
        chr_num=chr_lst[num]
        g_chr_df='g_'+chr_num  # name it as "g_"
        locals()[g_chr_df]=g_df[g_df["chrom"]==chr_num]
#         print(chr_num)
        g_chr_df=locals()[g_chr_df]
        g_chr_df=g_chr_df.sort_values("TxStart")
        g_df_chr_lst.append(g_chr_df)

    # remove any overlap
    g_df_chr_lst=merge_intervals(g_df_chr_lst)
    return g_df_chr_lst  # list of chromosome-wise df for all gene start and end


# In[78]:


def extNOTexp_by_compare(whole_gene_ref, cell_exp_ref):
    """
    whole_gene_ref: e.g.) chr.gene.refFlat"
    """
    whole_gene_ref_lst=extWholeGeneRef(whole_gene_ref)
    cell_exp_lst=Gexp_Gene2GLChr(cell_exp_ref)
    cell_exp_lst=merge_intervals(cell_exp_lst) 
    if len(whole_gene_ref_lst)!=len(cell_exp_lst):
        whole_gene_ref_lst=whole_gene_ref_lst[:-1]   
    non_exp_gene_lst=[]
    for i, whole_gene_chr in enumerate(whole_gene_ref_lst):
        exp_gene_mark = whole_gene_chr.merge(cell_exp_lst[i], on=['TxStart', 'TxEnd'])
        non_exp_gene_chr=whole_gene_chr.drop(exp_gene_mark.index)
        non_exp_gene_lst.append(non_exp_gene_chr)
    print("total length of non_expressed genes in this cell: ",len(pd.concat(non_exp_gene_lst)))
    return non_exp_gene_lst


# In[79]:


def extNsaveNOTexp_by_compare(whole_gene_ref_path,
                              exp_ref_path="./",
                              df_pickle_dir="./",
                              output_path_ref="./",
                              output_path_prom="./",
                              up_num=2000,down_num=4000,unit=200):
    """
    whole_gene_ref: e.g.) chr.gene.refFlat"
    """
    exp_ref_file_all=sorted([os.path.join(exp_ref_path,file) for file in os.listdir(exp_ref_path)])

    for exp_ref_file in exp_ref_file_all:
        cell_id=exp_ref_file.split("/")[-1][:4]
#         if cell_id=="E004":break # for test
        print(cell_id+" is now processing...")

        df_name=[file for file in os.listdir(df_pickle_dir) if cell_id in file][0]
        df_path=os.path.join(df_pickle_dir,df_name)
        with open(df_path,"rb") as f:
            df=pickle.load(f)

        non_exp_gene_lst=extNOTexp_by_compare(whole_gene_ref_path, exp_ref_file) # a list of chromosome-wise df
        #### refFlat for NOT expressed is pickled as a list of dataframe ####
        not_exp_ref_path=output_path_ref+cell_id+"_gene_not_expressed.pkl"
        with open(not_exp_ref_path,"wb") as g:
            pickle.dump(non_exp_gene_lst,g)        

        css_prom_lst_all=prom_expGene2css(non_exp_gene_lst, df, up_num=up_num, down_num=down_num)
        css_prom_lst_unit_all=Convert2unitCSS_main_new(css_prom_lst_all, unit=unit)

        output_name=output_path_prom+cell_id+"_not_exp_gene_prom_up2kdown4k.pkl"
        with open(output_name,"wb") as h:
            pickle.dump(css_prom_lst_unit_all,h)

    return print("refFlat is saved at {} and prom is saved at {}.".format(output_path_ref, output_path_prom))


# In[80]:


# # # test for extNsaveNOTexp_by_compare
# extNsaveNOTexp_by_compare(whole_gene_ref_path="../database/roadmap/gene_exp/chr.gene.refFlat",
#                               exp_ref_path="../database/roadmap/gene_exp/refFlat_byCellType/rpkm0/",
#                               df_pickle_dir="../database/roadmap/df_pickled/",
#                               output_path_ref="../database/roadmap/gene_exp/refFlat_byCellType/not_exp/",
#                               output_path_prom="../database/final_test/",
#                               up_num=2000,down_num=4000,unit=200)
# # # test passed


# #### Function `prom_css_Kmer_by_cell`
# * This function saves the kmerized promoter regions (of all genes)

# In[81]:


def prom_css_Kmer_by_cell(path="./", output_path="./",k=4):
    output_dir=str(k)+"mer/"
    output_path_fin=os.path.join(output_path, output_dir)

    os.makedirs(output_path_fin, exist_ok=True)

    all_files=sorted([os.path.join(path, file) for file in os.listdir(path)]) 

    for file in all_files:
        prom_kmer_all=[]
        cell_id=file.split("/")[-1][:4]
        # if cell_id=="E004": break # for test use
        with open(file, "rb") as f:
            prom=pickle.load(f)
        prom_css=flatLst(prom)  # make a list from list of a list
        prom_kmer=[seq2kmer(item,k) for item in prom_css]
        prom_kmer_all.append(prom_kmer)
        prom_kmer_all_flt=flatLst(prom_kmer_all)
        prom_kmer_all_flt_not_zero=[item for item in prom_kmer_all_flt if item!=""]
        output_name=cell_id+"_all_genes_prom_"+str(k)+"merized.txt"
        with open(output_path_fin+output_name, "w") as g:
            g.write("\n".join(prom_kmer_all_flt_not_zero))
    return 


# In[82]:


# test for prom_css_Kmer_by_cell
# prom_css_Kmer_by_cell(path="../database/roadmap/prom/up2kdown4k/all_genes/", output_path="../database/final_test/",k=4)
# test passed


# In[ ]:





# In[ ]:





# ### Prepare fine tuning data using chromatin state sequence lists
#  Note that these functions are designed for universal applicability and can be used across a wide range of general cases, not just for specific data types.

# #### Pipeline 
# 
# (1) `css_CUT_Kmer` : Cut and K-merize the chromatin state sequence list  <br>
# (2) `prep_by_merge_cell` : Merge multiple chromatin state sequence lists into a single list <br>
# (3) `kmerize_and_cut` : Process all functions above <br><br>
# 
# (4) `process_save_TF` : Prepare binary classification data, after finishing above processes

# In[83]:


# Cut if it is longer than 510
def css_CUT_Kmer(css, cut_thres=510, k=4):
    """ 
    Prepare kmer dataset for unit_css, as is if length<=510, else cut it to be length>510   
    Usage: css_CUT_Kmer(css, cut_thres, k)

    - css: unit-length css (e.g. comp_gene_css_all)
    - cut_thres: length of split, default=510
    - k: kmer

    Output: 1. splitted (before kmerization) 2. kmerized_unit_css (after kmerization) 
    """    
    splitted=[] # bucket for the all the splitted strings   
    for css_elm in css:
        if len(css_elm) <k:  # if the length of css_elm is shorter than k (cannot create k-mer)
            continue
        elif len(css_elm) <=cut_thres:
            splitted.append(css_elm)
        else:  
            prev=0
            while True:
                splitted.append(css_elm[prev:prev+cut_thres])
                prev+=cut_thres
                if prev>=len(css_elm)-1:
                    break      

    kmerized_unit_css_raw=[seq2kmer(item, k) for item in splitted] # k-merize here

    ### this part is updated to prevent any empty string to be generated ###
    kmerized_unit_css=[item for item in kmerized_unit_css_raw if item!=""]
    ########################################################################

    return splitted, kmerized_unit_css


# In[84]:


def prep_by_merge_cell(input_path):  # css is the list of chromatin state sequence list in the region of interest
    """
    This function merges multiple .pkl data (unit-length of Chromatin state sequence, per cell) into one

    Output
    - A list of merged chromatin state sequences in the "input_path"
    """
    file_list=os.listdir(input_path)
    file_path_list=[os.path.join(input_path,file) for file in file_list]
    # print(file_path_list)

    def contains_sublists(input_list):
        # Check if any element in the list is a sublist (list type)
        return any(isinstance(element, list) for element in input_list)

    css_concat=[]
    for file_path in file_path_list:
        with open(file_path,"rb") as f:
            css=pickle.load(f)
        # if it contains a sublist, then flatten it (normally it does)
        # print(contains_sublists(css))
        if contains_sublists(css):

            css=flatLst(css)
        # print(type(css))
        css_concat.extend(css)  
    return css_concat


# In[85]:


def kmerize_and_cut(input_path,output_path, output_file_name, k=4, cut_thres=510):
    """
    This function merges all the file under input_path, and cut if an entry is longer than cut_thres,
    and k-merize before save it at the output_path

    Parameters:
    - input_path: path to the chromatin state data in .pkl format
    - output_path: path to the output
    - output_file_name: desired file name.  e.g.) "output.txt"
    - k: desired number for k-merization, default=4
    - cut_thres: desired number for threshold for cutting long entries, default=510
    """
    css_mergerd=prep_by_merge_cell(input_path)
    _, kmerized_unit_css=css_CUT_Kmer(css_mergerd, cut_thres=cut_thres, k=k)
    # print(type(kmerized_unit_css))
    # print(len(kmerized_unit_css))
    # Open a text file for writing
    output_file_path=os.path.join(output_path,output_file_name)
    with open(output_file_path, 'w') as file:
        # Write each item on a new line
        for item in kmerized_unit_css:
            file.write(f"{item}\n")
    return print("The files in {} were merged, {}-merized, and saved at{}".format(input_path,k,output_path))


# In[86]:


# # test 
# kmerize_and_cut(input_path="../database/test_box/input_test",output_path="../database/test_box/output_test/",
#                 output_file_name="test.txt", k=4, cut_thres=510)
# # test passed


# In[87]:


def process_save_TF(cl1_path, cl2_path, output_path, k=4, wo_cont_o_state=True, len_tr=20000, len_dev=1000):
    """
    When you have k-merized .txt file for each class, process and save it for fine-tuning data 

    Parameters:
    - cl1_path (str): Path to class 1 (the class of interest) 
    - cl2_path (str): Path to class 2 (the class of comparison)
    - output_path (str): Path to save the result file (train.tsv, dev.tsv)
    - k (int): The length of k-mer, default=4
    - wo_cnt_o_state (bool): whether to remove continuous O state (True to remove)
    - len_tr: The length of train.tsv file. Default=20000
    - len_dev: The length of dev.tsv file. Default=1000

    Output:
    - Fine-tuning data file named "train.tsv" and "dev.tsv" are saved at output_path

    """
    cl1=pd.read_csv(cl1_path, header=None, names=["sequence"])
    cl1_list=cl1["sequence"].tolist()
    cl2=pd.read_csv(cl2_path, header=None, names=["sequence"])
    cl2_list=cl2["sequence"].tolist()

    if wo_cont_o_state:
        cl1_list = [item for item in cl1_list if k*"O" not in item]
        cl2_list = [item for item in cl2_list if k*"O" not in item]

    print("class 1 has {} elements.".format(len(cl1_list)))
    print("class 2 has {} elements.".format(len(cl2_list)))

    # make it dataframe
    df_cl1=pd.DataFrame(cl1_list, columns=["sequence"])
    df_cl1["label"]=1
    df_cl2=pd.DataFrame(cl2_list, columns=["sequence"])
    df_cl2["label"]=0

    # make them have the same length
    if len(df_cl1)>len(df_cl2):
        df_cl1=df_cl1[:len(df_cl2)] 
    elif len(df_cl1)<len(df_cl2):
        df_cl2=df_cl2[:len(df_cl1)]
    assert len(df_cl1)==len(df_cl2), "Check the data length."

    # shuffling 
    df_all=pd.concat([df_cl1,df_cl2]).sample(frac=1).reset_index(drop=True)  

    # cutting into train and dev
    assert len(df_all)> len_tr+len_dev, "Not enough data length."
    df_train=df_all[:len_tr]
    df_dev=df_all[len_tr:len_tr+len_dev]    

    train_name=os.path.join(output_path,"train.tsv")
    dev_name=os.path.join(output_path,"dev.tsv")

    df_train.to_csv(train_name, sep="\t", index=False)
    df_dev.to_csv(dev_name, sep="\t", index=False)

    return print("Fine-tuning data are saved at {}.".format(output_path))





# In[88]:


# # test
# cl1_path="../database/pretrain/crm/lim10/crm_lim10_allcell_4merized.txt"
# cl2_path="../database/pretrain/not_crm/lim10/not_crm_lim10_allcell_4merized.txt"
# output_path="../database/test_box/output_test/"
# process_save_TF(cl1_path=cl1_path, cl2_path=cl2_path, output_path=output_path, k=4, wo_cont_o_state=True, len_tr=20000, len_dev=1000)
# # test passed


# #### CRM Dataset preparation
# 
# The CRM regions are usually very short (for unit css length, everage is almost near 2). So the regions are screened to be longer than 6, 7, 8, 9, 10 (in terms of unit). Following functions extract CRM regions according to user-defined length and save it to the designated path. Final function saves the CRM regions with the designated length, k-mer. 
# 
# (1) `crm_df_maker` : prepare the dataframe of CRM from the raw bed file <br>
# (2) `extCRMfromCell` : cut the CRM regions of unit css for a sample cell <br>
# (2) `extCRMfromCell_all` : cut the CRM regions of unit css for all cells <br>
# (3) `saveCRMforPREall` : save the CRM extracted for various limit length (from 6 to 10)
# 

# In[84]:


def crm_df_maker(crm_path="../database/remap2022/remap2022_crm_macs2_hg19_v1_0.bed", limit_len=3):
    # Load the data
    crm_raw = pd.read_csv(crm_path, sep='\t', header=None, names=["chromosome", "start", "end", "name", "score", "strand", "thickStart", "thickEnd", "itemRgb"])

    # Convert start and end locations to units of 200 bps and calculate length
    crm_raw['start'] = (crm_raw['start'] / 200).round().astype(int)
    crm_raw['end'] = (crm_raw['end'] / 200).round().astype(int)
    crm_raw['length'] = crm_raw['end'] - crm_raw['start'] + 1

    # Filter by chromosome and length
    crm_df = crm_raw[crm_raw["chromosome"].str.contains('^chr[0-9XY]+$') & (crm_raw['length'] >= limit_len)].copy()

    # Define chromosome order
    chromosome_order = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10',
                        'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19',
                        'chr20', 'chr21', 'chr22', 'chrX', 'chrY']

    # Convert 'chromosome' to a categorical type with the defined order
    crm_df['chromosome'] = pd.Categorical(crm_df['chromosome'], categories=chromosome_order, ordered=True)

    # Sort by 'chromosome' and 'start'
    crm_df = crm_df.sort_values(['chromosome', 'start'])
    crm_df_fin = crm_df[["chromosome","start" ,"end","length","name"]]

    # Print summary
    print("{} out of total {} CRM entries are longer than 200x{}, which is approx. {} %".format(len(crm_df), len(crm_raw), limit_len, round(len(crm_df)/len(crm_raw), 3)))

    return crm_df_fin


# In[85]:


### cut the css according to the CRM position
def extCRMfromCell(css_sample_path="../database/roadmap/css_unit_pickled/E003_unitcss_woChrM.pkl",crm_path="../database/remap2022/remap2022_crm_macs2_hg19_v1_0.bed",limit_len=4):
    #### load unit-css per chromosome of one cell
    with open(css_sample_path, "rb") as s:
        unit_css=pickle.load(s)
    #### make CRM as a dataframe
    crm_df_fin=crm_df_maker(crm_path=crm_path,limit_len=limit_len)

    cut_lst_all=[]
    for chr in range(len(unit_css)):
        unit_css_chr=unit_css[chr]
        chromosome_order = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10',
                            'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19',
                            'chr20', 'chr21', 'chr22', 'chrX', 'chrY']
        crm_df_chr=crm_df_fin[crm_df_fin["chromosome"]==chromosome_order[chr]]

        cut_lst=[]
        for i in range(len(crm_df_chr)):
            start_loc=crm_df_chr["start"].iloc[i]
            end_loc=crm_df_chr["end"].iloc[i]
            # length=crm_df_fin["length"].iloc[i]     
            cut=unit_css_chr[start_loc:end_loc+1]
            cut_lst.append(cut)
        cut_lst_all.append(cut_lst)
    return cut_lst_all


# In[86]:


def extCRMfromCell_all(input_path="../database/roadmap/css_unit_pickled/", crm_path="../database/remap2022/remap2022_crm_macs2_hg19_v1_0.bed", output_path="../database/remap2022/crm/", limit_len=6):
    files=os.listdir(input_path)
    css_paths=[os.path.join(input_path, file) for file in files if "E" in file]  # list of paths for css of cells
    for css_sample_path in css_paths:
        cut_lst_all=extCRMfromCell(css_sample_path=css_sample_path,crm_path=crm_path,limit_len=limit_len)

        file_name=re.search(r'E\d{3}_unitcss_', css_sample_path).group(0)
        output_file=os.path.join(output_path,file_name+"limit_len"+str(limit_len)+".pkl")
        with open(output_file,"wb") as f:
            pickle.dump(cut_lst_all,f)
    return print("All files are saved at {}, with limit_len={}".format(output_path, limit_len))


# In[87]:


def saveCRMforPREall_mod(input_path="../database/remap2022/crm/",output_path="../database/pretrain/crm/",limit_len=10, k=4): 
    files=os.listdir(os.path.join(input_path,"lim"+str(limit_len)))
    css_all=[]
    for file in files:
        file_name=os.path.basename(file)
        if file_name[0] == 'E' and file_name[1:4].isdigit():
            file_id = file_name[:4]
        else:
            pass
        # ##########################
        # if str(file_id)=="E003":
        #     break  # for test
        # ##########################
        with open(os.path.join(input_path,"lim"+str(limit_len),file), "rb") as f:
            css_lst=pickle.load(f)
        css=flatLst(css_lst)
        css_kmer=[]
        for css_chr in css:
            css_chr_kmer=seq2kmer(css_chr,k)
            target_to_remove="O"*k   # get rid of the word with continuous 15th state "o"
            css_chr_kmer_trim = css_chr_kmer.replace(target_to_remove, "")
            # clean up extra spaces
            css_chr_kmer_trim = ' '.join(css_chr_kmer_trim.split())
            css_kmer.append(css_chr_kmer_trim)
        css_all.append(css_kmer)
    css_all_flt=flatLst(css_all)
    os.makedirs(os.path.join(output_path, "lim" + str(limit_len)), exist_ok=True)
    output_name="crm_lim"+str(limit_len)+"_allcell_wo_cnt_o"+str(k)+"merized.txt"
    with open(os.path.join(output_path,"lim"+str(limit_len),output_name), "w") as g:
            g.write("\n".join(css_all_flt))
    return print("File is saved at {}".format(output_path))


# #### Motif logo visualization
# 1. Logo style visualization using attention score <br><br>
# Usage: `motif_logo(mat_path, dev_path, motif="GBBBG")`

# In[88]:


def dev_conv(dev_file_path):
    """
    convert dev.tsv file to dataframe with restored sequence (original sequence)
    """
    dev_df=pd.read_csv(dev_file_path,sep="\t")
    dev_df.fillna(" ", inplace=True) # change the nan into empty string
    assert dev_df["sequence"].isnull().sum()==0, "check the dev file for nan values"

    def kmer2seq_or_space(seq):
        if seq == " ":
            return " "
        else:
            return kmer2seq(seq)

    dev_df["ori_seq"] = dev_df["sequence"].apply(kmer2seq_or_space)

    return dev_df


# In[89]:


def get_matWcss(mat_path,dev_path):
    """
    Read atten.npy and dev.tsv to create two dictionaries:
        all_dict_1: attention matrix for label 1 per data strip
        all_dict_0: attention matrix for label 0 per data strip
    """
    atten_raw=np.load(mat_path)
    atten=pd.DataFrame(atten_raw)
    dev_raw=dev_conv(dev_path) #dev_conv is a function in css_utility
    dev=dev_raw[["ori_seq","label"]]
    dev.reset_index(drop=True, inplace=True)  # remove the header

    dev_label_1=dev[dev["label"]==1]
    dev_label_0=dev[dev["label"]==0]

    atten["label"] = dev["label"].values
    atten_label_1=atten[atten["label"]==1]
    dev_label_1.pop("label") # remove the label column
    atten_label_1.pop("label") # remove the label column
    atten_label_0=atten[atten["label"]==0]
    dev_label_0.pop("label") # remove the label column
    atten_label_0.pop("label") # remove the label column

    assert len(dev_label_1)==len(atten_label_1)
    assert len(dev_label_0)==len(atten_label_0)

    all_dict_1={i:(dev_label_1.loc[i], atten_label_1.loc[i]) for i in dev_label_1.index}
    all_dict_0={i:(dev_label_0.loc[i], atten_label_0.loc[i]) for i in dev_label_0.index}

    ### to use, apply following 
#     for index, (dev_entry, atten_entry) in list(atten_dict_1.items()):
#         dev_tar=dev_entry['ori_seq']
#         atten_tar=atten_entry.values

    return all_dict_1, all_dict_0


# In[90]:


####################### draw the designated entry only ######################
def get_motifWScore(all_dict_1, motif, extend_len=0):   # modify the code for showing extended version
    """ 
    For a class of interest (labeled either 1 or 0, but normally 1), find the motif of interest with the corresponding attention scores.
    Outputs:
        - motif_found_all: a list of the list of found motif in the class of interest
        - score_found_all: a list of the list of corresponding scores (not normalized)
        - score_found_norm_all: a list of the list of corresponding scores (normalized, per each data strip)

    """
    #### small function to correctly show the counting number
    def get_ordinal_suffix(number):
        if 10 <= number % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(number % 10, 'th')
        return suffix

    # all_dict_1 is the dictionary (key:index, value=tuple of motif and )
    motif_found_all=[]
    score_found_all=[]
    score_found_norm_all=[]

    cnt=0
    for index, (dev_entry, atten_entry) in list(all_dict_1.items()):
        dev_tar = dev_entry['ori_seq']
        atten = atten_entry.values.reshape(-1, 1).T  # Reshape to 2D array for heatmap
        atten_tar = atten[0] # atten is a list of list, with one element

#         if motif in dev_tar:
#             start_index=dev_tar.find(motif)
#             end_index=start_index+len(motif)
        if motif in dev_tar:
            start_index=dev_tar.find(motif)-extend_len
            end_index=start_index+len(motif)+(extend_len*2)
            if end_index <= len(atten_tar)+1:
                cnt+=1
                print(f"{motif}: {cnt}{get_ordinal_suffix(cnt)} appearance")
                print("- Index in attention matrix: ",index)
                print("- Score min-max: {} ~ {}".format(round(min(atten_tar),3), round(max(atten_tar),3)))
                motif_found=dev_tar[start_index:end_index]
                score_found=atten_tar[start_index:end_index].tolist()
                score_found_norm=[item/sum(atten_tar) for item in score_found]  # normalize

                motif_found_all.append(motif_found)
                score_found_all.append(score_found)
                score_found_norm_all.append(score_found_norm)

                ### create the motif figures 
                show_len=end_index-start_index
                fig_width=show_len*0.35

                ######## Data strip with colored scores (to show the motif on the colored strip)

                # plt.figure(figsize=(fig_width, 1))  # Adjusted height to give space for text
                # # Add colored text for each letter in dev_tar above the heatmap
                # for i, letter in enumerate(motif_found):                    
                #     plt.text(i + 0.5, -0.2, letter, color=state_col_dict.get(letter, 'black'),
                #              ha='center', va='center', fontsize=32, family='monospace')

                # sns.heatmap(data=[score_found], cmap="viridis", yticklabels=False, cbar=False)
                # plt.show()
    return motif_found_all, score_found_all, score_found_norm_all #list of list


# In[91]:


def score2logo(motif_found_all, score_found_all, score_found_norm_all, norm=False):
    """
    Use the result of get_motifWScore, create logos
    """
    import logomaker
    num_row=len(motif_found_all[0])
    columns=[chr(i) for i in range(ord("A"), ord("O")+1)]

    # df for creating logo
    df_logo_all=[]
    df=pd.DataFrame(0.0, index=range(num_row), columns=columns)
    for i, motif_found in enumerate(motif_found_all):
        if norm:
            for j, letter in enumerate(motif_found):
                df.loc[j, letter]=round(score_found_norm_all[i][j],3)
        elif not norm:
            for j, letter in enumerate(motif_found):
                df.loc[j, letter]=round(score_found_all[i][j],3)   
        df_logo=df.copy()
        df_logo_all.append(df_logo)

        fig_width=num_row*0.5
        logo=logomaker.Logo(df_logo,color_scheme=state_col_dict_num, figsize=(fig_width,0.8))
        logo.style_spines(visible=False)
        logo.style_spines(spines=["left","bottom"], visible=True)
        plt.show()


    # Calculate the average
    total_df = pd.DataFrame()
    for df_logo in df_logo_all:
        total_df = total_df.add(df_logo, fill_value=0)
    average_df = total_df / len(df_logo_all)

    extracted_values = average_df.max(axis=1).tolist()   # make the average score into a list
    extracted_values = [round(item,3) for item in  extracted_values] 

    df_logo_all.append(average_df)

    print("---------- Average Score Motif Logo ----------")
    logo2=logomaker.Logo(average_df,color_scheme=state_col_dict_num, figsize=(fig_width+1,1))
    logo2.style_spines(visible=False)
    logo2.style_spines(spines=["left","bottom"], visible=True)
    plt.show()
#     print(average_df)
    logo_score = df_logo_all[-1].max(axis=1).tolist()
    logo_score = [round(item,3) for item in  extracted_values] 
    print("Average logo score:",logo_score)

    return logo_score 


# In[92]:


def motif_logo(mat_path, dev_path, motif):
    all_dict_1, all_dict_0=get_matWcss(mat_path,dev_path)
    motif_found_all,score_found_all,score_found_norm_all=get_motifWScore(all_dict_1, motif=motif)
    logo_score=score2logo(motif_found_all,score_found_all,score_found_norm_all)
    return


# In[93]:


# #### test
# mat_path="../database/ft_result/pred/4_gene_exp/test02_double_data/Ghexp_rpkm30_or_not/atten.npy"
# dev_path="../database/fine_tune/gene_exp/4mer/Ghexp_rpkm30_or_not/tr_len_40k/dev.tsv"
# motif_logo(mat_path, dev_path, motif="GBBBG")
# #### test


# #### Motif Word Cloud Visualization
# 
# * Usage: `motif2wordcloud(motif_dir, color_map="viridis")`
# * Note that `motif_dir` is a directory where the motif files are collected.

# In[35]:


def motif2wordcloud(path, color_map="viridis"):
    target=[word for word in path.split("/")[-2:] if word !=""][0]
    print("target", target)
    file_lst=[os.path.join(path,file) for file in os.listdir(path) if ".txt" in file]
    motifs={}
    for file_name in file_lst:
        motif, num_txt=file_name.split("/")[-1].split("_")[1:3]
        freq=num_txt.split(".")[0]
        motifs[motif]=int(freq)
    print("motifs = ", motifs)
    wc=WordCloud(width=800, height=400, background_color="white", colormap=color_map)
    wordcloud=wc.generate_from_frequencies(motifs)
    plt.figure(figsize=(6,2), facecolor=None)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()


# In[95]:


# #### test
# old_path="../database/motif/motif_test/"
# motif2wordcloud(old_path, color_map="viridis")
# #### test


# #### Motif Clustering

# In[36]:


def motif_init2df(input_path="./init_concat.csv"):
    """
    Read init.csv file and convert it to 
    """
    df=pd.read_csv(input_path)
    data_lst=df["motif"].to_list()
    def convert_sequence(sequence, mapping):
        return [mapping[letter] for letter in sequence]
    letter_to_num = {'A': 1,'B': 2,'C': 3,'D': 4,'E': 5,'F': 6,'G': 7,
                     'H': 8,'I': 9,'J': 10,'K': 11,'L': 12,'M': 13,'N': 14,'O': 15}
    numerical_sequences=[convert_sequence(seq, letter_to_num) for seq in data_lst]
    df_sequences = pd.DataFrame(numerical_sequences).astype('Int64').T
    # Add an 'entry' column at the beginning of the DataFrame with labels 'Entry 1', 'Entry 2', etc.
    df_sequences.insert(0, 'position', ['Pos ' + str(i+1) for i in range(df_sequences.shape[0])])
    return df_sequences


# In[97]:


# test for motif_init2df
# df_sequences=motif_init2df(input_path="./init_concat.csv")
# df_sequences
# test passed


# In[37]:


####### update completed for python 3.11.

def motif_init2pred_with_dendrogram(
    input_path="./init_concat.csv",
    categorical=False,
    fillna_method="ffill",
    n_clusters=None,
    linkage_method="complete",
    threshold=35
):
    """
    Read init.csv file and directly predict the class using DTW and Agglomerative Clustering.
    This version includes forward-reverse comparison. A dendrogram is provided to help the user
    determine the optimal number of clusters.
    """
    # from scipy.spatial.distance import squareform

    def dataframe_reverse_and_push_nan(df):
        # Reverse all columns except 'position'
        df_rev = df.loc[:, df.columns != 'position'].apply(lambda col: col[::-1].values, axis=0)
        df_rev.insert(0, 'position', df['position'])  # Reinsert position column
        df_rev = df_rev.reset_index(drop=True)

        for col in df_rev.columns:
            if col != 'position':
                non_nan = df_rev[col].dropna()
                nan_count = df_rev[col].isna().sum()

                # Ensure consistent dtype even if non_nan is empty
                dtype = non_nan.dtype if not non_nan.empty else 'object'
                nan_padding = pd.Series([pd.NA] * nan_count, dtype=dtype)

                # Explicit casting to ensure consistency
                df_rev[col] = pd.concat([non_nan.astype(dtype), nan_padding], ignore_index=True)

        return df_rev


    df_sequences = motif_init2df(input_path=input_path)

    X_train = df_sequences.loc[:, df_sequences.columns != 'position']
    if fillna_method == 0:
        X_train_filled = X_train.fillna(0)
    elif fillna_method == "ffill":
        X_train_filled = X_train.ffill().apply(pd.to_numeric, errors='coerce')

    n_columns = X_train_filled.shape[1]

    dtw_distance_matrix = np.zeros((n_columns, n_columns))

    df_sequences_rev = dataframe_reverse_and_push_nan(df_sequences)
    X_train_rev = df_sequences_rev.loc[:, df_sequences_rev.columns != 'position']
    if fillna_method == 0:
        X_train_filled_rev = X_train_rev.fillna(0)
    elif fillna_method == "ffill":
        X_train_filled_rev = X_train_rev.ffill().apply(pd.to_numeric, errors='coerce')

    def categorical_distance(a, b):
        return 0 if a == b else 1

    def dtw_categorical(seq1, seq2):
        n, m = len(seq1), len(seq2)
        dtw = np.full((n+1, m+1), np.inf)
        dtw[0, 0] = 0
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = categorical_distance(seq1[i-1], seq2[j-1])
                dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
        return dtw[n, m]


    start_dtw = time.time()
    for i in range(n_columns):
        for j in range(i, n_columns):
            if categorical:
                distance_ff = dtw_categorical(X_train_filled.iloc[:, i].values, X_train_filled.iloc[:, j].values)
                distance_fr = dtw_categorical(X_train_filled.iloc[:, i].values, X_train_filled_rev.iloc[:, j].values)
            else:
                distance_ff = dtw(X_train_filled.iloc[:, i].values, X_train_filled.iloc[:, j].values)
                distance_fr = dtw(X_train_filled.iloc[:, i].values, X_train_filled_rev.iloc[:, j].values)

            min_distance = min(distance_ff, distance_fr)
            dtw_distance_matrix[i, j] = min_distance
            dtw_distance_matrix[j, i] = min_distance

    end_dtw = time.time()
    print(f"DTW computation time: {end_dtw - start_dtw:.4f} seconds")

    start_clustering = datetime.now()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ClusterWarning)
        Z = linkage(dtw_distance_matrix, method=linkage_method)
    # Z = linkage(squareform(dtw_distance_matrix), method=linkage_method)

    plt.figure(figsize=(8, 6))
    dendrogram(Z, color_threshold=threshold)
    plt.title('Dendrogram')
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    plt.show()

    if n_clusters is None:
        y_pred = fcluster(Z, threshold, criterion='distance')
        estimated_clusters = len(np.unique(y_pred))
        print(f"Estimated number of clusters: {estimated_clusters}")
    else:
        from sklearn.cluster import AgglomerativeClustering
        clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage=linkage_method)
        y_pred = clustering.fit_predict(dtw_distance_matrix)
        estimated_clusters = n_clusters

    end_clustering = datetime.now()
    print(f'Clustering Duration: {end_clustering - start_clustering}')
    print(f"Number of cluster estimated by dendrogram with designated threshold {threshold}: [{estimated_clusters}] clusters")

    print("Min distance:", np.min(dtw_distance_matrix))
    print("Max distance:", np.max(dtw_distance_matrix))
    print("Median distance:", np.median(dtw_distance_matrix))

    return y_pred


# In[70]:


# # test for motif_init2pred_with_dendrogram
# y_pred=motif_init2pred_with_dendrogram(input_path="./init_concat.csv", categorical=False, fillna_method="ffill",linkage_method='complete', threshold=35)
# # test passed


# In[39]:


####### update completed for python 3.11.

def motif_init2pred(input_path="./init_concat.csv", categorical=False, fillna_method="ffill", n_clusters=11, linkage_method="complete"):
    """
    Read init.csv file and directly predict the class using DTW and Agglomerative Clustering.
    This version includes forward-reverse comparison.
    This version considers the difference in state as categorical if categorical=True.

    Parameters:
    - input_path (str): Path to the input CSV file.
    - categorical (bool): Whether to treat the data as categorical.
    - fillna_method (str or int): Method to fill missing values. Use "ffill" for forward fill or 0 to fill with zeros.
    - n_clusters (int): Number of clusters for Agglomerative Clustering.
    - linkage_method (str): Linkage method for Agglomerative Clustering. Options are "complete", "average", or "single".

    Returns:
    - dtw_distance_matrix (np.ndarray): DTW distance matrix.
    - y_pred (np.ndarray): Predicted cluster labels.

    """
    def dataframe_reverse_and_push_nan(df):
        # Reverse rows of each column except 'position'
        df_rev = df.loc[:, df.columns != 'position'].apply(lambda col: col[::-1].values, axis=0)
        # Add the 'position' column back without changing its order
        df_rev.insert(0, 'position', df['position'])
        # Reset row index to ensure continuous row index
        df_rev = df_rev.reset_index(drop=True)
        # Push NaN values to the end of each column except 'position'
        for col in df_rev.columns:
            if col != 'position':
                non_nan = df_rev[col].dropna()
                nan_count = df_rev[col].isna().sum()
                df_rev[col] = pd.concat([non_nan, pd.Series([pd.NA] * nan_count)], ignore_index=True)    
        return df_rev

    df_sequences = motif_init2df(input_path=input_path)
    # print(f"Initial number of entries: {len(df_sequences)}")  # Debug print

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=ClusterWarning)

        if fillna_method not in [0, "ffill"]:
            raise ValueError("Invalid fillna_method. Use 0 or 'ffill'.")

        X_train = df_sequences.loc[:, df_sequences.columns != 'position']
        if fillna_method==0:
            X_train_filled = X_train.fillna(0)
        if fillna_method=="ffill":
            X_train_filled = X_train.fillna(method=fillna_method) 
        n_columns = X_train_filled.shape[1]
        # print(f"Number of entries after filling NaNs: {n_columns}")  # Debug print

        dtw_distance_matrix = np.zeros((n_columns, n_columns))

        df_sequences_rev = dataframe_reverse_and_push_nan(df_sequences)  # Reverse
        X_train_rev = df_sequences_rev.loc[:, df_sequences_rev.columns != 'position']
        if fillna_method==0:
            X_train_filled_rev = X_train_rev.fillna(0)
        if fillna_method=="ffill":
            X_train_filled_rev = X_train_rev.fillna(method=fillna_method) 
        # X_train_filled_rev = X_train_rev.fillna(fillna_method)  # Fill missing values with zero    

    def categorical_distance(a, b):
        return 0 if a == b else 1

    def dtw_categorical(seq1, seq2):
        n, m = len(seq1), len(seq2)
        dtw = np.full((n+1, m+1), np.inf)
        dtw[0, 0] = 0
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = categorical_distance(seq1[i-1], seq2[j-1])
                dtw[i, j] = cost + min(dtw[i-1, j],    # insertion
                                    dtw[i, j-1],    # deletion
                                    dtw[i-1, j-1])  # match
        return dtw[n, m]

    import time
    start_dtw = time.time()
    for i in range(n_columns):
        for j in range(i, n_columns):  # No need to compute the distance twice for (i, j) and (j, i)
            if categorical:
                distance_ff=dtw_categorical(X_train_filled.iloc[:, i].values, X_train_filled.iloc[:, j].values)  # Forward-forward
                distance_fr=dtw_categorical(X_train_filled.iloc[:, i].values, X_train_filled_rev.iloc[:, j].values)  # Forward-reverse
            else:
                distance_ff = dtw(X_train_filled.iloc[:, i].values, X_train_filled.iloc[:, j].values)  # Forward-forward
                distance_fr = dtw(X_train_filled.iloc[:, i].values, X_train_filled_rev.iloc[:, j].values)  # Forward-reverse

                # Select the minimum distance and assign symmetrically
            min_distance = min(distance_ff, distance_fr)
            dtw_distance_matrix[i, j] = min_distance
            dtw_distance_matrix[j, i] = min_distance

    end_dtw = time.time()
    print(f"DTW computation time: {end_dtw - start_dtw} seconds")

    # Measure time for clustering
    start_clustering = datetime.now()

    # Use Agglomerative Clustering with the precomputed DTW distance matrix
    # for linkage option, 
    # complete: mazimize the minimum distance between points in different clusters
    # average: uses the average distance between all points in the two clusters
    # single: minimize the distance between the closest points of the clusters
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage=linkage_method)
    y_pred = clustering.fit_predict(dtw_distance_matrix)

    end_clustering = datetime.now()
    print('Clustering Duration: {}'.format(end_clustering - start_clustering))

    # print(f"Number of cluster labels: {len(y_pred)}")  # Debug print
    # return X_train_filled, X_train_filled_rev, y_pred
    return dtw_distance_matrix, y_pred



# In[41]:


# # test
# motif_init2pred(categorical=False, fillna_method="ffill", n_clusters=11, linkage_method="complete")
# # test passed


# In[47]:


def motif_init2class(input_path="./init_concat.csv", categorical=False, fillna_method="ffill", n_clusters=11, linkage_method="complete"): #,fillna_method='ffill'):
    df_sequences=motif_init2df(input_path=input_path)

    # Transpose df_test so that each entry becomes a row
    df_seq_transposed = df_sequences.T  
    # The first row will likely contain something other than data (e.g., time points), so let's keep it as a header
    new_header = df_seq_transposed.iloc[0]  # Grab the first row for the header
    df_seq_transposed = df_seq_transposed[1:]  # Take the data less the header row
    df_seq_transposed.columns = new_header  # Set the header row as the df header
    # Reset the index to make the entries into a column
    df_seq_transposed.reset_index(inplace=True)
    # Rename the 'index' column to something more descriptive, like 'Entry'
    df_seq_transposed.rename(columns={'index': 'Entry'}, inplace=True)

    _, y_pred=motif_init2pred(input_path=input_path,categorical=categorical,fillna_method=fillna_method,  n_clusters=n_clusters, linkage_method=linkage_method)

    # Add the cluster labels as a new column
    df_seq_transposed['Cluster'] = y_pred
    # Sort the DataFrame by the 'Cluster' column
    df_sorted_by_cluster = df_seq_transposed.sort_values(by='Cluster')
    # Reset the index of the sorted DataFrame
    df_sorted_by_cluster.reset_index(drop=True, inplace=True)
    # Display the sorted DataFrame
    # df_sorted_by_cluster
    # Reverse the letter_to_num mapping
    letter_to_num = {'A': 1,'B': 2,'C': 3,'D': 4,'E': 5,'F': 6,'G': 7,
                        'H': 8,'I': 9,'J': 10,'K': 11,'L': 12,'M': 13,'N': 14,'O': 15}
    num_to_letter = {v: k for k, v in letter_to_num.items()}

    # Function to convert a series of numbers to a letter string, ignoring NaNs
    def series_to_letters(series):
        return ''.join([num_to_letter.get(x, '') for x in series if pd.notna(x)])

    # Apply the conversion to each row (excluding the 'Cluster' column) and add the result to a new column
    df_sorted_by_cluster['LetterSequence'] = df_sorted_by_cluster.drop('Cluster', axis=1).apply(series_to_letters, axis=1)

    # Group by 'Cluster' and aggregate 'LetterSequence' into lists
    clustered_sequences = df_sorted_by_cluster.groupby('Cluster')['LetterSequence'].apply(list).reset_index()

    # Display the result
    return clustered_sequences


# In[103]:


# # test
# motif_init2class(categorical=True)
# # test passed


# In[42]:


def motif_init2class_vis(input_path="./init_concat.csv", categorical=False, fillna_method="ffill", n_clusters=11, linkage_method="complete"):
    """
    Read init.csv file and visualize the predicted class using DTW and Agglomerative Clustering.
    This version includes forward-reverse comparison.
    This version considers the difference in state as categorical if categorical=True.

    Parameters:
    - input_path (str): Path to the input CSV file.
    - categorical (bool): Whether to treat the data as categorical.
    - fillna_method (str or int): Method to fill missing values. Use "ffill" for forward fill or 0 to fill with zeros.
    - n_clusters (int): Number of clusters for Agglomerative Clustering.


    Returns:
    - Visualization of each motif clustered in the same line color and line style.
    - X axis indicates the position of the chromatin state in a sequence
    - Y axis represents the type of chromatin state
    """
    df_sequences = motif_init2df(input_path=input_path)
    _, y_pred = motif_init2pred(input_path=input_path, categorical=categorical,fillna_method=fillna_method, n_clusters=n_clusters, linkage_method=linkage_method)

    from itertools import cycle
    from matplotlib import cm

    # Set the figure size and legend location
    rcParams["figure.figsize"] = (12, 6)
    rcParams["legend.loc"] = 'upper right'

    # Assuming df_sequences is your DataFrame and y_pred is your array of predicted cluster labels
    item_list = df_sequences.columns.tolist()[1:]

    # Define a list of linestyles
    linestyles = ['-', '--', '-.', ':']

    # Create a cycle object from the linestyles list
    linestyle_cycle = cycle(linestyles)

    # Assign a linestyle to each cluster, cycling through the available styles
    cluster_linestyles = {i: next(linestyle_cycle) for i in range(n_clusters)}

    # Load a colormap
    cmap = cm.get_cmap('tab20', n_clusters) 

    # Map y_pred to colors
    cluster_colors = {i: cmap(i / n_clusters) for i in range(n_clusters)}

    # Create a figure and a subplot
    fig, ax = plt.subplots()

    # Use these colors in your plot
    for index, item in enumerate(item_list):
        linestyle = cluster_linestyles[y_pred[index]]
        color = cluster_colors[y_pred[index]]  # Get color from cluster_colors
        ax.plot(df_sequences["position"], df_sequences[item].astype('float'), 
                label=f"{item}_cluster{y_pred[index]}", 
            linestyle=linestyle, color=color)


    ax.set_xticks(df_sequences["position"])
    ax.set_xticklabels(df_sequences['position'].str.extract('(\d+)')[0].astype(int),fontsize=14)
    y_ticks = np.arange(1, len(state_col_dict_num.keys()) + 1)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(list(state_col_dict_num.keys()), fontsize=14)  
    # fig.savefig("./test_vis1.png",bbox_inches='tight', dpi=300)
    # Show the plot
    plt.show()


# In[44]:


# # test
# motif_init2class_vis()
# # test passed


# In[90]:


####### update completed for python 3.11.

def motif_init2cluster_vis(input_path="./init_concat.csv", categorical=False, n_clusters=11, fillna_method="ffill", linkage_method="complete", random_state=82, font_scale=0.004,font_v_scale=9, fig_w=12, fig_h=8, node_size=1000, node_dist=0.05):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)

        clustered_sequences=motif_init2class(input_path=input_path, categorical=categorical, fillna_method=fillna_method, n_clusters=n_clusters, linkage_method=linkage_method)
        scale_factor = font_scale  # Adjust this to change the font size

    def create_text_patch(x, y, text, state_col_dict_num, ax, scale_factor):
        # Determine the starting x position for the first letter
        x_offset = x
        for letter in text:
            color = state_col_dict_num.get(letter, (0, 0, 0))
            # fp = FontProperties(family="Arial", weight="bold")
            fp = FontProperties(family="DejaVu Sans", weight="bold")
            tp = TextPath((0, 0), letter, prop=fp)
            tp_transformed = transforms.Affine2D().scale(scale_factor).translate(x_offset, y) + ax.transData
            letter_patch = PathPatch(tp, color=color, lw=0, transform=tp_transformed)
            ax.add_patch(letter_patch)
            # Get the width of the letter and add a small margin
            letter_width = tp.get_extents().width * scale_factor
            x_offset += letter_width  # Increment the x position by the width of the letter

    df = clustered_sequences

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))  # Adjust figure size as needed

    ##### color modification for temporary use #####
    # Create a temporary copy with a different name
    temp_state_col_dict_num = state_col_dict_num.copy()

    # Modify the colors in the temporary dictionary
    # Update 'G' to a more visible color, such as a deep orange
    temp_state_col_dict_num['G'] = (1.0, 0.647, 0.0)  # Normalized deep orange

    # Update 'O' to ensure it stands out more, such as a darker gray
    temp_state_col_dict_num['O'] = (0.502, 0.502, 0.502)  # Normalized darker gray

    ################################################

    # Create a graph
    G = nx.Graph()
    for index, row in df.iterrows():
        G.add_node(row['Cluster'], elements=row['LetterSequence'])

    # Significantly increase the base size for each node
    base_node_size = node_size  # This increases the node size
    node_sizes = [len(elements) * base_node_size for elements in df['LetterSequence']]

    # Generate a color palette with a unique color for each node
    colors = plt.cm.rainbow(np.linspace(0, 1, len(df)))

    np.random.seed(random_state)
    # Draw the graph with a spring layout
    # Adjust k to manage the distance between nodes, which can be smaller since nodes can overlap

    ############# position test
    # pos = nx.spring_layout(G, k=node_dist, iterations=10)
    pos = nx.spring_layout(G, k=node_dist, iterations=10, seed=random_state)
    #############

    # Draw the nodes themselves
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=colors, alpha=0.3)

    # Draw the text
    for node, (node_pos, elements) in enumerate(zip(pos.values(), df['LetterSequence'])):      
        x_start, y_start = node_pos
        for i, element in enumerate(elements):
            x_position = x_start - 0.08
            y_position = y_start - (i * scale_factor * font_v_scale) + 0.015*len(elements)# Adjust line spacing
            create_text_patch(x_position, y_position, element, temp_state_col_dict_num, ax, scale_factor)
#             print("state_col_dict_num", state_col_dict_num)

    plt.axis('off')
    print(n_clusters)
    plt.show()

    # fig.savefig("./cluster_result.png",bbox_inches='tight', dpi=300,facecolor='white',  # Set the background color to white
    # edgecolor='black')
    fig.savefig("./cluster_result.pdf", bbox_inches='tight', dpi=300, facecolor='white', edgecolor='black')


# In[92]:


# # test
# motif_init2cluster_vis(categorical=True)
# # test passed


# In[ ]:


####### update completed for python 3.11.
def motif_init2umap(input_path="./init_concat.csv",categorical=False,  n_clusters=11, fillna_method="ffill", linkage_method="complete", n_neighbors=5, min_dist=0.3, random_state=2):
    """
    Generate a UMAP embedding of the given data.

    Parameters:

    - input_path: .csv file of all motifs with high attention score

    - n_clusters: number of clusters

    - n_neighbors: int (default=5), The size of local neighborhood (in terms of number of neighboring sample points) 
      used for manifold approximation. Larger values result in a more global view of the manifold, while smaller values emphasize local data structures. 
      Adjust according to the desired granularity of the embedding.

    - mid_dist: float (default=0.3), The minimum distance between embedded points in the low-dimensional space. 
      Smaller values allow points to cluster more tightly in the embedding, which is useful for identifying finer substructures within the data. 
      Larger values help preserve the overall topology of the data by preventing points from clustering too tightly.
    """
    df_sequences = motif_init2df(input_path=input_path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=ClusterWarning)

        X_train = df_sequences.loc[:, df_sequences.columns != 'position']
        X_train = X_train.astype('float64')  # Convert to float64
        # X_filled = X_train.fillna(X_train.mean())
        if fillna_method==0:
            X_train_filled = X_train.fillna(0)
        if fillna_method=="ffill":
            X_train_filled = X_train.fillna(method=fillna_method) 

        dtw_distance_matrix, y_pred = motif_init2pred(input_path=input_path, categorical=categorical, n_clusters=n_clusters, fillna_method=fillna_method, linkage_method=linkage_method)

        # Now apply UMAP on the cleaned data
        from umap import UMAP
        # # seed=111
        # # umap_reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed)
        # umap_reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
        umap_reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state, n_jobs=1)

        # umap_embedding = umap_reducer.fit_transform(X_train_filled.T)  # Ensure the data is transposed if necessary
        umap_embedding = umap_reducer.fit_transform(dtw_distance_matrix)  # Ensure the data is transposed if necessary

    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=y_pred, cmap='Spectral', s=100, edgecolors='white', linewidth=0.6)

    # Create a color bar with ticks for each cluster label
    colorbar = plt.colorbar(scatter, ticks=np.arange(0, 11))
    colorbar.set_label('Cluster label')

    # Set the plot title and labels
    plt.title('UMAP Projection After Agglomerative Clustering', fontsize=14)
    plt.xlabel('UMAP Dimension 1', fontsize=14)
    plt.ylabel('UMAP Dimension 2', fontsize=14)

    # Show the plot
    plt.show()


# In[95]:


# # test
# motif_init2umap(categorical=True,  n_clusters=11, fillna_method="ffill", linkage_method="complete", n_neighbors=5, min_dist=0.3, random_state=2)
# # test passed


# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[110]:


# ############# This is to enable easy execution of the function

# if __name__ == "__main__":
#     if len(sys.argv) > 1:
#         # Filter out arguments that are not function names
#         func_name = sys.argv[1]

#         # Check if func_name is a valid function in globals()
#         if func_name in globals() and callable(globals()[func_name]):
#             globals()[func_name]()  # Run the function by name
#         else:
#             print(f"Function '{func_name}' not found.")
# ###############################################################


# In[ ]:


# Parse arguments for command-line use
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run unzipped_to_df function")
    parser.add_argument("function", help="Name of the function to run")
    parser.add_argument("args", nargs="*", help="Arguments for the function")

    args = parser.parse_args()

    if args.function == "unzipped_to_df" and len(args.args) >= 1:
        unzipped_to_df(args.args[0], args.args[1] if len(args.args) > 1 else "./")
    else:
        print("Usage: python3 css_utility.py unzipped_to_df <path_unzipped> <output_path>")


# In[ ]:




