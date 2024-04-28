#!/usr/bin/env python
# coding: utf-8

# # CSS utility
# 
# Functions that can be exploited for data pre-processing and downstream analysis

# In[3]:


# ### To convert the file into .py
# !jupyter nbconvert --to script css_utility_working.ipynb


# In[60]:


import os
import re
import random
import operator
import itertools
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
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
import matplotlib.transforms as transforms
import networkx as nx

import seaborn as sns
from scipy.stats import norm
from sklearn.metrics import confusion_matrix, classification_report
from tslearn.clustering import TimeSeriesKMeans

from tqdm import tqdm, notebook
from tqdm.notebook import tqdm_notebook

from wordcloud import WordCloud
# import stylecloud


# ### Useful Dictionaries

# In[2]:


state_dict={1:"A", 2:"B", 3:"C", 4:"D", 5:"E",6:"F",7:"G",8:"H" ,
                9:"I" ,10:"J",11:"K", 12:"L", 13:"M", 14:"N", 15:"O"}


# In[71]:


css_name=['TssA','TssAFlnk','TxFlnk','Tx','TxWk','EnhG','Enh','ZNF/Rpts',
          'Het','TssBiv','BivFlnk','EnhBiv','ReprPC','ReprPcWk','Quies']


# In[72]:


css_dict=dict(zip(list(state_dict.values()), css_name))  # css_dict={"A":"TssA", "B":"TssAFlnk", ... }


# In[69]:


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


# In[73]:


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


# In[74]:


def colors2color_dec(css_color_dict):
    colors=list(css_color_dict.values())
    color_dec_list=[]
    for color in colors:
        color_dec=tuple(rgb_elm/255 for rgb_elm in color)
        color_dec_list.append(color_dec)        
    return color_dec_list


# **scale 0 to 1**

# In[75]:


state_col_dict=dict(zip(list(state_dict.values()),colors2color_dec(css_color_dict)))


# **scale 0 to 255**

# In[70]:


state_col_255_dict=dict(zip(list(state_dict.values()),list(css_color_dict.values())))


# **hexacode**

# In[76]:


hexa_state_col_dict={letter: "#{:02x}{:02x}{:02x}".format(*rgb) for letter,rgb in state_col_255_dict.items()}


# **name instead of alphabets**

# In[77]:


css_name_col_dict=dict(zip(css_name,state_col_dict.values()))


# ### Helper functions

# In[3]:


def flatLst(lst):
    flatten_lst=[elm for sublst in lst for elm in sublst]
    return flatten_lst


# In[68]:


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


# In[4]:


def seq2kmer(seq, k):
    """
    Convert original sequence to kmers
    """
    kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
    kmers = " ".join(kmer)
    return kmers


# In[5]:


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


# In[78]:


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

# In[6]:


def bed2df_expanded(filename):
    
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
               
    df["state_seq"]=df["state"].map(state_dict)
    df["state_seq_full"]=df["unit"]*df["state_seq"]
    
    return df 


# In[7]:


# # test for bed2df_expanded
# test_path_bed='../database/bed/unzipped/E001_15_coreMarks_stateno.bed'
# test_bed2df_expanded=bed2df_expanded(test_path_bed)
# test_bed2df_expanded.head()
# # test passed


# In[8]:


def unzipped_to_df(path_unzipped, output_path="./"):
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
        df=bed2df_expanded(file)
        df.to_pickle(output_name)
        # if cell_id=="E002":  ###### for test
        #     break
    return print("Files saved to {}".format(output_path))
# unzipped_to_df(unzipped_epi_files, output_path="../database/roadmap/df_pickled/")


# In[9]:


# # test for unzipped_to_df
# path_unzipped='../database/bed/unzipped'
# test_unzipped_to_df=unzipped_to_df(path_unzipped,output_path="../database/final_test")
# # test passed


# In[10]:


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


# In[11]:


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


# In[12]:


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


# In[13]:


# # test for df2unitcss
# with open("../database/final_test/E001_df_pickled.pkl","rb") as f:
#     test_df=pickle.load(f)
# all_unit_css=df2unitcss(test_df)
# print(len(all_unit_css))
# print(type(all_unit_css))
# # test passed


# In[14]:


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


# In[15]:


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


# In[16]:


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


# In[17]:


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


# In[18]:


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


# In[19]:


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


# In[20]:


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


# In[21]:


# # test for save_TSS_by_loc
# whole_gene_file='../database/RefSeq/RefSeq.WholeGene.bed'
# save_TSS_by_loc(whole_gene_file=whole_gene_file, input_path="../database/roadmap/df_pickled/",output_path="../database/final_test/",file_name="up2kdown4k", up_num=2000, down_num=4000, unit=200)
# # test passed


# In[22]:


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


# In[23]:


# # test for prom_css_Kmer_by_cell
# path="../database/roadmap/prom/up2kdown4k/all_genes/"
# output_path="../database/final_test/"
# prom_css_Kmer_by_cell(path=path, output_path=output_path, k=4)
# # test passed


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

# In[24]:


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

# In[25]:


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


# In[26]:


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

# In[27]:


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


# In[28]:


def gene_removeDupl(whole_gene_file='../database/RefSeq/RefSeq.WholeGene.bed'):
    g_df_chr_lst=whGene2GLChr(whole_gene_file)
    new_gene_lst_all=[]
    for chr_no in range(len(g_df_chr_lst)):
        gene_df=g_df_chr_lst[chr_no]
        gene_collapsed_df=removeOverlapDF(gene_df)
        new_gene_lst_all.append(gene_collapsed_df)
    return new_gene_lst_all # list of chromosome-wise dataframe for collapsed gene table


# #### Function `extNsaveProm_g_exp`
# * This function processes the above works (cut the prom region and make it unit length css) per cell
# * Input
#     * `exp_gene_dir`: directory where refFlat for each cell (subdir means the sub directory for different gene expression level)
#     * `df_pickle_dir`: dataframe of each cell
#     * `rpkm_val`: RPKM value, 10, 20, 30, or 50
#     * `up_num`: upstream of gene
#     * `down_num`: from TSS (gene initial part) to cut
#     * `unit`: because chromatin states are annotated by 200 bps
# * Output: save the file according to the `rpkm_val` at the output path

# In[29]:


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


# In[30]:


# test for extNsaveProm_g_exp
# extNsaveProm_g_exp(exp_gene_dir="../database/roadmap/gene_exp/refFlat_byCellType/", df_pickle_dir="../database/roadmap/df_pickled/",output_path="../database/final_test/",file_name="up2kdown4k",rpkm_val=50, up_num=2000, down_num=4000,unit=200)
# test passed


# ### Extract Promoter regions from not expressed genes

# #### Pipeline 
# 
# (1) `extWholeGeneRef` : Just extract the whole gene location files from `chr.gene.refFlat` <br>
# (2) `extNOTexp_by_compare` : Extract the not expressed genes by comparing with whole gene with rpkm>0 <br>
# (3) `extNsaveNOTexp_by_compare` : load the required file and process all, and save refFlat (.pkl) and prom-region css (.pkl)

# In[31]:


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


# In[32]:


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


# In[33]:


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


# In[34]:


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

# In[35]:


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


# In[36]:


# test for prom_css_Kmer_by_cell
# prom_css_Kmer_by_cell(path="../database/roadmap/prom/up2kdown4k/all_genes/", output_path="../database/final_test/",k=4)
# test passed


# #### Motif Clustering

# In[37]:


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


# In[38]:


# # test for motif_init2df
# df_sequences=motif_init2df(input_path="./init_concat.csv")
# df_sequences.head()
# # test passed


# In[39]:


def motif_init2pred(input_path="./init_concat.csv", n_clusters=11):
    """
    Read init.csv file and directly predict the class using DTW and k-mean
    """
    from tslearn.metrics import dtw

    df_sequences=motif_init2df(input_path=input_path)
    X_train = df_sequences.loc[:, df_sequences.columns != 'position']
    # Fill missing values with zero
    X_train_filled = X_train.fillna(0)
    # Then proceed with the DTW distance matrix computation
    n_series = X_train_filled.shape[0]
    dtw_distance_matrix = np.zeros((n_series, n_series))

    for i in range(n_series):
        for j in range(i, n_series):  # No need to compute the distance twice for (i, j) and (j, i)
            distance = dtw(X_train_filled[i], X_train_filled[j])
            dtw_distance_matrix[i, j] = distance
            dtw_distance_matrix[j, i] = distance
    seed=111
    start_time = datetime.now()
    # print("DTW k-means")
    dba_km = TimeSeriesKMeans(n_clusters=n_clusters,
        n_init=10, #2,  # number of time you run with different initial centroid 
        metric="dtw",
        verbose=False, #True,
        max_iter_barycenter=10,
        random_state=seed)
    y_pred = dba_km.fit_predict(X_train.T)
    # print(y_pred)
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
    return y_pred


# In[45]:


# # test for motif_init2pred
# y_pred=motif_init2pred(input_path="./init_concat.csv", n_clusters=11)
# # test passed


# In[53]:


def motif_init2elbow(input_path="./init_concat.csv", n_start=1, n_end=25):
    df_sequences = motif_init2df(input_path=input_path)
    X_train = df_sequences.loc[:, df_sequences.columns != 'position']
    n_cluster_range = range(n_start, n_end + 1)
    inertia = []
    seed = 111

    for n_clusters in tqdm(n_cluster_range, desc="Calculating clusters"):
        model = TimeSeriesKMeans(n_clusters=n_clusters,
                                 metric="dtw",
                                 verbose=False,
                                 max_iter_barycenter=10,
                                 random_state=seed)
        model.fit(X_train.T)
        inertia.append(model.inertia_)

    # Filter out infinite values and their corresponding cluster numbers
    finite_inertia = [i for i in inertia if np.isfinite(i)]
    n_cluster_finite = list(n_cluster_range)[:len(finite_inertia)]

    # Plotting the inertia with finite values only
    plt.figure(figsize=(12, 6))
    plt.plot(n_cluster_finite, finite_inertia, marker='o')
    plt.title('Elbow Method For Optimal Cluster Number', fontsize=16)
    plt.xlabel('Number of clusters', fontsize=16)
    plt.ylabel('Inertia', fontsize=16)
    plt.xticks(n_cluster_finite, fontsize=12)  # Increase x-axis ticks font size
    plt.yticks(fontsize=12)
    plt.grid()
    plt.show()



# In[54]:


# # test for motif_init2elbow
# motif_init2elbow(input_path="./init_concat.csv", n_start=1, n_end=25)
# # test passed


# In[55]:


def motif_init2class_df(input_path="./init_concat.csv", n_clusters=11):
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

    y_pred=motif_init2pred(input_path=input_path, n_clusters=n_clusters)

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


# In[57]:


# # test for motif_init2elbow
# clustered_sequences=motif_init2class_df(input_path="./init_concat.csv", n_clusters=11)
# clustered_sequences.head()
# # test passed


# In[61]:


def motif_init2class_vis(input_path="./init_concat.csv", n_clusters=11):
    df_sequences = motif_init2df(input_path=input_path)
    y_pred = motif_init2pred(input_path=input_path, n_clusters=n_clusters)

    from itertools import cycle

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

    # Create a figure and a subplot
    fig, ax = plt.subplots()

    # Plot each item with its corresponding color (automatically determined by matplotlib) and line style
    for index, item in enumerate(item_list):
        linestyle = cluster_linestyles[y_pred[index]]
        ax.plot(df_sequences["position"], df_sequences[item].astype('float'), 
                label=f"{item}_cluster{y_pred[index]}", 
                linestyle=linestyle)

    # Set x-tick labels with rotation
    ax.set_xticks(df_sequences["position"])
    ax.set_xticklabels(df_sequences["position"].astype(str), rotation=45)

    # Add a legend
    # plt.legend()

    # Show the plot
    plt.show()


# In[63]:


# test for motif_init2class_vis
# motif_init2class_vis(input_path="./init_concat.csv", n_clusters=11)
# # test passed


# In[64]:


def motif_init2umap(input_path="./init_concat.csv", n_clusters=11, n_neighbors=5, min_dist=0.3, random_state=111):
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
    X_train = df_sequences.loc[:, df_sequences.columns != 'position']
    X_train = X_train.astype('float64')  # Convert to float64
    X_filled = X_train.fillna(X_train.mean())

    y_pred = motif_init2pred(input_path=input_path, n_clusters=n_clusters)

    # Now apply UMAP on the cleaned data
    from umap import UMAP
    # # seed=111
    # # umap_reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed)
    # umap_reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
    umap_reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state, n_jobs=1)

    umap_embedding = umap_reducer.fit_transform(X_filled.T)  # Ensure the data is transposed if necessary

    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=y_pred, cmap='Spectral', s=100, edgecolors='white', linewidth=0.6)

    # Create a color bar with ticks for each cluster label
    colorbar = plt.colorbar(scatter, ticks=np.arange(0, 11))
    colorbar.set_label('Cluster label')

    # Set the plot title and labels
    plt.title('UMAP Projection After K-means clustering', fontsize=20)
    plt.xlabel('UMAP Dimension 1', fontsize=15)
    plt.ylabel('UMAP Dimension 2', fontsize=15)

    # Show the plot
    plt.show()


# In[66]:


# # test for motif_init2umap
# motif_init2umap(input_path="./init_concat.csv", n_clusters=11, n_neighbors=5, min_dist=0.3, random_state=111)
# # test passed


# In[4]:


def motif_init2cluster_vis_00(input_path="./init_concat.csv", n_clusters=11, random_state=95, font_scale=0.004,font_v_scale=9, fig_w=10, fig_h=10, node_size=600, node_dist=0.05):
    clustered_sequences=motif_init2class_df(input_path=input_path, n_clusters=n_clusters)
    scale_factor = font_scale  # Adjust this to change the font size

    def create_text_patch(x, y, text, state_col_dict_num, ax, scale_factor):
        # Determine the starting x position for the first letter
        x_offset = x
        for letter in text:
            color = state_col_dict_num.get(letter, (0, 0, 0))
            fp = FontProperties(family="Arial", weight="bold")
            tp = TextPath((0, 0), letter, prop=fp)
            tp_transformed = transforms.Affine2D().scale(scale_factor).translate(x_offset, y) + ax.transData
            letter_patch = PathPatch(tp, color=color, lw=0, transform=tp_transformed)
            ax.add_patch(letter_patch)
            # Get the width of the letter and add a small margin
            letter_width = tp.get_extents().width * scale_factor
            x_offset += letter_width  # Increment the x position by the width of the letter

    df = clustered_sequences

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))  # Adjust figure size as needed

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
    pos = nx.spring_layout(G, k=node_dist, iterations=10)

    # Draw the nodes themselves
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=colors, alpha=0.3)

    # Draw the text
    for node, (node_pos, elements) in enumerate(zip(pos.values(), df['LetterSequence'])):      
        x_start, y_start = node_pos
        for i, element in enumerate(elements):
            x_position = x_start - 0.08
            y_position = y_start - (i * scale_factor * font_v_scale) + 0.015*len(elements)# Adjust line spacing
            create_text_patch(x_position, y_position, element, state_col_dict_num, ax, scale_factor)

    plt.axis('off')
    plt.show()


# In[5]:


def motif_init2cluster_vis(input_path="./init_concat.csv", n_clusters=11, random_state=95, font_scale=0.004,font_v_scale=9, fig_w=10, fig_h=10, node_size=600, node_dist=0.05):
    clustered_sequences=motif_init2class_df(input_path=input_path, n_clusters=n_clusters)
    scale_factor = font_scale  # Adjust this to change the font size

    def create_text_patch(x, y, text, state_col_dict_num, ax, scale_factor):
        # Determine the starting x position for the first letter
        x_offset = x
        for letter in text:
            color = state_col_dict_num.get(letter, (0, 0, 0))
            fp = FontProperties(family="Arial", weight="bold")
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
    pos = nx.spring_layout(G, k=node_dist, iterations=10)

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
    plt.show()
    
    fig.savefig("./test_cluster_2.png",bbox_inches='tight', dpi=300)


# In[ ]:




