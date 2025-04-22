#! /usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pandas as pd

def parseRPKMMatrix(filename):
    df = pd.read_csv(filename, sep='\t', index_col=False, header=0)
    df.index = df.iloc[:,0]
    df = df.iloc[:,1:]
    return df

def parserefFlat(filename):
    refFlat = pd.read_csv(filename, sep='\t', header=0)
    refFlat['exonStarts'] = refFlat['exonStarts'].apply(lambda x: [int(i) for i in x.split(',')[:-1]])
    refFlat['exonEnds'] = refFlat['exonEnds'].apply(lambda x: [int(i) for i in x.split(',')[:-1]])
    refFlat.index = refFlat['name']
    return refFlat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input",  help="RPKM matrix", type=str)
    parser.add_argument("cellid", help="Cell type ID (e.g., E050)", type=str)
    parser.add_argument("gene",  help="refFlat", type=str)
    parser.add_argument("--thre_expressed", help="RPKM threshold for expressed or not (default: > 0)", type=int, default=0)
    parser.add_argument("--thre_highlyexpressed", help="RPKM threshold for highly expressed or not (default: > 50)", type=int, default=50)

    args = parser.parse_args()

    print(f"Processing gene classification for cell type: {args.cellid}")
    print(f"Using RPKM matrix: {args.input}")
    print(f"Using gene annotation file: {args.gene}")

    mat = parseRPKMMatrix(args.input)
    gene = parserefFlat(args.gene)

    # Get Not Expressed Genes (RPKM = 0)
    gene_not_expressed = mat[mat[args.cellid] == 0].index
    gene_not_expressed = gene_not_expressed.intersection(gene.index)
    gene_not_expressed_file = 'gene_not_expressed.refFlat'
    gene.loc[gene_not_expressed].to_csv(gene_not_expressed_file, sep='\t')
    print(f"Generated: {gene_not_expressed_file}")
    print(" - Contains genes with RPKM = 0, meaning they are not expressed in this cell type.")

    # Get Highly Expressed Genes (RPKM > args.thre_highlyexpressed)
    gene_highly_expressed = mat[mat[args.cellid] > args.thre_highlyexpressed].index
    gene_highly_expressed = gene_highly_expressed.intersection(gene.index)
    gene_highly_expressed_file = 'gene_highly_expressed.refFlat'
    gene.loc[gene_highly_expressed].to_csv(gene_highly_expressed_file, sep='\t')
    print(f"Generated: {gene_highly_expressed_file}")
    print(f" - Contains genes with RPKM > {args.thre_highlyexpressed}, meaning they are highly expressed in this cell type.")

    print("Processing complete!")

if __name__ == '__main__':
    main()
