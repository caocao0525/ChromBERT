from setuptools import setup, find_packages

setup(
    name="chrombert_utils",
    version="0.2.0",
    packages=find_packages(),
    author="Seohyun Lee and Ryuichiro Nakato",
    author_email="rnakato@iqb.u-tokyo.ac.jp",
    description="Data Preprocessing and Analysis Tools for ChromBERT",
    url="https://github.com/caocao0525/ChromBERT",
    license="MIT",
    python_requires=">=3.11",
    install_requires=[
        "numpy~=2.0",
        "pandas~=2.2",
        "matplotlib~=3.10",
        "networkx~=3.4",
        "seaborn~=0.13",
        "scipy~=1.14",
        "scikit-learn~=1.6",
        "tqdm~=4.67",
        "wordcloud~=1.9",
        "logomaker~=0.8",
        "tslearn~=0.6",
        "umap-learn~=0.5",
        "biopython",               # No version restriction (latest is fine)
        "statsmodels~=0.14",
        "seqeval~=1.2",
        "pyahocorasick",          # No version pinning, but watch for Py 3.11 support
        "sentencepiece~=0.1.99",
        "stylecloud",
        "pybedtools~=0.12",
        "tensorboardX",
        "importlib-metadata"      # Only needed for Py < 3.8, but harmless
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Bio-Informatics"
    ],
)

