from setuptools import find_packages, setup

setup(
    name="chrombert_utils",
    version="0.1.0",
    packages=find_packages(),
    author="Seohyun Lee and Ryuichiro Nakato",
    author_email="rnakato@iqb.u-tokyo.ac.jp",
    description="Data Preprocessing and Analysis for ChromBERT",
    url="https://github.com/caocao0525/ChromBERT",
    license='MIT',
    install_requires=[
        "numpy",
        'scipy',
        'pandas',
        'biopython',
        'scikit-learn',
        'umap-learn',
        'tslearn',
        'statsmodels',
        'seqeval',
        'sentencepiece',
        'matplotlib',
        'seaborn',
        'tqdm',
        'wordcloud',
        'stylecloud',
        'pybedtools'
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        'License :: OSI Approved :: MIT License',
    ],
)
