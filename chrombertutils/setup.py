from setuptools import find_packages, setup

setup(
    name="chrombertutils",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    author="Seohyun Lee and Ryuichiro Nakato",
    author_email="rnakato@iqb.u-tokyo.ac.jp",
    description="Data preprocessing and Analysis for ChromBERT",
    url="https://github.com/caocao0525/ChromBERT",
    license='MIT',
    install_requires=[
        "numpy",
        'scipy',
        'tensorflow==1.14.0',
        'tensorflow-estimator==1.14.0',
        'tensorboard==1.14.0',
        'pandas',
        'biopython',
        'scikit-learn==0.24.2',
        'umap-learn',
        'tslearn',
        'statsmodels',
        'seqeval',
        'pyahocorasick',
        'sentencepiece==0.1.91',
        'matplotlib',
        'seaborn',
        'tqdm',
        'wordcloud',
        'tensorboardX',
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