"""
Simple check list from AllenNLP repo: https://github.com/allenai/allennlp/blob/master/setup.py

To create the package for pypi.

1. Change the version in __init__.py, setup.py as well as docs/source/conf.py.

2. Commit these changes with the message: "Release: VERSION"

3. Add a tag in git to mark the release: "git tag VERSION -m'Adds tag VERSION for pypi' "
   Push the tag to git: git push --tags origin master

4. Build both the sources and the wheel. Do not change anything in setup.py between
   creating the wheel and the source distribution (obviously).

   For the wheel, run: "python setup.py bdist_wheel" in the top level directory.
   (this will build a wheel for the python version you use to build it).

   For the sources, run: "python setup.py sdist"
   You should now have a /dist directory with both .whl and .tar.gz source versions.

5. Check that everything looks correct by uploading the package to the pypi test server:

   twine upload dist/* -r pypitest
   (pypi suggest using twine as other methods upload files via plaintext.)
   You may have to specify the repository url, use the following command then:
   twine upload dist/* -r pypitest --repository-url=https://test.pypi.org/legacy/

   Check that you can install it in a virtualenv by running:
   pip install -i https://testpypi.python.org/pypi transformers

6. Upload the final version to actual pypi:
   twine upload dist/* -r pypi

7. Copy the release notes from RELEASE.md to the tag in github once everything is looking hunky-dory.

8. Update the documentation commit in .circleci/deploy.sh for the accurate documentation to be displayed

9. Update README.md to redirect to correct documentation.
"""

from setuptools import find_packages, setup

setup(
    name="chrombert",
    version="0.1.0",
    author="Seohyun Lee and Ryuichiro Nakato",
    author_email="",
    description="",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="",
    license="Apache",
    url="https://github.com/",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        "numpy",
        "tokenizers == 0.5.0",
        # accessing files from S3 directly
        "boto3",
        # filesystem locks e.g. to prevent parallel downloads
        "filelock",
        # for downloading models over HTTPS
        "requests",
        # progress bars in model download and training scripts
        "tqdm >= 4.27",
        # for OpenAI GPT
        "regex != 2019.12.17",
        # for XLNet
        "sentencepiece",
        # for XLM
        "sacremoses",
    ],
    extras_require=extras,
    scripts=["transformers-cli"],
    python_requires=">=3.5.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
