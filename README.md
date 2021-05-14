# NESM Python Workshop
Notebooks for the python workshop at the New England Society for Microscopy Spring 2021 Meeting

Prepared and presented by John Russell (johnrussell@g.harvard.edu) and
Ian Hunt-Isaak (ianhuntisaak@g.harvard.edu)


## Where to start:

If you are reading this independently we recommend you start in the `solutions` folder.


## Proposed Schedule
See [`outline.md`](outline.md) for a more detailed outline.

- **Part 1** Python basics
- **Part 2** Hardware Control
- **Part 3** Image analysis and I/O
- **Part 4** Advanced topics

## Getting started

### Online Version

To get a known working envinroment without setting up a local python environment go to this binder link: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Hekstra-Lab/nesm-python/spring-2021?urlpath=lab)

This binder can run all the code except for the end of `part3` where we use `Napari` as Napari cannot run in a browser window.

### Local Installation

Please install `conda`, a package manager, with python 3.8 or 3.9. Follow one of the links below and then follow instructions for your operating system. There are two options
1. [Miniconda](https://docs.conda.io/en/latest/miniconda.html) - an absolutely minimal version
    - [miniforge](https://github.com/conda-forge/miniforge#miniforge) version maintained by open source community
2. [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation) - this version includes conda as well as many other common python packages.


Actual installation steps

1. Open a terminal (mac or linux) or the Anaconda Prompt (windows)
1. Type the following and press enter: `conda create -n nesm python`. This uses `conda` to create a new environment with the name (`-n`) `nesm`. It will install the latest python version in that environment. Whenever you are starting a new project, it is best to create a new environement.
1. `conda info --envs` -- this will list all the environments on your computer. Note that `base` is the default. The `*` next to an environement means that you are currently using that environment.
1. `conda activate nesm` -- Switch to using the newly created `nesm` environment. (If you run `conda info --envs` again you will see that the `*` has moved.)
1. `conda install -c conda-forge numpy scipy matplotlib jupyterlab xarray scikit-image ipympl dask scikit-learn` -- This line uses `conda` to  `install` several common python libraries. We will use all of them in this workshop (time permitting). We also specify to install these libraries from `conda-forge` which is the community maintatined repository for conda installable software. This is where the most up to date versions of software are.
1. `pip install mpl_interactions tifffile dask_labextension hyperspy 'napari[all]'` - This uses `pip` the default python package installer to install a few extra libraries that are not available through conda.

