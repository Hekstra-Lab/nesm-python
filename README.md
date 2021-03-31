# NESM Python Workshop
Notebooks for the python workshop at the New England Society for Microscopy Spring 2021 Meeting

Prepared and presented by John Russell (johnrussell@g.harvard.edu) and
Ian Hunt-Isaak (ianhuntisaak@g.harvard.edu)

### Proposed Schedule

#### Morning 

***Part 1*** Python essentials
- Conda, virtual environments, installation
- Jupyter lab
- Pure python
 - For loops
 - Control flow
 - Lists and dictionaries 
 - Functions

- Numpy
 - Vectorized operations
 - Indexing
 - Basic Broadcasting (possible example something like background subtraction? also `keep-dims`)
 - Numpy for image data

- Matplotlib
 - plot
 - imshow
- Xarray (time permitting)
- Modular code and pip installation (time permitting)
- mpl-interactions feat. hyperslicer


#### Afternoon
 
***Part 2*** Image analysis and I/O
- Intro to the nightmare of I/O
- Structure: two motivating examples for which we illustrate the entire pipeline of analysis
- Napari 
- Many examples (see email suggestions)
- Skimage
- Scipy ndimage
- Frame-wise operations with `np.vectorize`
- Resource list

***Part 3*** Mini discussion of hardware control

***Part 4*** Advanced topics to explore

- Dask and big data
- Sklearn (NMF example and/or PCA on SRS beads data)
- Deep learning (Basically show off YeaZ unet?)
