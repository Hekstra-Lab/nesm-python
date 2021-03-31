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
- Xarray (time permitting)
- Modular code and pip installation (time permitting)


#### Afternoon (~2 hours)
***Part 2*** Image analysis and I/O
- Intro to the nightmare of I/O
- Skimage
- Frame-wise operations with `np.vectorize`
- napari 

***Part 3*** Mini discussion of hardware control?
- In the spirit of less is more maybe we should spend the whole morning
on python essentials and move part 2 to the afternoon? Feels like you can never have enough? But also who
knows what the experience level of this audience will be compared to undergrads...

***Part 4*** Advanced topics to explore

- Dask and big data
- Sklearn (NMF example and/or PCA on SRS beads data)
- Deep learning (Basically show off YeaZ unet?)
- Other GPU stuff
