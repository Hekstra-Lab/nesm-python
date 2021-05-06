__all__=[
    "interactive_threshold"
]

import matplotlib.pyplot as plt
import mpl_interactions.ipyplot as iplt
def interactive_threshold(image, bins='auto', figsize=(12,5)):
    """
    Use sliders to interactively determine the best threshold for an image.


    Parameters
    ----------
    image : (M, N) array
    bins : int or sequence of scalars or str
        The number of bins to use. Passed through to numpy.histogram
    figsize : tuple of numbers
        The size in inches of the figure

    Returns
    -------
    controls : mpl_interactions.controls
        Access the parameters using `controls.params`
    axes : list of matplotlib axes
        In case you want to modify the plot further.

    Notes
    -----
    This is based on the example from: https://mpl-interactions.readthedocs.io/en/stable/examples/range-sliders.html#Using-a-RangeSlider-for-Scalar-arguments---Thresholding-an-Image
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # plot histogram of pixel intensities
    axs[1].hist(image.flatten(), bins=bins)
    axs[1].set_title("Histogram of Pixel Intensities")

    # create interactive controls
    ctrls = iplt.imshow(image, vmin_vmax=("r", image.min(), image.max()), ax=axs[0])
    iplt.axvline(ctrls["vmin"], ax=axs[1], c="k")
    _ = iplt.axvline(ctrls["vmax"], ax=axs[1], c="k")
    return ctrls, axs
