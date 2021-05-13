__all__ = ["interactive_threshold"]

import matplotlib.pyplot as plt
import mpl_interactions.ipyplot as iplt
import numpy as np
import scipy.ndimage as ndi
import scipy
from scipy.optimize import linear_sum_assignment
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

try:
    import napari
except ImportError:
    pass


def interactive_threshold(image, bins="auto", figsize=(12, 5)):
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


def individualize_single_frame(mask, min_distance=10):
    """
    Perform a watershed segmentation on a single frame.

    Parameters
    ----------
    mask : (M, N) arraylike of bool

    Returns
    -------
    labels : (M, N) array
    peaks : (M, N) array
    """
    distance = ndi.distance_transform_edt(mask)
    coords = peak_local_max(distance, min_distance=min_distance)
    peaks = np.zeros(distance.shape, dtype=bool)
    peaks[tuple(coords.T)] = True
    markers, _ = ndi.label(peaks)
    labels = watershed(-distance, markers, mask=mask, connectivity=2)
    return labels, peaks

def get_segment_cmap(labels, alpha=.75):
    """
    Create a colormap that useful for overlaying segmentation labels onto an image.
    The 0 label will be transparent.

    Parameters
    ----------
    labels : int, or (M, N) array of int
        The number of unique labels, or the array of labels from which the number of colors will
        be automatically determined.
    alpha : float, default: 0.75
        The alpha value of the colormap

    Returns
    -------
    matplotlib.colormap
    """
    from matplotlib.colors import LinearSegmentedColormap
    if isinstance(labels, int):
        N = labels
    else:
        N = len(np.unique(labels))
    colors = np.random.rand(N, 4)
    colors[:,-1] = alpha
    colors[0,-1] = 0
    cmap = LinearSegmentedColormap.from_list('segment_cmap', colors, N=N)
    return cmap

###############################
# The rest of this file is a slightly modified version of:
# https://github.com/Hekstra-Lab/microutil/blob/4683c72644dc51c6e6da2608db97aa58f10028c9/microutil/track_utils.py


def construct_cost_matrix(
    prev,
    curr,
    weights=[1, 1, 1 / 20],
    pad=1e4,
    debug_info="",
    normalize=False,
    distance_cutoff=0.5,
    compute_overlap=True,
):
    """
    prev : (X, Y) array of int
        The previous time point's labels
    curr : (X, Y) array of int
        The current time point's labels.
    weights : (3,) array-like, default: [1, 1, 1/5]
        The weighting of features to use for the minkowski distance.
        The current order is [X, Y, area]
    pad : number or None, default: 1e4
        The value to use when padding the cost matrix to be square. Set to *None*
        to not pad.
    normalize : bool, default: False
        Whether to normalize the each frames features. Optional as it sometimes seems
        to cause issues.
    distance_cutoff : float, default: .5
        Float between [0,1] the maximum distance relative to the frame size
        for which cells can be considered to be tracked. Cell pairs with a distance
        geater than the computed maximum will be given an entry into the cost matrix of
        1e6
    compute_overlap : bool, default: True
        Whether to to weight the assignments by how much the cells overlap between the
        two timesteps. This may be a slow step.
    Returns
    -------
    C : (N, N) array
        The cost matrix. Where *N* is the larger of the number of cells of the two
        time points
    M : int
        The number of cells in the previous time point.
    """
    prev_features = frame_to_features(prev)
    curr_features = frame_to_features(curr)
    min_areas = np.minimum.outer(prev_features[:, -1], curr_features[:, -1])
    xy_dist = scipy.spatial.distance.cdist(prev_features[:, :2], curr_features[:, :2])
    if normalize:
        norm(prev_features, curr_features)

    C = scipy.spatial.distance.cdist(
        prev_features, curr_features, metric="minkowski", w=weights
    )

    max_dist = np.sqrt(prev.shape[0] ** 2 + prev.shape[1] ** 2)
    too_far_idx = xy_dist > distance_cutoff * max_dist
    C[too_far_idx] = 1e6

    # figure out if masks overlap and make those ones more likely
    if compute_overlap:
        overlaps = overlap(prev, curr, (C.shape[0] + 1, C.shape[1] + 1))[1:, 1:].astype(
            float
        )
        overlaps /= min_areas
        C *= 1 - overlaps

    if np.any(np.isnan(C)):
        print(prev_features)
        print(curr_features)
        print(C)

    M, N = C.shape
    if pad is not None:
        if M < N:
            # maybe these should be low cost connections?
            row_pad = N - M
            C = np.pad(C, ((0, row_pad), (0, 0)), constant_values=pad)
        elif M > N:
            print("Fewer cells in the current frame than the last frame")
            print("for best results run `correct_watershed` again")
            print(debug_info + f" - {M} {N}")
        return C, M
    return C, M


def _process_seeds(seeds, idxs=None):
    if idxs is None:
        Ss, Ts, Ys, Xs = np.nonzero(seeds)
    else:
        idxs = np.asarray(idxs).astype(np.int)
        Ss, Ts = idxs.T
        Ys, Xs = seeds[:, -2], seeds[:, -1]
    # get the maximum number of cells in any frame so we know what to pad to
    # probs could make this part speedier
    max_N = 0
    for s in np.unique(Ss):
        N = np.unique(Ts[Ss == s], return_counts=True)[1].max()
        if N > max_N:
            max_N = N

    T = Ts.max() + 1
    S = Ss.max() + 1
    _seeds = np.zeros([S, T, max_N, 2], np.float32)
    _seeds[...] = np.nan
    for s in range(S):
        s_idx = Ss == s
        for t in range(T):
            t_idx = Ts[s_idx] == t

            _seeds[s, t, : np.sum(t_idx)] = np.hstack(
                [Ys[s_idx][t_idx][:, None], Xs[s_idx][t_idx][:, None]]
            )
    return _seeds


def peak_mask_to_napari_points(peak_mask):
    """
    Convert a peak mask array into the points format that napari expects
    Parameters
    ----------
    peak_mask : (S, T, Y, X) array of bool
    Returns
    -------
    points : (N, 4) array of int
    """
    points = _process_seeds(peak_mask)
    s = points.shape[:-1]
    N = np.cumprod(s)[-1]
    points_transformed = np.hstack(
        [
            a.ravel()[:, None]
            for a in np.meshgrid(
                *[np.arange(s) for s in points.shape[:-1]], indexing="ij"
            )
        ]
        + [points.reshape((N, 2))]
    )[:, [0, 1, 3, 4]]
    return points_transformed[~np.isnan(points_transformed).any(axis=1)]


def napari_points_to_peak_mask(points, shape, S, T):
    """
    Parameters
    ----------
    points : (N, d) array
        The *data* attribute of a napari points layer
    shape : tuple
        The shape of the output mask
    S, T : int
    Returns
    -------
    peak_mask : array of bool
    """
    new_seeds = _process_seeds(points[:, -2:], points[:, :2])[S, T]
    new_seeds = new_seeds[~np.any(np.isnan(new_seeds), axis=1)]
    peak_mask = np.zeros(shape, dtype=bool)
    peak_mask[tuple(new_seeds.astype(int).T)] = True
    return peak_mask


def watershed_single_frame_preseeded(mask, peak_mask):
    """
    Perform a watershed on a single frame of a dataset. This will
    not populate the watershed labels. They must already exist.
    You probably don't want to use this function when scripting. This is primarily
    provided for usage inside of correct_watershed.
    Parameters
    ----------
    ds : Dataset
    S, T : int
    """
    print(mask.shape)
    print(peak_mask.shape)
    topology = -ndi.distance_transform_edt(mask)
    return watershed(topology, ndi.label(peak_mask)[0], mask=mask)



def scroll_time(viewer, time_axis=1):
    def scroll_callback(layer, event):
        if "Shift" in event.modifiers:
            new = list(viewer.dims.current_step)

            # get the max time
            max_time = viewer.dims.range[time_axis][1]

            # event.delta is (float, float) for horizontal and vertical scroll
            # on linux shift-scroll gives vertical
            # but on mac it gives horizontal. So just take the max and hope
            # for the best
            if max(event.delta) > 0:
                if new[time_axis] < max_time:
                    new[time_axis] += 1
            else:
                if new[time_axis] > 0:
                    new[time_axis] -= 1
            viewer.dims.current_step = new

    viewer.mouse_wheel_callbacks.append(scroll_callback)


def apply_label_keybinds(labels):
    @labels.bind_key("q")
    def paint_mode(viewer):
        labels.mode = "erase"

    @labels.bind_key("w")
    def paint_mode(viewer):
        labels.mode = "fill"

    @labels.bind_key("s")
    def paint_mode(viewer):
        labels.selected_label = 0
        labels.mode = "fill"

    @labels.bind_key("e")
    def paint_mode(viewer):
        labels.mode = "paint"

    @labels.bind_key("r")
    def paint_mode(viewer):
        labels.mode = "pick"

    @labels.bind_key("t")
    def new_cell(viewer):
        labels.selected_label = labels.data.max() + 1

    # scrolling in paint mode changes the brush size
    # shift-scroll changes the time point
    time_axis = 1

    def scroll_callback(layer, event):
        if len(event.modifiers) == 0 and labels.mode in ["paint", "erase"]:
            if event.delta[1] > 0:
                labels.brush_size += 1
            else:
                labels.brush_size = max(labels.brush_size - 1, 1)

    labels.mouse_wheel_callbacks.append(scroll_callback)


def apply_points_keybinds(points):
    @points.bind_key("q")
    def remove_selected(layer):
        points.remove_selected()

    @points.bind_key("w")
    def add_mode(layer):
        points.mode = "add"

    @points.bind_key("e")
    def select_mode(layer):
        points.mode = "select"


def correct_watershed(images, labels_arr, peak_mask):
    """
    Manually correct parts of an image with a bad watershed.
    This will modify the 'peak_mask' and 'labels' variables of the Dataset inplace.
    Keybindings:
    2 : toggle between mask and labels
    3 : toggle between controlling mask/labels or points
    4 : Toggle visibility of mask+labels on and off
    5 : Toggle whether painting paints through all time points
    Control-l : rereun current frame's watershed
    Control-Shift-l : rereun watershed for all the frames
    Shift + Scroll : scrub through time points
    Point Layer Keybindings:
    q : delete selected points
    w : Switch to `Add Points` mode
    e : Switch to `Select` mode
    Mask/Labels layer Keybindngs:
    q : erase
    w : fill
    e : paint
    r : pick
    t : create new label
    s : fill with background
    Scroll : modify brush size when in paint mode
    Parameters
    ----------
    ds : (S, T, ... , Y, X) xarray dataset
    """

    viewer = napari.view_image(images)
    labels = viewer.add_labels(labels_arr, name="labels", visible=True)
    peak_coords = peak_mask_to_napari_points(peak_mask[None, :, ...])
    points = viewer.add_points(peak_coords, size=5, name="watershed seeds")
    apply_label_keybinds(labels)
    scroll_time(viewer)
    apply_points_keybinds(points)

    through_time = False

    def setup_through_time(layer):
        old_paint = layer.paint
        old_fill = layer.fill

        def paint_through_time(coord, new_label, refresh=True):
            if through_time:
                for i in range(labels.data.shape[1]):
                    c = list(coord)
                    c[1] = i
                    old_paint(c, new_label, refresh)
            else:
                old_paint(coord, new_label, refresh)

        layer.paint = paint_through_time

    setup_through_time(labels)

    def toggle_masks(*args):
        if mask.visible and labels.visible:
            # ugh - I guess set the labels to visible
            labels_and_points()
        elif mask.visible:
            labels_and_points()
        else:
            mask_and_points()
        if viewer.active_layer in [mask, labels]:
            set_correct_active_labels()

    def mask_and_points(*args):
        mask.visible = True
        labels.visible = False

    def labels_and_points(*args):
        mask.visible = False
        labels.visible = True

    def toggle_points_vs_labels(viewer):
        if viewer.active_layer == points:
            set_correct_active_labels()
        else:
            viewer.layers.unselect_all()
            points.selected = True

    def gogogo(viewer):
        T = viewer.dims.current_step[:1]
        peak_mask[T] = napari_points_to_peak_mask(points.data, images.shape[1:], 0, T)
        labels_arr[T] = watershed_single_frame_preseeded(
            labels.data[T] != 0, peak_mask[T]
        )
        labels.data = labels_arr

    def gogogo_all(viewer):
        for T in range(images.shape[0]):
            peak_mask[T] = napari_points_to_peak_mask(
                points.data, images.shape[1:], 0, T
            )
            labels_arr[T] = watershed_single_frame_preseeded(
                labels.data[T] != 0, peak_mask[T]
            )
            labels.data = labels_arr

    def toggle_through_time(*args):
        nonlocal through_time
        through_time = not through_time

    viewer.bind_key("2", toggle_masks)
    viewer.bind_key("3", toggle_points_vs_labels)
    viewer.bind_key("5", toggle_through_time)
    viewer.bind_key("Control-l", gogogo)
    viewer.bind_key("Control-Shift-l", gogogo_all)
    return labels




def _py_overlap(prev, curr, shape):
    p_uniq = np.unique(prev)
    c_uniq = np.unique(curr)
    arr = np.zeros(shape)
    for i in p_uniq:
        for j in c_uniq:
            arr[i, j] = np.sum((prev == i) * (curr == j))
    return arr


def overlap(prev, curr, shape):
    """
    Calculate the pairwise overlap the labels for two arrays.
    This will use `fast_overlap` if that library is available.
    Parameters
    ----------
    prev, curr : 2D array-like of int
        curr will have at least as many unique labels as prev
    shape : tuple of int
        The shape of the output array. This should reflect the maximum
        value of labels.
    Returns
    -------
    arr : (N, M) array of int
        N is the number of unique labels in prev and M the number of unique in curr.
        The ijth entry in the array gives the number of pixels for which label *i* in prev
        overlaps with *j* in curr.
    """

    # figure out if masks overlap and make those ones more likely
#     try:
#         from fast_overlap import overlap

#         return np.asarray(overlap(prev, curr, shape))
#     except ImportError:
    return _py_overlap(prev, curr, shape)


def norm(prev, curr):
    for col in range(3):
        r = np.hstack([prev[:, col], curr[:, col]])
        m = np.mean(r)
        s = np.std(r)
        prev[:, col] = (prev[:, col] - m) / s
        curr[:, col] = (curr[:, col] - m) / s


def frame_to_features(frame):
    """
    Parameters
    ----------
    frame : (X, Y) array-like
        The mask of labels to create features from
    Returns
    -------
    features : (N, 3)
        The features array of (com_x, com_y, area)
    """
    labels, areas = np.unique(frame, return_counts=True)
    com = np.asarray(ndi.center_of_mass(frame, frame, labels[1:]))
    return np.hstack([com, areas[1:, None]])


def construct_cost_matrix(
    prev,
    curr,
    weights=[1, 1, 1 / 20],
    pad=1e4,
    debug_info="",
    normalize=False,
    distance_cutoff=0.5,
    compute_overlap=True,
):
    """
    prev : (X, Y) array of int
        The previous time point's labels
    curr : (X, Y) array of int
        The current time point's labels.
    weights : (3,) array-like, default: [1, 1, 1/5]
        The weighting of features to use for the minkowski distance.
        The current order is [X, Y, area]
    pad : number or None, default: 1e4
        The value to use when padding the cost matrix to be square. Set to *None*
        to not pad.
    normalize : bool, default: False
        Whether to normalize the each frames features. Optional as it sometimes seems
        to cause issues.
    distance_cutoff : float, default: .5
        Float between [0,1] the maximum distance relative to the frame size
        for which cells can be considered to be tracked. Cell pairs with a distance
        geater than the computed maximum will be given an entry into the cost matrix of
        1e6
    compute_overlap : bool, default: True
        Whether to to weight the assignments by how much the cells overlap between the
        two timesteps. This may be a slow step.
    Returns
    -------
    C : (N, N) array
        The cost matrix. Where *N* is the larger of the number of cells of the two
        time points
    M : int
        The number of cells in the previous time point.
    """
    prev_features = frame_to_features(prev)
    curr_features = frame_to_features(curr)
    min_areas = np.minimum.outer(prev_features[:, -1], curr_features[:, -1])
    xy_dist = scipy.spatial.distance.cdist(prev_features[:, :2], curr_features[:, :2])
    if normalize:
        norm(prev_features, curr_features)

    C = scipy.spatial.distance.cdist(
        prev_features, curr_features, metric="minkowski", w=weights
    )

    max_dist = np.sqrt(prev.shape[0] ** 2 + prev.shape[1] ** 2)
    too_far_idx = xy_dist > distance_cutoff * max_dist
    C[too_far_idx] = 1e6

    # figure out if masks overlap and make those ones more likely
    if compute_overlap:
        overlaps = overlap(prev, curr, (C.shape[0] + 1, C.shape[1] + 1))[1:, 1:].astype(
            float
        )
        overlaps /= min_areas
        C *= 1 - overlaps

    if np.any(np.isnan(C)):
        print(prev_features)
        print(curr_features)
        print(C)

    M, N = C.shape
    if pad is not None:
        if M < N:
            # maybe these should be low cost connections?
            row_pad = N - M
            C = np.pad(C, ((0, row_pad), (0, 0)), constant_values=pad)
        elif M > N:
            print("More cells in the current frame than the previous frame")
            print('for best results rerun correct_watershed and fix this')
            print(debug_info + f" - {M} {N}")
        return C, M
    return C, M


def track(arr, weights=[1, 1, 1 / 5], pad=1e4):
    """
    Track cells over a time lapse.

    Parameters
    ----------
    arr : (T, M, N) array-like
        The mask of cell labels. Should be integers.
    weights : (3,) array-like, default: [1, 1, 1/5]
        The weighting of features to use for the minkowski distance.
        The current order is [X, Y, area]
    pad : number or None, default: 1e4
        The value to use when padding the cost matrix to be square. Set to *None*
        to not pad.

    Returns
    -------
    tracked : (T, M, N) array
        The labels mask with cells tracked through time. So ideally
        a cell labelled 4 at t0 will also be labelled 4 at t1.
    """
    tracked = np.zeros_like(arr)
    tracked[0] = arr[0]

    for t in range(1, len(arr)):
        C, M = construct_cost_matrix(
            tracked[t - 1], arr[t], weights=weights, debug_info=f"t={t}"
        )
        row_ind, col_ind = linear_sum_assignment(C)
        assignments = np.stack([row_ind, col_ind], axis=1)

        for i in range(len(assignments)):
            prev, curr = assignments[i]
            idx = arr[t] == curr + 1
            tracked[t][idx] = prev + 1
    return tracked
