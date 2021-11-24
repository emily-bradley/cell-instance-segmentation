import numpy as np


def to_coord(index, height):
    """
    Parameters
    ----------
    index: int
        Zero-based linear index
    """
    row = index % height
    col = index // height

    return row, col


def to_img(annotation, height, width):
    starts, runs = to_runs(annotation)
    img = np.zeros((height, width))

    for start, run in zip(starts, runs):
        idxs = range(start - 1, start - 1 + run)
        img[np.unravel_index(idxs, img.shape)] = 1

    return img


def find_runs(logical_vec, offset=0):
    """
    Find the (run-length encoded) runs in the given logical vector.

    A run is a list of consecutive indices for which `logical_vec` is equal to
    True/one.

    Parameters
    ----------
    logical_vec : list of bool
        A list or one-dimensional numpy array of booleans (or zero/one values).
        No checking is done to ensure this is of the correct form.

    Returns
    -------
    runs : matrix
        A numpy array/matrix whose first column indicates the starting
        positions of runs and whose second column indicates the lengths of
        runs.

    Examples
    --------
    > v = [0, 0, 1, 1, 1, 0, 0, 1, 1, 0]
    > find_runs(v)
    numpy.array([[2, 3], [7, 2]])
    """
    # A run starts when the difference between consecutive elements is one
    # (False-to-True), and a run ends when the difference is negative one
    # (True-to-False).  We need to make sure that the first index is included
    # if it's one, so we insert a zero at the beginning.
    zero_padded = np.insert(1 * logical_vec, 0, 0)
    diffs = np.diff(zero_padded)
    starts = np.nonzero(diffs == 1)[0]
    ends = np.nonzero(diffs == -1)[0]
    lengths = ends - starts

    return np.stack([starts + offset, lengths], axis=1)


def to_runs(img: np.ndarray):
    _, width = img.shape

    row_runs = [
        find_runs(row, row_num * width + 1)
        for row_num, row in enumerate(img)
    ]
    runs = np.concatenate(row_runs)

    return runs