# %% [code] {"execution":{"iopub.status.busy":"2021-11-18T22:00:07.627315Z","iopub.execute_input":"2021-11-18T22:00:07.627957Z","iopub.status.idle":"2021-11-18T22:00:07.632485Z","shell.execute_reply.started":"2021-11-18T22:00:07.627918Z","shell.execute_reply":"2021-11-18T22:00:07.631825Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"execution":{"iopub.status.busy":"2021-11-18T22:00:34.118378Z","iopub.execute_input":"2021-11-18T22:00:34.118643Z","iopub.status.idle":"2021-11-18T22:00:34.123715Z","shell.execute_reply.started":"2021-11-18T22:00:34.118616Z","shell.execute_reply":"2021-11-18T22:00:34.122576Z"}}
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# %% [code] {"execution":{"iopub.status.busy":"2021-11-18T22:21:46.888831Z","iopub.execute_input":"2021-11-18T22:21:46.889797Z","iopub.status.idle":"2021-11-18T22:21:46.895172Z","shell.execute_reply.started":"2021-11-18T22:21:46.889751Z","shell.execute_reply":"2021-11-18T22:21:46.894119Z"}}
matplotlib.rcParams['figure.figsize'] = (20, 10)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-18T22:09:19.014961Z","iopub.execute_input":"2021-11-18T22:09:19.015735Z","iopub.status.idle":"2021-11-18T22:09:19.024812Z","shell.execute_reply.started":"2021-11-18T22:09:19.015687Z","shell.execute_reply":"2021-11-18T22:09:19.023549Z"}}
def to_runs(annotation):
    encoded_ints = [int(s) for s in encoding.split(" ")]
    starts = encoded_ints[::2]
    runs = encoded_ints[1::2]
    
    return starts, runs


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


def find_runs(idxs, offset=0):
    """
    Parameters
    ----------
    idxs : list of int

    Examples
    --------
    > idxs = [2, 3, 4, 7, 8]
    > find_runs(idxs)
    numpy.array([[2, 3], [7, 2]])
    """
    return np.array([[1, 2], [4, 5]])


def to_runs(img: np.ndarray):
    height, width = img.shape

    row_runs = [
        find_runs(row.nonzero()[0], row_num * width + 1)
        for row_num, row in enumerate(img)
    ]
    runs = np.concatenate(row_runs)

    return runs


# %% [code] {"execution":{"iopub.status.busy":"2021-11-18T22:10:11.359820Z","iopub.execute_input":"2021-11-18T22:10:11.360113Z","iopub.status.idle":"2021-11-18T22:10:11.745870Z","shell.execute_reply.started":"2021-11-18T22:10:11.360081Z","shell.execute_reply":"2021-11-18T22:10:11.744972Z"}}
train_df = pd.read_csv('/kaggle/input/../input/sartorius-cell-instance-segmentation/train.csv')
train_df.head(10)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-18T22:01:34.548331Z","iopub.execute_input":"2021-11-18T22:01:34.548627Z","iopub.status.idle":"2021-11-18T22:01:34.596475Z","shell.execute_reply.started":"2021-11-18T22:01:34.548591Z","shell.execute_reply":"2021-11-18T22:01:34.595685Z"}}
train_df.where(train_df['id']=='0030fd0e6378')['annotation']

# %% [code] {"execution":{"iopub.status.busy":"2021-11-18T22:01:36.247688Z","iopub.execute_input":"2021-11-18T22:01:36.248347Z","iopub.status.idle":"2021-11-18T22:01:36.265099Z","shell.execute_reply.started":"2021-11-18T22:01:36.248299Z","shell.execute_reply":"2021-11-18T22:01:36.264247Z"}}
img = mpimg.imread('/kaggle/input/../input/sartorius-cell-instance-segmentation/train/0030fd0e6378.png')

# %% [code] {"execution":{"iopub.status.busy":"2021-11-18T22:01:38.455848Z","iopub.execute_input":"2021-11-18T22:01:38.456171Z","iopub.status.idle":"2021-11-18T22:01:39.097745Z","shell.execute_reply.started":"2021-11-18T22:01:38.456140Z","shell.execute_reply":"2021-11-18T22:01:39.093146Z"}}
imgplot = plt.imshow(img)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-18T22:19:53.857820Z","iopub.execute_input":"2021-11-18T22:19:53.858141Z","iopub.status.idle":"2021-11-18T22:19:53.877262Z","shell.execute_reply.started":"2021-11-18T22:19:53.858105Z","shell.execute_reply":"2021-11-18T22:19:53.876429Z"}}
type(np.where(train_df['id'] == '0030fd0e6378')[0].tolist())

# %% [code] {"execution":{"iopub.status.busy":"2021-11-18T22:21:31.875747Z","iopub.execute_input":"2021-11-18T22:21:31.876017Z","iopub.status.idle":"2021-11-18T22:21:32.300979Z","shell.execute_reply.started":"2021-11-18T22:21:31.875987Z","shell.execute_reply":"2021-11-18T22:21:32.300011Z"}}
height = train_df.loc[0, "height"]
width = train_df.loc[0, "width"]
cell_img = np.zeros((width, height))
idxs = np.where(train_df['id'] == '0030fd0e6378')[0].tolist()

for idx in idxs:
    encoding = train_df.loc[idx, "annotation"]
    cell_img += to_img(encoding, height, width)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-18T22:21:52.847704Z","iopub.execute_input":"2021-11-18T22:21:52.848288Z","iopub.status.idle":"2021-11-18T22:21:53.235975Z","shell.execute_reply.started":"2021-11-18T22:21:52.848248Z","shell.execute_reply":"2021-11-18T22:21:53.235152Z"}}
plt.imshow(cell_img.T > 0)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-18T22:12:57.756525Z","iopub.execute_input":"2021-11-18T22:12:57.757267Z","iopub.status.idle":"2021-11-18T22:12:57.763025Z","shell.execute_reply.started":"2021-11-18T22:12:57.757227Z","shell.execute_reply":"2021-11-18T22:12:57.762118Z"}}
img.shape

# %% [code] {"execution":{"iopub.status.busy":"2021-11-18T22:22:02.670680Z","iopub.execute_input":"2021-11-18T22:22:02.671217Z","iopub.status.idle":"2021-11-18T22:22:03.270645Z","shell.execute_reply.started":"2021-11-18T22:22:02.671182Z","shell.execute_reply":"2021-11-18T22:22:03.269989Z"}}
plt.imshow(cell_img.T > 0, alpha=1)
# plt.imshow(np.flipud(cell_img.T > 0), alpha=1)
plt.imshow(img, alpha=0.9, cmap="gray")
plt.show()

# %% [code]

