import os
import numpy as np


def get_file_paths(path):
    """Get the file paths in the given path.

    Parameters
    ----------
    path : str
        Path to the folder containing the files.

    Returns
    -------
    list
        List of file paths in the given path.
    """
    yield from (os.path.join(path, file) for file in os.listdir(path))

def train_test_val_split(file_paths, train_size=0.8, test_size=0.1, val_size=0.1):
    """Split the given file paths into train, test and validation sets.

    Parameters
    ----------
    file_paths : list
        List of file paths.
    train_size : float, optional
        Size of the train set, by default 0.8
    test_size : float, optional
        Size of the test set, by default 0.1
    val_size : float, optional
        Size of the validation set, by default 0.1

    Returns
    -------
    tuple
        Tuple of lists containing the file paths for the train, test and validation sets.
    """
    #random shuffle
    file_paths = np.random.permutation(file_paths)
    train_size = int(train_size * len(file_paths))
    test_size = int(test_size * len(file_paths))
    val_size = int(val_size * len(file_paths))

    train_paths = file_paths[:train_size]
    test_paths = file_paths[train_size:train_size + test_size]
    val_paths = file_paths[train_size + test_size:train_size + test_size + val_size]

    return train_paths, test_paths, val_paths


#todo: update this function
def data_loader(file_paths, batch_size=32):
    """Load the data from the given file paths.

    Parameters
    ----------
    file_paths : list
        List of file paths.
    batch_size : int, optional
        Size of the batch, by default 32

    Yields
    -------
    tuple
        Tuple of numpy arrays containing the data and labels.
    """
    data = []
    labels = []
    for file_path in file_paths:
        data.append(np.load(file_path))
        labels.append(file_path.split('/')[-2])
    data = np.array(data)
    labels = np.array(labels)
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size], labels[i:i + batch_size]
