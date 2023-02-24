import os
import cv2
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


def data_loader(reference_file_path, path, batch_size = 32):
    """Load the data from the given path and reference file.

    Parameters
    ----------
    reference_file_path : str
        Path to the reference file.
    path : str
        Path to the folder containing the data files.
    batch_size : int, optional
        Size of the batch, by default 32

    Yields
    -------
    tuple
        Tuple of the data and the labels.
    """
    with open(reference_file_path, 'r') as f:
        reference = f.read().splitlines()
    reference = {line.split(',')[0]+'.jpeg': line.split(',')[1] for line in reference}

    file_paths = get_file_paths(path)
    file_paths = np.random.permutation(list(file_paths))

    for i in range(0, len(file_paths), batch_size):
        batch = file_paths[i:i+batch_size]
        data = []
        labels = []
        for file_path in batch:
            # read image and convert to numpy array
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            image = np.array(image)
            data.append(image)
            labels.append(reference[os.path.basename(file_path)])
        yield np.array(data), np.array(labels)




