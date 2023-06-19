import os
import pandas as pd
import cv2
import numpy as np
import shutil
import torch
from argparse import ArgumentParser
from torchvision import datasets, transforms
from scipy.io import loadmat

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


def get_jpegs_recursively(path):
    """Get the jpeg files in the given path recursively.

    Parameters
    ----------
    path : str
        Path to the folder containing the data files.

    Yields
    -------
    str
        Path to the jpeg file.
    """
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".jpeg"):
                yield os.path.join(root, file)


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
    # random shuffle
    file_paths = np.random.permutation(list(file_paths))
    train_size = int(train_size * len(file_paths))
    test_size = int(test_size * len(file_paths))
    val_size = int(val_size * len(file_paths))

    train_paths = file_paths[:train_size]
    test_paths = file_paths[train_size : train_size + test_size]
    val_paths = file_paths[train_size + test_size : train_size + test_size + val_size]

    return train_paths, test_paths, val_paths


def make_directories(path, dest_folder, reference_file, split):
    """Make the directories for the train, test and validation sets.

    Parameters
    ----------
    path : str
        Path to the folder containing the data files.
    reference_file : str
        Path to the reference file.
    """
    split = float(split)
    train_size = split
    test_size = (1 - split) / 2
    val_size = (1 - split) / 2

    os.makedirs(dest_folder, exist_ok=True)

    train_path = os.path.join(dest_folder, "train")
    test_path = os.path.join(dest_folder, "test")
    val_path = os.path.join(dest_folder, "val")

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)

    with open(reference_file, "r") as f:
        reference = f.read().splitlines()
    reference = {line.split(",")[0] + ".jpeg": line.split(",")[1] for line in reference}

    file_paths = get_file_paths(path)
    train_paths, test_paths, val_paths = train_test_val_split(
        file_paths, train_size, test_size, val_size
    )

    for file_path in train_paths:
        os.makedirs(
            os.path.join(train_path, reference[os.path.basename(file_path)]),
            exist_ok=True,
        )
        ## copy file to the new directory
        shutil.copy(
            file_path, os.path.join(train_path, reference[os.path.basename(file_path)])
        )

    for file_path in test_paths:
        os.makedirs(
            os.path.join(test_path, reference[os.path.basename(file_path)]),
            exist_ok=True,
        )
        # copy file to the new directory
        shutil.copy(
            file_path, os.path.join(test_path, reference[os.path.basename(file_path)])
        )

    for file_path in val_paths:
        os.makedirs(
            os.path.join(val_path, reference[os.path.basename(file_path)]),
            exist_ok=True,
        )
        # copy file to the new directory
        shutil.copy(
            file_path, os.path.join(val_path, reference[os.path.basename(file_path)])
        )


def data_loader(reference_file_path, path, batch_size=32):
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
    with open(reference_file_path, "r") as f:
        reference = f.read().splitlines()
    reference = {line.split(",")[0] + ".jpeg": line.split(",")[1] for line in reference}

    file_paths = get_file_paths(path)
    file_paths = np.random.permutation(list(file_paths))

    for i in range(0, len(file_paths), batch_size):
        batch = file_paths[i : i + batch_size]
        data = []
        labels = []
        for file_path in batch:
            # read image and convert to numpy array
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            image = np.array(image)
            data.append(image)
            labels.append(reference[os.path.basename(file_path)])
        yield np.array(data), np.array(labels)


class ECGLoader(torch.utils.data.Dataset):
    """ECG dataset."""

    def __init__(self, path, batch_size=32):
        """
        Args:
            reference_file_path (string): Path to the reference file.
            path (string): Path to the folder containing the data files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(ECGLoader, self).__init__()
        self.class_mapping = {
            "N": 0,
            "A": 1,
            "O": 2,
            "~": 3,
        }
        IMG_SHAPE = (256, 256, 3)

        self.transform = transforms.Compose(
            [
                transforms.Resize(IMG_SHAPE[:-1]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.batch_size = batch_size
        self.file_paths = list(get_jpegs_recursively(path))

    def __len__(self):
        return len(self.file_paths)

    def __iter__(self):
        """Return an iterator over the dataset."""
        self.file_paths = np.random.permutation(self.file_paths)
        for i in range(0, len(self.file_paths), self.batch_size):
            batch = self.file_paths[i : i + self.batch_size]
            data = []
            labels = []
            from PIL import Image

            for file_path in batch:
                image = Image.open(file_path)
                data.append(self.transform(image))
                labels.append(self.class_mapping[file_path.split("/")[-2]])
            yield torch.stack(data), torch.tensor(labels)


class ECGLoaderV2(torch.utils.data.Dataset):
    """
        ECG dataset loader for multimodal learning
    """
    
    @staticmethod
    def get_files_recursively(path, extension):
        """Get all the files recursively from the given path.

        Parameters
        ----------
        path : str
            Path to the folder containing the files.
        extension : str
            Extension of the files to get.

        Returns
        -------
        list
            List of the file paths.
        """
        file_paths = []
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(extension):
                    file_paths.append(os.path.join(root, file))
        return file_paths


    def __init__(self, image_path, signal_path, reference_file, batch_size, ):

        """
        Args:
            path (string): Path to the folder containing the data files.
            reference_file (string): Path to the reference file.
            batch_size (int): Batch size for the data loader.
        """
        super(ECGLoaderV2, self).__init__()
        self.class_mapping = {
            "N": 0,
            "A": 1,
            "O": 2,
            "~": 3,
        }
        self.image_path = image_path
        self.signal_path = signal_path

        self.batch_size = batch_size
        IMG_SHAPE = (256, 256, 3)
        self.transform = transforms.Compose(
                [
                    transforms.Resize(IMG_SHAPE[:-1]),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                    ),
                ]
            )
        
        self.reference_file = reference_file
        self.reference_df = pd.read_csv(self.reference_file, names=["file_name", "label"])

        self.image_paths = list(self.get_files_recursively(image_path, 'jpeg'))
        self.signal_paths = list(self.get_files_recursively(signal_path, 'mat'))

    def __len__(self):
        return self.reference_df.shape[0]

    def __iter__(self):
        """Return an iterator over the dataset."""
        self.reference_df = self.reference_df.sample(frac=1).reset_index(drop=True)
        for i in range(0, len(self.reference_df), self.batch_size):
            batch = self.reference_df.iloc[i : i + self.batch_size]
            data = []
            labels = []
            from PIL import Image

            for _, row in batch.iterrows():
                image_path = os.path.join(self.image_path, row["file_name"] + ".jpeg")
                signal_path = os.path.join(self.signal_path, row["file_name"] + ".mat")
                signal = torch.tensor(loadmat(signal_path)['val'][0])
                image = Image.open(image_path)
                data.append((self.transform(image), signal))
                labels.append(self.class_mapping[row['label']])
            yield data, torch.tensor(labels)

        

def data_loader_directory(path, batch_size=32):
    """
    loads data from directory structure
    """
    file_paths = get_jpegs_recursively(path)
    file_paths = np.random.permutation(list(file_paths))

    for i in range(0, len(file_paths), batch_size):
        batch = file_paths[i : i + batch_size]
        data = []
        labels = []
        for file_path in batch:
            # read image and convert to numpy array
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            image = np.array(image)
            data.append(image)
            labels.append(file_path.split("/")[-2])
        yield data, np.array(labels)


def parse_args():
    ap = ArgumentParser()
    ap.add_argument("-src", "--dataset", required=True, help="path to input dataset")
    ap.add_argument("-ref", "--reference", required=True, help="path to reference file")
    ap.add_argument("-dest", "--destination", required=True, help="path to destination")

    ap.add_argument("-s", "--split", type=float, default=0.8, help="split ratio")
    return ap.parse_args()


def main(args):
    make_directories(args.dataset, args.destination, args.reference, args.split)


if __name__ == "__main__":
    main(parse_args())
