import lmdb
import pyarrow as pa
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from utils import LoadConfig
from log_tracker import Logger
import io
import random


class LMDBImageNetDataset(Dataset):
    def __init__(self, lmdb_path, transform=None, subset_keys=None):
        """
                Initializes the LMDBImageNetDataset.

                Args:
                    lmdb_path (str): Path to the LMDB database.
                    transform (callable, optional): Transformations to apply to the images.
                    subset_keys (list, optional): List of keys to use as a subset of the dataset.
        """
        self.lmdb_path = lmdb_path
        self.transform = transform

        self.subset_keys = subset_keys
        self.env = None
        self.keys = None
        self.length = None
        self.classes = None
        self.readable_classes = None

    def __getstate__(self):
        """
            Ensures that the LMDB environment is not serialized during pickling.
        """
        state = self.__dict__.copy()
        state['env'] = None
        return state

    def __setstate__(self, state):
        """
            Restores the object state after unpickling.
        """
        self.__dict__.update(state)

    @staticmethod
    def sample_keys(lmdb_path, fraction, seed):
        """
            Randomly samples a fraction of keys from the LMDB database.

            Args:
                lmdb_path (str): Path to the LMDB database.
                fraction (float): Fraction of keys to sample (0 < fraction <= 1).
                seed (int): Random seed for reproducibility.

            Returns:
                list: A list of randomly sampled keys.
        """
        env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with env.begin() as txn:
            all_keys = pa.deserialize(txn.get(b'__keys__'))
        random.seed(seed)
        sample_size = max(1, int(len(all_keys) * fraction))
        return random.sample(all_keys, sample_size)

    def _initialize_lmdb(self):
        """
        Initializes the LMDB environment and loads keys, class labels, and readable class names.
        """
        if self.env is not None:
            return
        self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False)
        with self.env.begin() as txn:
            full_keys = pa.deserialize(txn.get(b'__keys__'))
            self.keys = self.subset_keys if self.subset_keys is not None else full_keys
            self.length = len(self.keys)
            self.classes = pa.deserialize(txn.get(b'__classes__'))

            # Load readable classes if available
            readable_classes_bytes = txn.get(b'__readable_classes__')
            if readable_classes_bytes is not None:
                self.readable_classes = pa.deserialize(readable_classes_bytes)
            else:
                self.readable_classes = None

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        if self.length is None:
            self._initialize_lmdb()
        return self.length

    def __getitem__(self, index):
        """
        Retrieves the image and label at the specified index.

        Args:
            index (int): Index of the data item.

        Returns:
            tuple: A tuple containing the image and its corresponding label.
        """
        if self.env is None:
            self._initialize_lmdb()

        key = self.keys[index]
        with self.env.begin() as txn:
            value = txn.get(key)
        img_bytes, label = pa.deserialize(value)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label

    def get_class_name(self, label):
        """
        Returns the human-readable class name for a given label.

        Args:
            label (int): Class label index.

        Returns:
            str: Human-readable class name.
        """
        if self.readable_classes and 0 <= label < len(self.readable_classes):
            return self.readable_classes[label]
        elif self.classes and 0 <= label < len(self.classes):
            return self.classes[label]
        else:
            return str(label)

    @staticmethod
    def create_transform(is_train, img_height, img_width):
        """
        Creates a data transformation pipeline.

        Args:
            is_train (bool): Whether the transform is for training or evaluation.
            img_height (int): Height of the input image.
            img_width (int): Width of the input image.

        Returns:
            callable: A transform function.
        """
        return create_transform(
            input_size=(3, img_height, img_width),
            is_training=is_train,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD
        )


class LMDBImageNetDataLoader:
    def __init__(self):
        """
        Initializes the LMDBImageNetDataLoader with configuration settings.
        """
        self.config = LoadConfig.load_config("configs/data_loader_config.yaml")
        self.logger = Logger.get_logger()

        self.train_lmdb_path = self.config['train_loader']['train_lmdb_path']
        self.val_lmdb_path = self.config['val_loader']['val_lmdb_path']
        self.train_batch_size = self.config['train_loader']['train_batch_size']
        self.val_batch_size = self.config['val_loader']['val_batch_size']
        self.train_num_workers = self.config['train_loader']['train_num_workers']
        self.val_num_workers = self.config['val_loader']['val_num_workers']
        self.transform_img_height = self.config['transforms']['img_height']
        self.transform_img_width = self.config['transforms']['img_width']
        self.train_subset_fraction = self.config['train_loader']['train_subset_fraction']
        self.val_subset_fraction = self.config['val_loader']['val_subset_fraction']

    def create_train_loader(self):
        """
        Creates and returns a DataLoader for the full training dataset.

        Returns:
            DataLoader: PyTorch DataLoader for the training dataset.
        """
        transform = LMDBImageNetDataset.create_transform(
            is_train=True,
            img_height=self.transform_img_height,
            img_width=self.transform_img_width
        )
        dataset = LMDBImageNetDataset(self.train_lmdb_path, transform)
        loader = DataLoader(
            dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.train_num_workers,
            pin_memory=True
        )
        self.logger.info(f"Train DataLoader created with {len(dataset)} samples.")
        return loader

    def create_val_loader(self):
        """
        Creates and returns a DataLoader for the full validation dataset.

        Returns:
            DataLoader: PyTorch DataLoader for the validation dataset.
        """
        transform = LMDBImageNetDataset.create_transform(
            is_train=False,
            img_height=self.transform_img_height,
            img_width=self.transform_img_width
        )
        dataset = LMDBImageNetDataset(self.val_lmdb_path, transform)
        loader = DataLoader(
            dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.val_num_workers,
            pin_memory=True
        )
        self.logger.info(f"Validation DataLoader created with {len(dataset)} samples.")
        return loader

    def create_subset_train_loader(self):
        """
        Creates and returns a DataLoader for a random subset of the training dataset.

        Returns:
            DataLoader: PyTorch DataLoader for the subset of the training dataset.
        """
        if not self.train_subset_fraction:
            self.logger.warn("train_subset_fraction is 0. Falling back to full train loader.")
            return self.create_train_loader()
        subset_keys = LMDBImageNetDataset.sample_keys(self.train_lmdb_path, self.train_subset_fraction, seed=42)
        transform = LMDBImageNetDataset.create_transform(
            is_train=True,
            img_height=self.transform_img_height,
            img_width=self.transform_img_width
        )
        dataset = LMDBImageNetDataset(self.train_lmdb_path, transform, subset_keys=subset_keys)
        loader = DataLoader(
            dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.train_num_workers,
            pin_memory=True
        )
        self.logger.info(f"Subset Train DataLoader created with {len(dataset)} samples "
                         f"({self.train_subset_fraction * 100:.1f}% of full dataset).")
        return loader

    def create_subset_val_loader(self):
        """
        Creates and returns a DataLoader for a random subset of the validation dataset.

        Returns:
            DataLoader: PyTorch DataLoader for the subset of the validation dataset.
        """
        if not self.val_subset_fraction:
            self.logger.warn("val_subset_fraction is 0. Falling back to full val loader.")
            return self.create_val_loader()
        subset_keys = LMDBImageNetDataset.sample_keys(self.val_lmdb_path, self.val_subset_fraction, seed=42)
        transform = LMDBImageNetDataset.create_transform(
            is_train=False,
            img_height=self.transform_img_height,
            img_width=self.transform_img_width
        )
        dataset = LMDBImageNetDataset(self.val_lmdb_path, transform, subset_keys=subset_keys)
        loader = DataLoader(
            dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.val_num_workers,
            pin_memory=True
        )
        self.logger.info(f"Subset Validation DataLoader created with {len(dataset)} samples "
                         f"({self.val_subset_fraction * 100:.1f}% of full dataset).")
        return loader
