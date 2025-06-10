import lmdb
import os
import shutil
import pyarrow as pa
from utils import LoadConfig
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from log_tracker import Logger


class CreateLMDB:
    def __init__(self):
        """
        Initializes the CreateLMDB class by loading configuration,
        setting logger, and storing dataset paths and other parameters.
        """
        self.config = LoadConfig.load_config("configs/lmdb_config.yaml")
        self.logger = Logger.get_logger()
        self.train_dataset_path = self.config['train_dataset_path']
        self.val_dataset_path = self.config['val_dataset_path']
        self.train_output = self.config['train_output']
        self.val_output = self.config['val_output']
        self.write_freq = self.config['write_freq']
        self.num_workers = self.config['num_workers']
        self.train_map_size_gb = self.config['train_map_size_gb']
        self.val_map_size_gb = self.config['val_map_size_gb']
        self.class_map_location = self.config['class_map_location']

    def _dumps(self, obj):
        """
        Serializes a Python object using pyarrow.
        """
        return pa.serialize(obj).to_buffer()

    def _loads(self, buf):
        """
        Deserializes a pyarrow buffer back into a Python object.
        """
        return pa.deserialize(buf)

    def _read_image_bytes(self, path):
        """
        Reads an image file from disk and returns its bytes.
        """
        with open(path, "rb") as f:
            return f.read()

    def _load_readable_classes(self):
        """
        Loads a human-readable class list from a text file if available.
        """
        if not os.path.exists(self.class_map_location):
            self.logger.warn(f"Class map file not found: {self.class_map_location}")
            return None

        with open(self.class_map_location, "r") as f:
            readable_classes = [line.strip() for line in f.readlines()]
        return readable_classes

    @staticmethod
    def collate_first_element(batch):
        """
        Custom collate function for DataLoader to return only the first element of the batch.
        Used because batch_size=1.
        """
        # batch is a list of length 1 because batch_size=1, so return the first item
        return batch[0]

    def _create_lmdb(self, subset_path, lmdb_path, map_size):
        """
        Creates an LMDB database from an image dataset at a specified path.
        Stores image bytes, labels, and metadata including keys and class names.
        """
        self.logger.info(f"Loading dataset from {subset_path}...")

        dataset = ImageFolder(root=subset_path, loader=self._read_image_bytes)
        data_len = len(dataset)
        self.logger.info(f"Found {data_len} images.")

        os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)
        env = lmdb.open(lmdb_path, map_size=map_size * (1 << 30))

        keys = []
        txn = env.begin(write=True)

        data_loader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.num_workers,
            collate_fn=CreateLMDB.collate_first_element
        )

        readable_classes = self._load_readable_classes()  # load once here

        for idx, (img_bytes, label) in enumerate(data_loader):
            key = str(idx).encode("ascii")
            value = self._dumps((img_bytes, label))
            txn.put(key, value)
            keys.append(key)

            if (idx + 1) % self.write_freq == 0:
                self.logger.info(f"[{idx + 1}/{data_len}] written to LMDB...")
                txn.commit()
                txn = env.begin(write=True)

        # Store metadata after all entries written
        txn.put(b'__len__', self._dumps(data_len))
        txn.put(b'__classes__', self._dumps(dataset.classes))
        txn.put(b'__keys__', self._dumps(keys))

        if readable_classes:
            if len(dataset.classes) == len(readable_classes):
                txn.put(b'__readable_classes__', self._dumps(readable_classes))
            else:
                self.logger.info(
                    f"Readable class list has {len(readable_classes)} entries, "
                    f"but dataset has {len(dataset.classes)} classes. "
                    f"Skipping readable_classes metadata."
                )

        txn.commit()
        self.logger.info(f"LMDB dataset saved to: {lmdb_path}")

    def create_train(self):
        """
        Creates the training LMDB if it doesn't already exist or is invalid.
        """
        if self.check_lmdb_valid(self.train_output) is False:
            self._create_lmdb(self.train_dataset_path, self.train_output, self.train_map_size_gb)

    def create_val(self):
        """
        Creates the validation LMDB if it doesn't already exist or is invalid.
        """
        if self.check_lmdb_valid(self.val_output) is False:
            self._create_lmdb(self.val_dataset_path, self.val_output, self.val_map_size_gb)

    def check_lmdb_valid(self, lmdb_path):
        """
        Validates that the LMDB at the given path contains required metadata.
        Deletes and returns False if validation fails.
        Returns True if LMDB is valid.
        """
        try:
            env = lmdb.open(lmdb_path, readonly=True, lock=False)
            with env.begin() as txn:
                # Check for metadata keys
                for key in [b'__len__', b'__classes__', b'__keys__']:
                    if txn.get(key) is None:
                        self.logger.warn(f"Missing metadata key {key.decode()} in LMDB at {lmdb_path}")
                        self.logger.warn(f"LMDB path {lmdb_path} exists. Deleting...")

                        shutil.rmtree(lmdb_path)
                        return False
            self.logger.info(f"LMDB at {lmdb_path} is valid.")
            return True
        except Exception as e:
            self.logger.warn(f"Error opening LMDB at {lmdb_path}: {e}")
            return False
