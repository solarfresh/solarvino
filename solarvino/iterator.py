import cv2
import os
import numpy as np
from typing import Tuple


class BaseIterator:
    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError


class DirectoryIterator(BaseIterator):
    def __init__(self, directory, extension: Tuple = ('jpg', 'png', 'bmp', 'jpeg'), shuffle=True):
        self.file_path_list = [filepath for filepath in self._file_path_generator(directory, extension)]
        self.item_count = len(self.file_path_list)
        self.item_index = -1
        self.index_array = np.arange(self.item_count)
        if shuffle:
            np.random.shuffle(self.index_array)

    def __next__(self):
        self.item_index += 1
        if self.item_index < self.item_count:
            file_id = self.index_array[self.item_index]
            file_path = self.file_path_list[file_id]
            return file_id, file_path

        raise StopIteration

    @staticmethod
    def _file_path_generator(directory, extension):
        for root, _, fnames in os.walk(directory):
            for fname in fnames:
                if not fname.lower().endswith(extension):
                    continue

                yield os.path.join(root, fname)
