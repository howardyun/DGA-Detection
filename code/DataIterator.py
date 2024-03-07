import pandas as pd

from DGADataset_ysx import DGATrueDataset_ysx


class DataIterator:
    """
    返回一个固定批次的数据集dataset
    """
    def __init__(self, file_path, chunksize):
        self.file_path = file_path
        self.chunksize = chunksize
        self.reader = pd.read_csv(self.file_path, chunksize=self.chunksize, iterator=True)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            # chunk = self.reader.get_chunk(self.chunksize)
            chunk = next(self.reader)
            return DGATrueDataset_ysx(chunk, True)
        except StopIteration:
            raise StopIteration
