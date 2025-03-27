"""Custom Dataloader for Contigs CSV file"""

from torch.utils.data import Dataset


class ContigDataset(Dataset):
    """Custom Dataset for DNA Sequences

    Args:
        Dataset (Dataset): Pytorch Dataset
    """

    def __init__(self, data: list) -> None:
        """_summary_

        Args:
            csv_file (str): Filepath to contig csv
            tokenizer (object): Model Tokenizer
        """
        self.sequences = data

    def __len__(self) -> int:
        """Get length of the dataset

        Returns:
            int: N
        """
        return len(self.sequences)

    def __getitem__(self, idx: int):
        """Retrieve item from dataset

        Args:
            idx (int): index of the item

        Returns:
            str: DNA Sequence
        """
        sequence = self.sequences[idx]
        return sequence
