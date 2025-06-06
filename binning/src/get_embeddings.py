import os
import warnings

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast
from torch.amp.autocast_mode import is_autocast_available

from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

from transformers.models.bert.configuration_bert import BertConfig

from sklearn.preprocessing import normalize
from src.utils import (
    validate_input_array,
    sort_sequences,
    get_available_device,
    Logger,
    read_embeddings,
)
from src.dataset import ContigDataset


warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", message="Increasing alibi size")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class Embedder:
    def check_params():
        pass

    def __init__(
        self,
        dna_sequences: list[str],
        contig_names: list[str],
        batch_sizes: list[int],
        model_name: str,
        model_path: str,
        weight_path: str,
        save_path: str,
        normalize_embeddings: bool,
        log: Logger,
    ):

        self.dna_sequences = dna_sequences
        self.contig_names = contig_names
        self.batch_sizes = batch_sizes
        self.model_name = model_name
        self.model_path = model_path
        self.weight_path = weight_path
        self.save_path = save_path
        self.normalize_embeddings = normalize_embeddings
        self.log = log

    def get_embeddings(self) -> np.array:
        """
        Generate or load embeddings for a given set of DNA sequences using the specified model.

        This function generates embeddings using different models based on the specified `model_name`.
        If embeddings have already been computed and saved, it loads them from the provided path.
        Otherwise, it calculates new embeddings, saves them, and returns the result.
        This function calls other functions to calculate embeddings.

        model_name (str): Name of the model to use for embedding generation. Models can be found in congis/model.yml.
        model_path (str): Path to the pretrained model file or directory
        dna_sequences (list of str): List of DNA sequences for which embed required for specific models.
        save_path (str): Path to save the computed embeddings or to load existing ones.

        Returns:
        np.array: Array of embeddings with shape (num_sequences, embedding_dimension).

        """

        try:
            embeddings, _ = read_embeddings(self.save_path, self.model_name, self.log)
            return embeddings
        except FileNotFoundError:
            self.log.append(
                f"Embeddings not found at {self.save_path}, generating new ones"
            )

        os.makedirs(os.path.join(self.save_path, "embeddings"), exist_ok=True)

        if self.model_name == "tnf":
            embeddings = self.calculate_tnf()
        elif self.model_name == "tnfkernel":
            embeddings = self.calculate_tnf(use_kernel=True)
        elif self.model_name == "dna2vec":
            embeddings = self.calculate_dna2vec()

        if self.model_name in ["tnf", "tnfkernel", "dna2vec"]:
            self.log.append(f"Embeddings shape: {embeddings.shape}")
            if self.normalize_embeddings:
                embeddings = normalize(embeddings)
            save_path = os.path.join(self.save_path, "embeddings", f"embeddings.npy")
            np.savez(save_path, embeddings=embeddings, contig_names=self.contig_names)
            torch.cuda.empty_cache()
            return embeddings

        elif self.model_name in [
            "dnaberts",
            "dnaberth_400kv2",
            "dnaberth_2mv3",
            "dnaberth_2mv4",
            "dnaberth_2mv5",
            "dnaberth_400k",
            "dnaberth_2mv1",
            "dnaberth_2mv2",
            "dnaberth_2mv1_150k",
            "dnabert2",
            "dnabert2random",
        ]:
            embeddings = self.calculate_llm_embedding()
            torch.cuda.empty_cache()
            return embeddings

    def calculate_llm_embedding(self) -> np.array:
        """Get llm embeddings. Process dna sequences based on their length to increase efficiency, i.e. use a large batch size"""

        ### Model Setup ###
        self.device, self.n_gpu = get_available_device()
        self.log.append(
            f"Using device: {self.device}\nwith {self.n_gpu} GPUs\n Using mixed precision (autocast): {is_autocast_available(self.device.type)}"
        )

        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            padding_side="right",
            trust_remote_code=True,
            use_fast=True,
            padding="max_length",
        )

        config = BertConfig.from_pretrained(
            self.model_path,
        )

        if self.model_name != "dnabert2random":
            self.llm_model = AutoModel.from_pretrained(
                self.model_path,
                config=config,
                trust_remote_code=True,
            )
            if "dnaberth" in self.model_name:
                self.llm_model.load_state_dict(
                    torch.load(os.path.join(self.weight_path, "pytorch_model.bin"))
                )
                self.log.append(
                    f"Loading DNABERTH model weights from path {self.weight_path}"
                )

        elif self.model_name == "dnabert2random":
            self.llm_model = AutoModel.from_config(
                config, trust_remote_code=True
            )  # no pretrained weights

        self.llm_model = self.llm_model.to(self.device).eval()

        if self.n_gpu > 1:
            self.llm_model = nn.DataParallel(self.llm_model)

        ### Saving on memmap file to avoid OOM errors ###
        N = len(self.dna_sequences)
        D = self.llm_inference([self.dna_sequences[0]], batch_size=1).shape[1]
        self.log.append(f"Embeddings memmap shape: {N} x {D}")
        emb_path = os.path.join(self.save_path, "embeddings", f"embeddings.npy")
        mmap = np.memmap(emb_path, dtype="float32", mode="w+", shape=(N, D))

        names_path = os.path.join(
            self.save_path, "embeddings", f"contignames.npy"
        )  # save contig names seperately
        np.save(names_path, np.array(self.contig_names, dtype="<U"))

        ### Looping through the sequences ###
        min_sequence_lengths = [
            min([len(seq) for seq in self.dna_sequences]) - 1,
            10000,
            20000,
        ]
        max_sequence_lengths = [
            10000,
            20000,
            max([len(seq) for seq in self.dna_sequences]) + 1,
        ]

        for sequence_length_min, sequence_length_max, batch_size in zip(
            min_sequence_lengths, max_sequence_lengths, self.batch_sizes
        ):

            indices_filtered, dna_sequences_filtered = zip(
                *[
                    (index, seq)
                    for (index, seq) in enumerate(self.dna_sequences)
                    if (sequence_length_min <= len(seq) < sequence_length_max)
                    # and (if len(seq) < LLM_SEQ_MAX_LENGTH) #set max length to avoid OOM errors. Already handles in utils.py.
                ]
            )
            if len(dna_sequences_filtered) == 0:
                continue
            # self.log.append(
            #     f"Running {len(dna_sequences_filtered)} sequences with len between {sequence_length_min} to {sequence_length_max}"
            # )

            # Process in 5000 sequences at a time to avoid OOM errors
            n_dna_sequences_filtered_mini_processing = 5000
            for start in tqdm(
                range(
                    0,
                    len(dna_sequences_filtered),
                    n_dna_sequences_filtered_mini_processing,
                ),
                desc=f"Outer: Processing {len(dna_sequences_filtered)} contigs in mini-batches of {n_dna_sequences_filtered_mini_processing}, length: {sequence_length_min} to {sequence_length_max}",
                position=0,
            ):
                end = min(
                    start + n_dna_sequences_filtered_mini_processing,
                    len(dna_sequences_filtered),
                )
                mini_idxs = indices_filtered[start:end]
                mini_seqs = dna_sequences_filtered[start:end]

                embeddings = self.llm_inference(list(mini_seqs), batch_size)
                if self.normalize_embeddings:
                    embeddings = normalize(embeddings).astype("float32")

                for j, orig_idx in enumerate(mini_idxs):
                    mmap[orig_idx, :] = embeddings[j]

        mmap.flush()
        self.log.append(
            f"Wrote embeddings memmap to {emb_path} and names to {names_path}"
        )
        return np.memmap(emb_path, dtype="float32", mode="r", shape=(N, D))

    def collate_fn(self, batch: list[str]):
        # batch: list of raw DNA sequences
        encodings = self.llm_tokenizer(
            batch,
            padding="longest",  # pad up to longest in this batch
            truncation=True,  # truncate if too long
            return_tensors="pt",
            max_length=1000000,
        )

        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
        }

    def llm_inference(
        self,
        dna_sequences_filtered: list[str],
        batch_size: int,
        log_tokenlengths: bool = False,
    ) -> np.array:
        """Calculates embeddings for DNA sequences using a specified language model (LLM).

        This function uses a pretrained model to generate embeddings for each DNA sequence.

        Args:
            dna_sequences (list[str]): List of DNA sequences for which embeddings are generated.
            batch_size (int): Size of batches for processing sequences.
            model_name (str): Name of the model to use for generating embeddings (e.g., "DNABERT_2", "EVO", "NT", "GROVER").
            model_path (str): Path to the pretrained model files (e.g. from Huggingface)

        Returns:
            np.array: Array of embeddings for the input DNA sequences, with shape (num_sequences, embedding_dimension).
        """

        sorted_dna_sequences, idx = sort_sequences(
            dna_sequences_filtered
        )  # To reduce Padding overhead
        dna_sequences = ContigDataset(sorted_dna_sequences)

        data_loader = DataLoader(
            dna_sequences,
            batch_size=batch_size * self.n_gpu,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=2 * self.n_gpu,
        )
        if log_tokenlengths:
            all_token_lengths = []

        for i, batch in tqdm(
            enumerate(data_loader),
            desc="Inner: Running LLM Inference",
            position=1,
            leave=False,
        ):

            input_ids, attention_mask = batch["input_ids"].to(self.device), batch[
                "attention_mask"
            ].to(self.device)

            with torch.inference_mode(), torch.autocast(device_type=self.device.type):
                model_output = (
                    self.llm_model.forward(
                        input_ids=input_ids, attention_mask=attention_mask
                    )[0]
                    .detach()
                    .cpu()
                )
                attention_mask = attention_mask.unsqueeze(-1).detach().cpu()
                embedding = torch.sum(model_output * attention_mask, dim=1) / torch.sum(
                    attention_mask, dim=1
                )  # along the sequence length

                if i == 0:
                    embeddings = embedding
                else:
                    embeddings = torch.cat(
                        (embeddings, embedding), dim=0
                    )  # concatenate along the batch dimension
                if log_tokenlengths:
                    token_lengths = attention_mask.sum(dim=1).cpu().numpy()
                    all_token_lengths.extend(token_lengths)

        if log_tokenlengths:
            min_token_length = min(all_token_lengths)
            max_token_length = max(all_token_lengths)
            self.log.append(
                f"Min token length: {min_token_length}, Max token length: {max_token_length}"
            )

        embeddings = np.array(embeddings.detach().cpu())

        embeddings = embeddings[np.argsort(idx)]  # sort back to normal indices

        return embeddings

    def calculate_dna2vec(self) -> np.array:
        """
        Calculates the DNA2Vec embedding for a list of DNA sequences.

        The function then multiplies the TNF embedding with a 4-mer embedding matrix to obtain
        the DNA2Vec embedding. The 4-mer embedding matrix is pretrained embeddings on the hg38 (human genome) obtained from https://github.com/MAGICS-LAB/DNABERT_S/blob/main/evaluate/helper/4mer_embedding.npy.
        See more in paper dna2vec https://arxiv.org/abs/1701.06279.
        Args:
            dna_sequences (List[str]): A list of DNA sequences, where each sequence is a list
                                            of nucleotide characters (A, T, C, or G).
            model_path (str): Path to the model to be used for the embeddings.
        Returns:
            np.ndarray: A 2D numpy array representing the DNA2Vec embedding, where each row corresponds
                        to the embedding of a DNA sequence.
        """

        tnf_embeddings = self.calculate_tnf()

        pretrained_4mer_embedding = np.load(self.model_path)  # dim (256,100)
        embeddings = np.dot(tnf_embeddings, pretrained_4mer_embedding)

        return embeddings

    def calculate_tnf(self, use_kernel: bool = False) -> np.ndarray:
        """Calculates tetranucleotide frequencies in a list of DNA sequences.

        This function computes the frequencies of all possible tetranucleotides (sequences of four nucleotides)
        within each DNA sequence in `dna_sequences`. Corresponds to calculate 4-mers. There are 4^4=256 combinations.

        Args:
            dna_sequences (List[str]): A list of DNA sequences, where each sequence is a list of nucleotide characters (A, T, C, or G) or possibly other letters such as N.
            kernel (bool): Whether to use VAMB-kernel to downproject TNFs into 103-dimensions. We download the the pre-computed kernel from VAMB (https://github.com/RasmussenLab/vamb/blob/master/vamb/kernel.npz) and save it in helper/.
            See more at: https://github.com/RasmussenLab/vamb/blob/master/src/create_kernel.py

        Returns:
                - embeddings (np.ndarray): A 2D numpy array of shape (n, 256), where `n` is the number of DNA sequences,
                and each row contains the tetranucleotide frequency vector for a sequence.

        """
        nucleotides = ["A", "T", "C", "G"]
        tetra_nucleotides = [
            a + b + c + d
            for a in nucleotides
            for b in nucleotides
            for c in nucleotides
            for d in nucleotides
        ]

        # Build mapping from tetra-nucleotide to index
        tnf_index = {tn: i for i, tn in enumerate(tetra_nucleotides)}

        # Build embeddings by counting TNFs
        embeddings = np.zeros((len(self.dna_sequences), len(tetra_nucleotides)))

        for j, seq in tqdm(
            enumerate(self.dna_sequences),
            total=len(embeddings),
            desc="Calculating TNFs",
        ):
            for i in range(len(seq) - 3):
                try:
                    tetra_nuc = seq[i : i + 4]
                    embeddings[j, tnf_index[tetra_nuc]] += 1
                except KeyError:  # there exist nucleotide N which will give error
                    continue

            if embeddings[j, :].sum() == 0:
                raise ValueError(
                    f"TNF value of contig at index {j} is all zeros. "
                    + "This implies that the sequence contained no 4-mers of A, C, G, T or U, "
                    + "making this sequence uninformative. This is probably a mistake. "
                    + "Verify that the sequence contains usable information (e.g. is not all N's)"
                )

        total_counts = np.sum(embeddings, axis=1)
        total_counts[total_counts == 0] = 1e-10  # avoid division by 0
        embeddings = embeddings / total_counts[:, None]

        if use_kernel:
            kernel_raw = np.load(self.model_path)
            kernel = validate_input_array(kernel_raw["arr_0"])

            embeddings += -(1 / 256)
            embeddings = np.dot(embeddings, kernel)

        return embeddings
