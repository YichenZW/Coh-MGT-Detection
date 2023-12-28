import torch
from torch.utils.data import Dataset
from easydl import clear_output
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from sentence_transformers import SentenceTransformer, util
import numpy as np
import networkx as nx


class VGraph(object):
    def __init__(self, g):
        """
        g: a networkx graph
        label: an integer graph label
        node_tags: a list of integer node tags
        node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
        edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
        neighbors: list of neighbors (without self-loop)
        """
        self.g = g
        self.edge_mat = 0
        self.max_neighbor = 0


class sentsplitter(SpacySentenceSplitter):
    def my_splitter(self, text: str):
        if self._is_version_3:
            return [
                sent.text.strip()
                for sent in self.spacy(text).sents
                if sent.text != "\n"
            ]
        else:
            return [
                sent.string.strip()
                for sent in self.spacy(text).sents
                if sent.text != "\n "
            ]


def cal_sim_matrix(text, model):
    sents = sentsplitter().my_splitter(text=text)
    embeddings = model.encode(sents)
    sim_matrix = util.cos_sim(embeddings, embeddings)
    return sim_matrix, sents


def build_ssv_graph(sim_matrix):
    edge_list = []
    sim_matrix = sim_matrix - np.eye(len(sim_matrix))
    for i in range(1, len(sim_matrix), 1):
        max_sent = int(np.argmax(sim_matrix[i, :]))
        distance = abs(max_sent - i)
        weight = float(max(sim_matrix[i, :]) / distance)
        edge_list.append((i, max_sent, weight))
    return edge_list


class TokenizedDataset(Dataset):
    def __init__(
        self,
        args,
        training_args,
        tokenizer,
        raw_dataset,
        flags,
        sent_model,
    ):
        self.args = args
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.raw_dataset = raw_dataset
        self.sent_model = sent_model

    def __getitem__(self, index):
        raw_item = self.raw_dataset[index]
        batch_idx = index

        seq_in = raw_item["article"]
        label = 0 if raw_item["label"] == "human" else 1
        sim_matrix, sents = cal_sim_matrix(seq_in, self.sent_model)
        ssv_graph = build_ssv_graph(sim_matrix)
        g = nx.DiGraph()
        g.add_weighted_edges_from(ssv_graph)
        graph = VGraph(g)
        edges = [list(pair) for pair in graph.g.edges()]
        graph.edge_mat = torch.LongTensor(edges).transpose(0, 1)

        tokenized_input = self.tokenizer.batch_encode_plus(
            sents,
            padding="max_length",
            truncation=True,
            max_length=self.training_args.input_max_length,
            is_split_into_words=True,
        )

        item = {
            "input_ids": torch.LongTensor(tokenized_input.data["input_ids"]),
            "attention_mask": torch.LongTensor(tokenized_input.data["attention_mask"]),
            "g": graph,
            "labels": torch.LongTensor([label]),
            "batch_idx": batch_idx,
        }
        return item

    def __len__(self):
        return len(self.raw_dataset)


class Collator:
    def __init__(self, args, training_args):
        self.args = args
        self.training_args = training_args

    def __call__(self, batch):
        """
        batch is a list containing [item1, item2, ...]
        should operate on every item in dataset
        shape of dataset
        item = {
        'input_ids': torch.LongTensor(tokenized_input.data["input_ids"]),
        'attention_mask': torch.LongTensor(tokenized_input.data["attention_mask"]),
        'graph': list(g),
        'labels': torch.LongTensor([label]),
        'batch_idx': batch_idx,
        }
        method: put adj_matrix into a diagnal matrix
        """
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]
        batched_batch_idx = [item["batch_idx"] for item in batch]

        batched_input_ids = torch.vstack(input_ids)
        batched_attention_mask = torch.vstack(attention_mask)
        batched_labels = torch.stack(labels)
        batched_graph = [item["g"] for item in batch]

        return {
            "input_ids": torch.LongTensor(batched_input_ids),
            "attention_mask": torch.LongTensor(batched_attention_mask),
            "labels": torch.LongTensor(batched_labels),
            "g": batched_graph,
            "batch_ids": torch.LongTensor(batched_batch_idx),
        }
