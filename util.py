# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Based on code from the above authors, modifications made by Xi'an Jiaotong University.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" GLUE processors and helpers """

import logging
import os
import numpy as np
import json
import re
import copy
from transformers.file_utils import is_tf_available
from transformers.data.processors.utils import DataProcessor


logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    def __init__(
        self,
        guid,
        text_a,
        text_b=None,
        label=None,
        nodes_index=None,
        adj_metric=None,
        all_tokens=None,
        sen2node=None,
        nodes_ent=None,
    ):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.nodes_index = nodes_index
        self.adj_metric = adj_metric
        self.all_tokens = all_tokens
        self.sen2node = sen2node
        self.nodes_ent = nodes_ent

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """
    A single set of features of data.
    """

    def __init__(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        label=None,
        nodes_index=None,
        adj_metric=None,
        sen2node=None,
        nodes_ent=None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label
        self.nodes_index = nodes_index
        self.adj_metric = adj_metric
        self.sen2node = sen2node
        self.nodes_ent = nodes_ent

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def glue_convert_examples_to_features(
    examples,
    tokenizer,
    max_length=512,
    task=None,
    label_list=None,
    output_mode=None,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    """
    Loads a data file into a list of ``InputFeatures``
    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)
    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.
    """

    if task is not None:
        processor = glue_processors[task]()
    if label_list is None:
        label_list = processor.get_labels()
        logger.info("Using label list %s for task %s" % (label_list, task))
    if output_mode is None:
        output_mode = glue_output_modes[task]
        logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for ex_index, example in enumerate(examples):
        len_examples = 0

        len_examples = len(examples)
        if ex_index % 10000 == 0:
            logger.info("Writing example %d/%d" % (ex_index, len_examples))

        inputs = tokenizer.encode_plus(
            example.text_a,
            add_special_tokens=True,
            max_length=max_length,
            return_token_type_ids=True,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # Tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = (
                [0 if mask_padding_with_zero else 1] * padding_length
            ) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + (
                [0 if mask_padding_with_zero else 1] * padding_length
            )
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(
            len(input_ids), max_length
        )
        assert (
            len(attention_mask) == max_length
        ), "Error with input length {} vs {}".format(len(attention_mask), max_length)
        assert (
            len(token_type_ids) == max_length
        ), "Error with input length {} vs {}".format(len(token_type_ids), max_length)

        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info(
                "attention_mask: %s" % " ".join([str(x) for x in attention_mask])
            )
            logger.info(
                "token_type_ids: %s" % " ".join([str(x) for x in token_type_ids])
            )
            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label=label,
                nodes_index=example.nodes_index,
                adj_metric=example.adj_metric,
                sen2node=example.sen2node,
                nodes_ent=example.nodes_ent,
            )
        )

    return features


class DeepFakeProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def _read_jsonl(self, path):
        file = open(path, "r", encoding="utf8")
        data = file.readlines()
        return data

    def get_train_examples(
        self, with_relation, data_dir, train_file="gpt2_500_train_Graph.jsonl"
    ):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, train_file)))
        return self._create_examples(
            with_relation, self._read_jsonl(os.path.join(data_dir, train_file)), "train"
        )

    def get_dev_examples(
        self, with_relation, data_dir, dev_file="gpt2_dev_Graph.jsonl"
    ):
        """See base class."""
        return self._create_examples(
            with_relation, self._read_jsonl(os.path.join(data_dir, dev_file)), "val"
        )

    def get_test_examples(
        self, with_relation, data_dir, test_file="gpt2_test_Graph.jsonl"
    ):
        """See base class."""
        return self._create_examples(
            with_relation, self._read_jsonl(os.path.join(data_dir, test_file)), "test"
        )

    def get_labels(self):
        """See base class."""
        return ["human", "machine"]

    def _get_nodes(self, nodes):
        all_nodes_index = []
        all_nodes_ent = []
        for node in nodes:
            all_nodes_index.append(node["spans"])
            all_nodes_ent.append(self.clean_string(node["text"]))
        return all_nodes_index, all_nodes_ent

    def _get_adj_metric(self, edges, drop_nodes, node_num, with_relation):
        if with_relation == 0:
            adj_matrix = np.eye(node_num)
            for edge in edges:
                adj_matrix[edge[0], edge[1]] = 1
                adj_matrix[edge[1], edge[0]] = 1

            for idx in drop_nodes:
                adj_matrix[idx, :] = 0
                adj_matrix[:, idx] = 0
        else:
            adj_matrix = list()
            relation = {"inner": 0, "inter": 1}
            for i in range(with_relation):
                adj_matrix.append(np.eye(node_num))
            adj_matrix = np.stack(adj_matrix, axis=0)
            for edge in edges:
                r_i = relation[edge[2]]
                adj_matrix[r_i][edge[0], edge[1]] = 1
                adj_matrix[r_i][edge[1], edge[0]] = 1

            for idx in drop_nodes:
                for r_i in range(len(relation)):
                    adj_matrix[r_i][idx, :] = 0
                    adj_matrix[r_i][:, idx] = 0

        return adj_matrix

    def clean_string(self, string):
        return re.sub(r"[^a-zA-Z0-9 ]+", "", string)

    def _create_examples(self, with_relation, inputs, set_type):
        """Creates examples for the training and dev sets."""
        lines = inputs
        examples = []
        bad = 0
        for i, line in enumerate(lines):
            line = json.loads(line.strip())
            if "split" not in line.keys() or line["split"] == set_type:
                guid = "%s-%s" % (set_type, i)
                text_a = line["article"]
                text_b = None
                if "label" in line.keys():
                    label = line["label"]
                else:
                    label = "machine"
                graph_info = line["information"]["graph"]
                nodes, edges, all_tokens, drop_nodes, sen2node = (
                    graph_info["nodes"],
                    graph_info["edges"],
                    graph_info["all_tokens"],
                    graph_info["drop_nodes"],
                    graph_info["sentence_to_node_id"],
                )
                nodes_index, nodes_ent = self._get_nodes(nodes)
                adj_metric = self._get_adj_metric(
                    edges, drop_nodes, len(nodes_index), with_relation
                )

                if len(all_tokens) != 0:
                    examples.append(
                        InputExample(
                            guid=guid,
                            text_a=text_a,
                            text_b=text_b,
                            label=label,
                            nodes_index=nodes_index,
                            adj_metric=adj_metric,
                            all_tokens=all_tokens,
                            sen2node=sen2node,
                            nodes_ent=nodes_ent,
                        )
                    )
                else:
                    bad += 1
                    continue
        logger.info("\n {} instances has no input".format(bad))
        return examples

glue_tasks_num_labels = {"deepfake": 2}

glue_processors = {"deepfake": DeepFakeProcessor}

glue_output_modes = {"deepfake": "classification"}

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }

def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }

def glue_compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc" or task_name == "deepfake":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "hans":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)

def xnli_compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "xnli":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)
