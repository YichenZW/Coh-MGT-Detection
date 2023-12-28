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
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import (
    RobertaModel,
    RobertaForSequenceClassification,
    BertPreTrainedModel,
)
from additional_model import GCNGraphAgg

logger = logging.getLogger(__name__)


def dequeue_and_enqueue(hidden_batch_feats, selected_batch_idx, queue):
    """
    Update memory bank by batch window slide; hidden_batch_feats must be normalized
    """
    assert hidden_batch_feats.size()[1] == queue.size()[1]

    queue[selected_batch_idx] = F.normalize(hidden_batch_feats, dim=1)

    return queue


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, graph_node_size=None):
        super(RobertaClassificationHead, self).__init__()
        if graph_node_size:
            self.dense = nn.Linear(
                config.hidden_size + graph_node_size, config.hidden_size
            )
        else:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class RobertaForGraphBasedSequenceClassification(
    BertPreTrainedModel
): 
    def __init__(self, config):
        config.output_hidden_states = True
        config.output_attentions = True

        super(RobertaForGraphBasedSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.classifier = RobertaClassificationHead(config, graph_node_size=None)
        self.graph_aggregation = GCNGraphAgg(
            config.hidden_size, self.node_size, self.max_sentence_size
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        nodes_index_mask=None,
        adj_metric=None,
        node_mask=None,
        sen2node=None,
        sentence_mask=None,
        sentence_length=None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0][:, 0, :]
        
        hidden_states = outputs[2][0]

        graph_rep = self.graph_aggregation(
            hidden_states,
            nodes_index_mask,
            adj_metric,
            node_mask,
            sen2node,
            sentence_mask,
            sentence_length,
        )
        whole_rep = torch.cat([sequence_output, graph_rep], dim=-1)

        logits = self.classifier(whole_rep, dim=-1)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (loss,) + outputs

        return outputs, whole_rep  

class RobertaForGraphBasedSequenceClassification_CL(BertPreTrainedModel):
    def __init__(self, config):
        config.output_hidden_states = True
        config.output_attentions = True

        super(RobertaForGraphBasedSequenceClassification_CL, self).__init__(config)
        self.temperature = 0.2
        self.num_labels = config.num_labels
        self.gcn_layer = config.task_specific_params["gcn_layer"]
        self.max_node_num = config.task_specific_params["max_nodes_num"]
        self.max_sentences = config.task_specific_params["max_sentences"]
        self.max_sen_replen = config.task_specific_params["max_sen_replen"]
        self.attention_maxscore = config.task_specific_params["attention_maxscore"]
        self.relation_num = config.task_specific_params["relation_num"]

        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(
            config, graph_node_size=self.max_sen_replen
        )
        self.graph_aggregation = GCNGraphAgg(
            config.hidden_size,
            self.max_sentences,
            self.gcn_layer,
            self.max_sen_replen,
            self.attention_maxscore,
            self.relation_num,
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        nodes_index_mask=None,
        adj_metric=None,
        node_mask=None,
        sen2node=None,
        sentence_mask=None,
        sentence_length=None,
        batch_id=None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0][:, 0, :]
        hidden_states = outputs[2][0]

        graph_rep = self.graph_aggregation(
            hidden_states,
            nodes_index_mask,
            adj_metric,
            node_mask,
            sen2node,
            sentence_mask,
            sentence_length,
        )
        whole_rep = torch.cat([sequence_output, graph_rep], dim=-1)

        logits = self.classifier(torch.cat([sequence_output, graph_rep], dim=-1))

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                batch_size = len(labels)
                batch_idx_by_label = {}
                for i in range(2):
                    batch_idx_by_label[i] = [
                        idx
                        for idx in range(batch_size)
                        if int(labels.view(-1)[idx]) == i
                    ] 

                contraloss = self.contrastive_loss_labelwise_winslide(
                    batch_size, batch_idx_by_label, whole_rep
                )

                loss_fct = CrossEntropyLoss()
                ce_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

                contraloss_weight = 0.6
                loss = (
                    1.0 - contraloss_weight
                ) * ce_loss + contraloss_weight * contraloss
            outputs = (loss,) + outputs
        return outputs, whole_rep 

    def get_key(self, dic, value):
        return [k for k, v in dic.items() if value in v]

    def contrastive_loss_labelwise_winslide(
        self, batch_size, batch_idx_by_label, hidden_feats
    ):
        """
        Hidden feats must be normalized

        """
        hidden_feats = F.normalize(hidden_feats, dim=1)
        sim_matrix = torch.mm(hidden_feats, hidden_feats.T) 
        loss = 0.0

        for i in range(batch_size):
            label_list = self.get_key(batch_idx_by_label, i)
            label = label_list[0]
            one_same_label = (
                torch.zeros((batch_size,))
                .to(sim_matrix.device)
                .scatter_(
                    0,
                    torch.tensor(batch_idx_by_label[label]).to(sim_matrix.device),
                    1.0,
                )
            )
            one_diff_label = (
                torch.ones((batch_size,))
                .to(sim_matrix.device)
                .scatter_(
                    0,
                    torch.tensor(batch_idx_by_label[label]).to(sim_matrix.device),
                    0.0,
                )
            )
            one_for_not_i = (
                torch.ones((batch_size,))
                .to(sim_matrix.device)
                .scatter_(0, torch.tensor([i]).to(sim_matrix.device), 0.0)
            )  
            one_for_numerator = one_same_label.mul(one_for_not_i)

            numerator = torch.sum(
                one_for_numerator * torch.exp(sim_matrix[i, :] / self.temperature)
            )
            denominator = torch.sum(
                one_for_not_i * torch.exp(sim_matrix[i, :] / self.temperature)
            )

            if numerator == 0:
                numerator += 1e-6
            if denominator == 0:
                denominator += 1e-6

            loss += -torch.log(numerator / denominator)

        return loss / batch_size


class EncoderForMBCL(BertPreTrainedModel):
    def __init__(self, config):
        super(EncoderForMBCL, self).__init__(config)
        self.max_sen_replen = config.task_specific_params["max_sen_replen"]
        self.max_sentences = config.task_specific_params["max_sentences"]
        self.gcn_layer = config.task_specific_params["gcn_layer"]
        self.attention_maxscore = config.task_specific_params["attention_maxscore"]
        self.relation_num = config.task_specific_params["relation_num"]

        self.roberta = RobertaModel(config)
        self.graph_aggregation = GCNGraphAgg(
            config.hidden_size,
            self.max_sentences,
            self.gcn_layer,
            self.max_sen_replen,
            self.attention_maxscore,
            self.relation_num,
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        nodes_index_mask=None,
        adj_metric=None,
        node_mask=None,
        sen2node=None,
        sentence_mask=None,
        sentence_length=None,
        batch_id=None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0][:, 0, :]  
        hidden_states = outputs[2][0]  

        graph_rep = self.graph_aggregation(
            hidden_states,
            nodes_index_mask,
            adj_metric,
            node_mask,
            sen2node,
            sentence_mask,
            sentence_length,
        )

        whole_rep = torch.cat([sequence_output, graph_rep], dim=-1) 

        return outputs[2:], whole_rep

class RobertaForGraphBasedSequenceClassification_MBCL(BertPreTrainedModel):
    def __init__(self, config, mb_dataloader, train_idx_by_label):
        config.output_hidden_states = True
        config.output_attentions = True

        super(RobertaForGraphBasedSequenceClassification_MBCL, self).__init__(config)
        self.temperature = 0.2
        self.num_labels = config.num_labels
        self.gcn_layer = config.task_specific_params["gcn_layer"]
        self.max_node_num = config.task_specific_params["max_nodes_num"]
        self.max_sentences = config.task_specific_params["max_sentences"]
        self.max_sen_replen = config.task_specific_params["max_sen_replen"]
        self.attention_maxscore = config.task_specific_params["attention_maxscore"]
        self.relation_num = config.task_specific_params["relation_num"]
        self.train_idx_by_label = train_idx_by_label
        self.classifier = RobertaClassificationHead(
            config, graph_node_size=self.max_sen_replen
        )
        self.model_q = EncoderForMBCL(config)
        self.model_k = EncoderForMBCL(config)
        for param_q, param_k in zip(
            self.model_q.parameters(), self.model_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  
        self.model_q.cuda()
        self.model_k.cuda()
        with torch.no_grad():
            for k, item in enumerate(mb_dataloader):
                input_ids = item[0].cuda()
                attention_mask = item[1].cuda()
                labels = item[3].cuda()
                nodes_index_mask = item[4].cuda()
                adj_metric = item[5].cuda()
                node_mask = item[6].cuda()
                sen2node = item[7].cuda()
                sentence_mask = item[8].cuda()
                sentence_length = item[9].cuda()

                output = self.model_q(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    nodes_index_mask=nodes_index_mask,
                    adj_metric=adj_metric,
                    node_mask=node_mask,
                    sen2node=sen2node,
                    sentence_mask=sentence_mask,
                    sentence_length=sentence_length,
                )
                init_feat = F.normalize(output[1], dim=1)
                if k == 0:
                    self.queue = init_feat
                else:
                    self.queue = torch.vstack((self.queue, init_feat))

        print(self.queue.size())
        print("***queue already builded***")

        self.config = self.model_q.config
        self.feat_dim = self.config.hidden_size

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        nodes_index_mask=None,
        adj_metric=None,
        node_mask=None,
        sen2node=None,
        sentence_mask=None,
        sentence_length=None,
        batch_id=None,
    ):
        if self.training:
            batch_size = int(input_ids.size(0))
            output_q = self.model_q(
                input_ids,
                attention_mask,
                token_type_ids,
                position_ids,
                head_mask,
                inputs_embeds,
                labels,
                nodes_index_mask,
                adj_metric,
                node_mask,
                sen2node,
                sentence_mask,
                sentence_length,
                batch_id,
            )  
            q_feat = output_q[1]
            logits = self.classifier(output_q[1])
            outputs = (logits,) + output_q[0]
            loss_fct = CrossEntropyLoss()
            q_ce_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            output_k = self.model_k(
                input_ids,
                attention_mask,
                token_type_ids,
                position_ids,
                head_mask,
                inputs_embeds,
                labels,
                nodes_index_mask,
                adj_metric,
                node_mask,
                sen2node,
                sentence_mask,
                sentence_length,
                batch_id,
            )
            k_feat = output_k[1]
            self.dequeue_and_enqueue(k_feat, batch_id)
            batch_idx_by_label = {}
            for i in range(2):
                batch_idx_by_label[i] = [
                    idx for idx in range(batch_size) if labels[idx] == i
                ]  
            contraloss = self.contrastive_loss_es(
                batch_size, batch_idx_by_label, q_feat
            )
            self.momentum_update(m=0.999)
            contraloss_weight = 0.6
            loss = (
                1.0 - contraloss_weight
            ) * q_ce_loss + contraloss_weight * contraloss

            outputs = (loss,) + outputs

            return outputs, output_q[1]  
        else:
            batch_size = int(input_ids.size(0))
            output_q = self.model_q(
                input_ids,
                attention_mask,
                token_type_ids,
                position_ids,
                head_mask,
                inputs_embeds,
                labels,
                nodes_index_mask,
                adj_metric,
                node_mask,
                sen2node,
                sentence_mask,
                sentence_length,
                batch_id,
            ) 
            q_feat = output_q[1]
            logits = self.classifier(output_q[1])
            outputs = (logits,) + output_q[0]
            loss_fct = CrossEntropyLoss()
            q_ce_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            batch_idx_by_label = {}
            for i in range(2):
                batch_idx_by_label[i] = [
                    idx for idx in range(batch_size) if labels[idx] == i
                ]  
            contraloss = self.contrastive_loss_es(
                batch_size, batch_idx_by_label, q_feat
            )
            contraloss_weight = 0.6
            loss = (
                1.0 - contraloss_weight
            ) * q_ce_loss + contraloss_weight * contraloss

            outputs = (loss,) + outputs

            return outputs, output_q[1] 

    def get_key(self, dic, value):
        return [k for k, v in dic.items() if value in v]

    def contrastive_loss_es(self, batch_size, batch_idx_by_label, hidden_feats):
        hidden_feats = F.normalize(hidden_feats, dim=1)
        change_dic = {0: 1, 1: 0}
        loss = 0

        for i in batch_idx_by_label:
            q = hidden_feats[batch_idx_by_label[i]]
            pos_bank = self.queue[self.train_idx_by_label[i]]
            pos_pair = torch.mm(q, pos_bank.transpose(0, 1))
            bottom_k = torch.topk(pos_pair, k=100, dim=1, largest=False).values
            neg_bank = self.queue[self.train_idx_by_label[change_dic[i]]]
            neg_pair = torch.mm(q, neg_bank.transpose(0, 1))
            top_k = torch.topk(neg_pair, k=100, dim=1).values
            numerator = torch.sum(torch.exp(bottom_k / self.temperature), dim=1)
            denominator = (
                torch.sum(torch.exp(top_k / self.temperature), dim=1) + numerator
            )

            for nid in range(len(numerator)):
                if numerator[nid] == 0:
                    numerator[nid] += 1e-6
            for did in range(len(denominator)):
                if denominator[did] == 0:
                    denominator[did] += 1e-6
            loss += torch.sum(-1.0 * torch.log(numerator / denominator))

        return loss / batch_size

    @torch.no_grad()
    def momentum_update(self, m=0.999):
        """
        encoder_k = m * encoder_k + (1 - m) encoder_q
        """
        for param_q, param_k in zip(
            self.model_q.parameters(), self.model_k.parameters()
        ):
            param_k.data = param_k.data * m + param_q.data * (1.0 - m)

    def dequeue_and_enqueue(self, hidden_batch_feats, selected_batch_idx):
        """
        Update memory bank by batch window slide; hidden_batch_feats must be normalized
        """
        assert hidden_batch_feats.size()[1] == self.queue.size()[1]

        self.queue[selected_batch_idx] = F.normalize(hidden_batch_feats, dim=1)


class RobertaForGraphBasedSequenceClassification_RFCL(BertPreTrainedModel):
    def __init__(self, config):
        config.output_hidden_states = True
        config.output_attentions = True

        super(RobertaForGraphBasedSequenceClassification_RFCL, self).__init__(config)
        self.temperature = 0.2
        self.num_labels = config.num_labels
        self.gcn_layer = config.task_specific_params["gcn_layer"]
        self.max_node_num = config.task_specific_params["max_nodes_num"]
        self.max_sentences = config.task_specific_params["max_sentences"]
        self.max_sen_replen = config.task_specific_params["max_sen_replen"]
        self.attention_maxscore = config.task_specific_params["attention_maxscore"]
        self.relation_num = config.task_specific_params["relation_num"]

        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(
            config, graph_node_size=self.max_sen_replen
        )
        self.graph_aggregation = GCNGraphAgg(
            config.hidden_size,
            self.max_sentences,
            self.gcn_layer,
            self.max_sen_replen,
            self.attention_maxscore,
            self.relation_num,
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        nodes_index_mask=None,
        adj_metric=None,
        node_mask=None,
        sen2node=None,
        sentence_mask=None,
        sentence_length=None,
        batch_id=None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0][:, 0, :]
        hidden_states = outputs[2][0]

        graph_rep = self.graph_aggregation(
            hidden_states,
            nodes_index_mask,
            adj_metric,
            node_mask,
            sen2node,
            sentence_mask,
            sentence_length,
        )
        whole_rep = torch.cat([sequence_output, graph_rep], dim=-1)

        logits = self.classifier(torch.cat([sequence_output, graph_rep], dim=-1))

        outputs = (logits,) + outputs[2:]

        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                batch_size = len(labels)
                batch_idx_by_label = {}
                for i in range(2):
                    batch_idx_by_label[i] = [
                        idx
                        for idx in range(batch_size)
                        if int(labels.view(-1)[idx]) == i
                    ]  

                contraloss = self.contrastive_loss_es(
                    batch_size, batch_idx_by_label, whole_rep
                )

                loss_fct = CrossEntropyLoss()
                ce_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

                contraloss_weight = 0.6
                loss = (
                    1.0 - contraloss_weight
                ) * ce_loss + contraloss_weight * contraloss

            outputs = (loss,) + outputs

        return outputs, whole_rep 

    def get_key(self, dic, value):
        return [k for k, v in dic.items() if value in v]

    def contrastive_loss_es(self, batch_size, batch_idx_by_label, hidden_feats):
        hidden_feats = F.normalize(hidden_feats, dim=1)
        loss = 0
        sim_matrix = torch.mm(hidden_feats, hidden_feats.T)  
        loss = 0.0

        for i in range(batch_size):
            label_list = self.get_key(batch_idx_by_label, i)
            label = label_list[0]
            one_same_label = (
                torch.zeros((batch_size,))
                .to(sim_matrix.device)
                .scatter_(
                    0,
                    torch.tensor(batch_idx_by_label[label]).to(sim_matrix.device),
                    1.0,
                )
            )
            one_diff_label = (
                torch.ones((batch_size,))
                .to(sim_matrix.device)
                .scatter_(
                    0,
                    torch.tensor(batch_idx_by_label[label]).to(sim_matrix.device),
                    0.0,
                )
            )
            one_for_not_i = (
                torch.ones((batch_size,))
                .to(sim_matrix.device)
                .scatter_(0, torch.tensor([i]).to(sim_matrix.device), 0.0)
            )  
            one_for_numerator = one_same_label.mul(one_for_not_i)
            one_for_neg = one_diff_label.mul(one_for_not_i)

            numerator = torch.sum(
                one_for_numerator * torch.exp(sim_matrix[i, :] / self.temperature)
            )
            denominator = torch.sum(
                one_for_not_i * torch.exp(sim_matrix[i, :] / self.temperature)
            )

            if numerator == 0:
                numerator += 1e-6
            if denominator == 0:
                denominator += 1e-6

            loss += -torch.log(numerator / denominator)

        return loss / batch_size
