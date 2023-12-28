import torch
import torch.nn as nn
from custom_LSTM import CustomRNN

class GCN(nn.Module):
    def __init__(self, GCN_layer, input_size, relation_n):
        super(GCN, self).__init__()
        self.GCN_layer = GCN_layer
        self.relation_n = relation_n
        self.GCNweight = nn.ModuleList()

        if relation_n == 0:
            relation_n = 1
        for _ in range(GCN_layer * relation_n):
            self.GCNweight.append(nn.Linear(input_size, input_size))

    def normalize_laplacian_matrix(self, adj):
        row_sum_invSqrt, temp = torch.pow(adj.sum(2) + 1e-30, -0.5), []
        for i in range(adj.size()[0]):
            temp.append(torch.diag(row_sum_invSqrt[i]))
        degree_matrix = torch.cat(temp, dim=0).view(adj.size())
        return torch.matmul(torch.matmul(degree_matrix, adj), degree_matrix).to(
            degree_matrix.device
        )

    def forward(self, nodes_rep, adj_metric):
        relation_num = self.relation_n
        if relation_num == 0:
            normalized_laplacian_matrix = self.normalize_laplacian_matrix(adj_metric)
            normalized_laplacian_matrix.requires_grad = False
            nodes_rep_history = [nodes_rep]
            for i in range(self.GCN_layer):
                tmp_rep = torch.bmm(normalized_laplacian_matrix, nodes_rep_history[i])
                nodes_rep_history.append(torch.tanh(self.GCNweight[i](tmp_rep)))
            nodes_rep_history = torch.stack(nodes_rep_history, dim=0)
            return nodes_rep_history
        else:
            adj_idx = 0
            tot_nodes_rep_history = list()
            adj_metric = adj_metric.transpose(0, 1)
            for adj in adj_metric:
                normalized_laplacian_matrix = self.normalize_laplacian_matrix(adj)
                normalized_laplacian_matrix.requires_grad = False
                nodes_rep_history = [nodes_rep]
                for i in range(self.GCN_layer):
                    tmp_rep = torch.bmm(
                        normalized_laplacian_matrix, nodes_rep_history[i]
                    )
                    nodes_rep_history.append(
                        torch.tanh(
                            self.GCNweight[adj_idx * self.GCN_layer + i](tmp_rep)
                        )
                    )
                nodes_rep_history = torch.stack(nodes_rep_history, dim=0)
                tot_nodes_rep_history.append(nodes_rep_history)
                adj_idx += 1
            tot_nodes_rep_history = torch.stack(tot_nodes_rep_history, axis=0)
            tot_nodes_rep_history = torch.sum(tot_nodes_rep_history, axis=0)
            return tot_nodes_rep_history

def function_align(x, y, x_mask, y_mask, input_size):
    x_mask_tile = x_mask.unsqueeze(-1).repeat(1, 1, x.shape[-1])
    y_mask_tile = y_mask.unsqueeze(-1).repeat(1, 1, y.shape[-1])
    x = x * x_mask_tile.float()
    y = y * y_mask_tile.float()
    return torch.cat([x - y, x * y], dim=2)

def mask_mean(x, x_mask, dim):
    """
    :param x: batch, nodes_num, hidden_size
    :param x_mask: batch, nodes_num
    :param dim:
    :return: x
    """
    x_mask_tile = x_mask.unsqueeze(-1).repeat(1, 1, x.shape[-1])
    assert x.shape == x_mask_tile.shape, "x shape {}, x_mask_tile shape {}".format(
        x.shape, x_mask_tile.shape
    )

    result = torch.sum(x * x_mask_tile.float(), dim=dim) / (
        torch.sum(x_mask_tile.float(), dim=dim) + 1e-30
    )

    return result

class GCNGraphAgg(nn.Module):
    def __init__(
        self,
        input_size,
        max_sentence_num,
        gcn_layer,
        node_size,
        attention_maxscore=None,
        relation_num=None,
    ):
        super(GCNGraphAgg, self).__init__()
        self.input_size = input_size
        self.max_sentence_num = max_sentence_num
        self.gcn_layer = gcn_layer
        self.node_size = node_size
        self.relation_num = relation_num
        self.graph_node_proj = nn.Linear(input_size, node_size)
        self.align_proj = nn.Linear(self.node_size * 2, self.node_size)
        self.GCN = GCN(self.gcn_layer, self.node_size, self.relation_num)
        self.rnn_coherence_proj = CustomRNN(
            input_size=self.node_size,
            hidden_size=self.node_size,
            batch_first=True,
            max_seq_length=max_sentence_num,
            attention_maxscore=attention_maxscore,
        )

    def forward(
        self,
        hidden_states,
        nodes_index_mask,
        adj_metric,
        node_mask,
        sen2node,
        sentence_mask,
        sentence_length,
    ):
        """
        :param hidden_states: batch, seq_len, hidden_size
        :param nodes_mask: batch, node_num, seq_len
        :param claim_node_mask: batch, claim_node_num, seq_len
        :return: logits
        """
        """evidence nodes and edges presentation"""
        nodes_rep = torch.bmm(nodes_index_mask, hidden_states)
        nodes_rep = torch.relu(self.graph_node_proj(nodes_rep))

        """GCN propagation"""
        nodes_rep_history = self.GCN(nodes_rep, adj_metric)
        joint_nodes_rep = nodes_rep_history[-1, :, :, :]
        sens_rep = torch.bmm(sen2node, joint_nodes_rep)

        final_rep, padded_output = self.rnn_coherence_proj(
            sens_rep, sentence_length, sentence_mask
        )

        return final_rep
