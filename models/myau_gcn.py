import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer_encoder import TransformerEncoder


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features,
                 bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features,
                                                out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter("bias", None)
        self.init_parameters()

    def init_parameters(self):
        std = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, x):
        x = x @ self.weight
        if self.bias is not None:
            x += self.bias
        return x


class GCN(nn.Module):
    def __init__(self, adj_matrix, hidden_features: int = 80,
                 num_embeddings: int = 15, in_features: int = 40,
                 out_features: int = 160):
        super(GCN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings,
                                      embedding_dim=in_features)

        self.graph_weight_one = GraphConvolution(in_features=in_features,
                                                 out_features=hidden_features,
                                                 bias=False)

        self.graph_weight_two = GraphConvolution(in_features=hidden_features,
                                                 out_features=out_features,
                                                 bias=False)

        self.adjEncoder = nn.Sequential(*[
            TransformerEncoder(input_dim=9,
                               forward_dim=40,
                               num_heads=8,
                               head_dim=16,
                               drop_rate=0.1)
            for _ in range(6)
        ])

    def forward(self, x, au_seq):
        # Before: Shape of x: (batch_size, 9)
        # After: Shape of x: (batch_size, 9, in_features)
        # embed = self.embedding(au_seq)

        batch_size = x.size(0)
        temp = torch.empty((batch_size, 9, 9)).cuda()
        node = torch.empty((batch_size, 9, 40)).cuda()
        for idx in range(batch_size):
            # Learning of adjacency matrix
            temp[idx] = torch.diag(x[idx])
            min_au = torch.min(x[idx]) [0.2, 0.2, ..., 1] [0.8, 0.8, ..., 1]
            max_au = torch.max(x[idx])
            temp_feature = (x[idx] - min_au) / (max_au - min_au + 1e-4)
            # Learning of node features
            node[idx] = self.embedding(torch.floor(temp_feature * 15).long())

        adj = self.adjEncoder(temp)

        # Go through two-layers GCN
        # Shape of x: (batch_size, 9, hidden_features)
        x = adj @ self.graph_weight_one(node)
        x = F.leaky_relu(x, 0.2)

        # Shape of x: (batch_size, 9, output_features)
        x = adj @ self.graph_weight_two(x)
        x = F.leaky_relu(x, 0.2)
        return x


if __name__ == "__main__":
    adj_matrix = torch.FloatTensor([
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [1, 0, 1, 0]
    ])

    model = GCN(adj_matrix, num_embeddings=4, hidden_features=80)
    test_tensor = torch.randint(low=0, high=3, size=(4,))
    print(model(test_tensor).shape)
