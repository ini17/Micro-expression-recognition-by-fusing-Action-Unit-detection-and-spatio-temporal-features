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
    def __init__(self, hidden_features: int = 80,
                 num_embeddings: int = 15, in_features: int = 40,
                 out_features: int = 160):
        super(GCN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings,
                                      embedding_dim=in_features)

        self.gcn1 = GraphConvolution(in_features=1,
                                     out_features=in_features,
                                     bias=False)

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
        batch_size = x.size(0)
        temp = torch.empty((batch_size, 9, 9)).cuda()
        for idx in range(batch_size):
            temp[idx] = torch.diag(x[idx])

        adj = self.adjEncoder(temp)

        # Go through three-layers GCNs
        x = x.unsqueeze(-1)
        node = adj @ self.gcn1(x)
        node = F.leaky_relu(node, 0.2)
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

    model = GCN(num_embeddings=4, hidden_features=80)
    test_tensor = torch.randint(low=0, high=3, size=(4,))
    print(model(test_tensor).shape)
