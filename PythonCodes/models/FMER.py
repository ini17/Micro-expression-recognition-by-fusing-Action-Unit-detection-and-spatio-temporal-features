import torch
import torch.nn as nn
from .graph_model import GraphLearningModel
from .myau_gcn import GCN
from .au_fusion import AUFusion
from .p3d_model import P3D63


class FMER(nn.Module):
    def __init__(self, num_classes,
                 device, hidden_features=80):
        super(FMER, self).__init__()
        self.graph = GraphLearningModel()
        self.au_gcn = GCN(hidden_features=hidden_features)
        self.au_fusion = AUFusion(num_classes=num_classes)
        self.P3D = P3D63(num_classes=9)

        # Used to train the embedding
        self.au_seq = torch.arange(9).to(device)

    def forward(self, patches, frames):
        batch_size = patches.size(0)

        # Node learning and edge learning
        # Shape of patches: (batch_size, 30, 7, 7)
        eyebrow, mouth = self.graph(patches)

        # AU features extraction by P3D
        au_feature = self.P3D(frames)

        # Training the GCN
        # Shape of au_seq: (9)
        # Shape of gcn_output: (batch_size, 9, 160)
        gcn_output = self.au_gcn(au_feature, self.au_seq)

        # Fuse the graph learning and GCN
        fusion_output = self.au_fusion(eyebrow, mouth, gcn_output)

        return fusion_output, au_feature


if __name__ == "__main__":
    test_tensor = torch.rand(1, 30, 7, 7)
    adj_matrix = torch.rand(9, 9)
    model = FMER(num_classes=5,
                 device="cpu")

    print(model(test_tensor).shape)