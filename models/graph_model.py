import torch
import torch.nn as nn
from .transformer_encoder import TransformerEncoder


class GraphLearningModel(nn.Module):
    def __init__(self,
                 parallel=9,
                 input_dim: int = 49,
                 forward_dim: int = 128,
                 num_heads: int = 8,
                 head_dim: int = 16,
                 num_layers: int = 6,
                 attn_drop_rate: float = 0.1,
                 proj_drop_rate: float = 0.5,
                 in_channels: int = 30,
                 stride: int = 1,
                 kernel_size: int = 3):
        super(GraphLearningModel, self).__init__()

        self.parallel = parallel
        # Depth wise convolution for the input
        self.DWConv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels,
                      out_channels=in_channels,
                      stride=stride,
                      kernel_size=(parallel, kernel_size, kernel_size),
                      padding=(0, kernel_size//2, kernel_size//2),
                      groups=in_channels
                      ),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(start_dim=2)
        )

        self.OFConv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels,
                      out_channels=in_channels,
                      stride=stride,
                      kernel_size=(2, kernel_size, kernel_size),
                      padding=(0, kernel_size//2, kernel_size//2),
                      groups=in_channels),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.OFConv_List = nn.ModuleList([
            self.OFConv for _ in range(parallel)
        ])

        self.eyebrow_encoder = nn.Sequential(*[
            TransformerEncoder(input_dim=input_dim,
                               forward_dim=forward_dim,
                               num_heads=num_heads,
                               head_dim=head_dim,
                               drop_rate=attn_drop_rate)
            for _ in range(num_layers)
        ])
        self.mouth_encoder = nn.Sequential(*[
            TransformerEncoder(input_dim=input_dim,
                               forward_dim=forward_dim,
                               num_heads=num_heads,
                               head_dim=head_dim,
                               drop_rate=attn_drop_rate)
            for _ in range(num_layers)
        ])

        self.eyebrow_layer = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(490, 320),
            nn.ReLU(inplace=True),
            nn.Dropout(p=proj_drop_rate),
            nn.Linear(320, 160),
            nn.ReLU(inplace=True)
        )

        self.mouth_layer = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(980, 320),
            nn.ReLU(inplace=True),
            nn.Dropout(p=proj_drop_rate),
            nn.Linear(320, 160),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x.shape: [B, N, 2, 30, 7, 7]
        data = torch.empty(x.shape[0], 30, self.parallel, 7, 7).cuda()
        # data: [B, 30, N, 7, 7]，这里将N和30调换位置是为了输入后续的3DConv进行卷积
        for idx in range(self.parallel):
            temp = x[:, idx].transpose(1, 2)
            # temp: [B, 30, 2, 7, 7]，构建一个5D数据作为Conv3D的输入
            temp = self.OFConv_List[idx](temp).squeeze()
            # temp: [B, 30, 7, 7]
            data[:, :, idx] = temp
        x = self.DWConv(data)  # x: [B, 30, 49]
        o_eye = x[:, :10]
        o_mou = x[:, 10:]
        o_eye = self.eyebrow_encoder(o_eye)
        o_mou = self.mouth_encoder(o_mou)

        o_eye = self.eyebrow_layer(o_eye)
        o_mou = self.mouth_layer(o_mou)
        # o_eye, o_mou: [B, 160]
        return o_eye, o_mou


if __name__ == "__main__":
    test_vector = torch.ones(32, 9, 2, 30, 7, 7).cuda()
    model = GraphLearningModel(9).to(torch.device("cuda:0"))

    eyebrow, mouth = model(test_vector)

    print(eyebrow.shape)
    print(mouth.shape)
