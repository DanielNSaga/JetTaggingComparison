import torch
import torch_geometric
from torch_geometric.nn import BatchNorm, MessagePassing


class ParticleStaticEdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * in_channels, out_channels[0], bias=False),
            BatchNorm(out_channels[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels[0], out_channels[1], bias=False),
            BatchNorm(out_channels[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels[1], out_channels[2], bias=False),
            BatchNorm(out_channels[2]),
            torch.nn.ReLU()
        )

    def forward(self, x, edge_index):
        # Kaller propagate med x som tuple for å få x_i og x_j
        return self.propagate(edge_index, x=(x, x))

    def message(self, x_i, x_j):
        tmp = torch.cat([x_i, x_j - x_i], dim=1)
        return self.mlp(tmp)

    def update(self, aggr_out):
        return aggr_out


class ParticleDynamicEdgeConv(ParticleStaticEdgeConv):
    def __init__(self, in_channels, out_channels, k=7):
        super().__init__(in_channels, out_channels)
        self.k = k
        self.skip_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, out_channels[2], bias=False),
            BatchNorm(out_channels[2]),
        )
        self.act = torch.nn.ReLU()

    def forward(self, pts, fts, batch=None):
        # Bygg knn-graph basert på posisjonene
        edges = torch_geometric.nn.knn_graph(pts, self.k, batch, loop=False)
        aggrg = super().forward(fts, edges)
        x = self.skip_mlp(fts)
        return self.act(aggrg + x)


class ParticleNet(torch.nn.Module):
    def __init__(self, settings):
        super().__init__()
        previous_output_shape = settings["input_features"]
        self.input_bn = BatchNorm(settings["input_features"])

        self.conv_process = torch.nn.ModuleList()
        for K, channels in settings["conv_params"]:
            self.conv_process.append(ParticleDynamicEdgeConv(previous_output_shape, channels, k=K))
            previous_output_shape = channels[-1]

        self.fc_process = torch.nn.ModuleList()
        for drop_rate, units in settings["fc_params"]:
            self.fc_process.append(torch.nn.Sequential(
                torch.nn.Linear(previous_output_shape, units),
                torch.nn.Dropout(p=drop_rate),
                torch.nn.ReLU()
            ))
            previous_output_shape = units

        self.output_mlp_linear = torch.nn.Linear(previous_output_shape, settings["output_classes"])

    def forward(self, batch):
        x = self.input_bn(batch.x)
        for layer in self.conv_process:
            x = layer(batch.pos, x, batch.batch)
        x = torch_geometric.nn.global_mean_pool(x, batch.batch, batch.num_graphs)

        for layer in self.fc_process:
            x = layer(x)
        return self.output_mlp_linear(x)
