import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TopKPooling, SAGEConv, EdgePooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.data import Data, DataLoader, Batch


class Net(nn.Module):
    def __init__(self, n_feats, device = 'cpu'):
        super(Net, self).__init__()
        self.atom_embedding = nn.Linear(n_feats, n_feats)
        self.conv1 = SAGEConv(n_feats, n_feats)
        self.pool1 = EdgePooling(n_feats)
        self.conv2 = SAGEConv(n_feats, n_feats)
        self.pool2 = EdgePooling(n_feats)
        self.linear = torch.nn.Linear(2*n_feats, 1)
        self.device = device

    def forward(self, x, edge_index, batch=None):
        if len(x.shape) == 3: #NEEDED FOR EXAI, NOTICE THAT IT MUST BE ONLY ONE MOL
            data_list = []
            for x_i, edge_index_i in zip(x, edge_index):
                data_list.append(Data(x=x_i, edge_index=edge_index_i))
            data = Batch.from_data_list(data_list).to(self.device)
            x = data.x
            batch = data.batch
            edge_index = data.edge_index
        shape = x.shape
        x = x.reshape(-1,shape[-1])
        x = self.atom_embedding(x)
        x = x.reshape(shape)
        x = x.squeeze(1)
        
        x = F.relu(self.conv1(x, edge_index))

        x, edge_index, batch, _ = self.pool1(x, edge_index, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
     
        x, edge_index, batch, _ = self.pool2(x, edge_index, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = x1 + x2
        x = self.linear(x)    
        #x = torch.sigmoid(x)
        x = x.squeeze(1)
        return x
