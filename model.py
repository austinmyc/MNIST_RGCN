from torch_geometric.nn import RGCNConv, GraphNorm, global_mean_pool
import torch
from torch.nn import Linear
import torch.nn.functional as F

class MNISTRGCN(torch.nn.Module):
    def __init__(self, num_features=1, hidden_channels=128, num_classes=10, num_relations=4):
        super(MNISTRGCN, self).__init__()
        torch.manual_seed(2024)
        self.conv1 = RGCNConv(num_features, hidden_channels, num_relations=num_relations)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels*2, num_relations=num_relations)
        self.bn1 =  GraphNorm(hidden_channels)
        self.bn2 = GraphNorm(hidden_channels*2)
        self.lin = Linear(hidden_channels*2, num_classes)

    def forward(self, x, edge_index, edge_type, batch):
        x = self.conv1(x, edge_index, edge_type)
        x = self.bn1(x)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_type)
        x = self.bn2(x)
        x = x.relu()
        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x
    
class Trainer:
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def train(self, train_loader):
        self.model.train()
        total_loss = 0
        for data in train_loader:
            self.optimizer.zero_grad()
            out = self.model(data.x, data.edge_index, data.edge_type, data.batch)
            loss = self.criterion(out, data.y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)
    
    def test(self, loader):
        self.model.eval()
        correct = 0
        for data in loader:
            out = self.model(data.x, data.edge_index, data.edge_type, data.batch)
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
        return correct / len(loader.dataset)
    
    