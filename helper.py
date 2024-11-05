import torch
from torch_geometric.data import Data

# Create edges for the grid graph
def create_edges() -> tuple[torch.Tensor, torch.Tensor]:
    horizontal_edges = []
    for i in range(28):  
        for j in range(27):  
            horizontal_edges.append([i * 28 + j, i * 28 + j + 1])

    vertical_edges = []
    for i in range(27):  
        for j in range(28):  
            vertical_edges.append([i * 28 + j, (i + 1) * 28 + j])

    diagonal_right_edges = []
    for i in range(27):  
        for j in range(27):  
            diagonal_right_edges.append([i * 28 + j, (i + 1) * 28 + j + 1])

    diagonal_left_edges = []
    for i in range(27):  
        for j in range(1, 28):  
            diagonal_left_edges.append([i * 28 + j, (i + 1) * 28 + j - 1])

    edge_indices = horizontal_edges + vertical_edges + diagonal_right_edges + diagonal_left_edges
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    
    edge_type = torch.tensor([0]*len(horizontal_edges) + [1]*len(vertical_edges) + [2]*len(diagonal_right_edges) + [3]*len(diagonal_left_edges), dtype=torch.long)
    return edge_index, edge_type

# Create a PyG Data object for a given list and label
def create_data(row: list[int], label: int) -> Data:
    # Convert row to tensor and reshape
    x = torch.tensor(row.reshape(-1, 1), dtype=torch.float)
    
    # Get indices of non-zero nodes
    non_zero_indices = torch.where(x > 0)[0]
    
    # Filter x to keep only non-zero nodes
    x_filtered = x[non_zero_indices]
    
    # Get the original edges
    edge_index, edge_type = create_edges()
    
    # Create a mapping from old indices to new indices
    idx_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(non_zero_indices)}
    
    # Filter and remap edges
    valid_edges_mask = []
    new_edge_index = []
    new_edge_type = []
    
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[:, i]
        # Check if both source and destination nodes are in our non-zero set
        if src.item() in idx_map and dst.item() in idx_map:
            valid_edges_mask.append(True)
            new_edge_index.append([idx_map[src.item()], idx_map[dst.item()]])
            new_edge_type.append(edge_type[i].item())
        else:
            valid_edges_mask.append(False)
    
    # Convert to tensor
    new_edge_index = torch.tensor(new_edge_index, dtype=torch.long).t()
    new_edge_type = torch.tensor(new_edge_type, dtype=torch.long)
    
    # Create the filtered data object
    data = Data(
        x=x_filtered, 
        edge_index=new_edge_index, 
        edge_type=new_edge_type, 
        y=torch.tensor([label], dtype=torch.long)
    )
    return data