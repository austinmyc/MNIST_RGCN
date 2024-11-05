# MNIST RGCN

This is a simple implementation of the [RGCN model](https://arxiv.org/pdf/1703.06103) for the MNIST dataset using [PyG](https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.nn.conv.RGCNConv.html).

Details:
- Each picture is turned into a graph, where each non-zero pixel is a node, and the pixel's value is the node feature.
- 4 types of relationship are defined: vertical, horizonal, left and right diagonal.
- Without careful tuning, the model was trained with 1600 datapoints only and attained 84.5% accruacy on test set with 10000 datapoints.

Note:
- Only included the test data here due to file size
