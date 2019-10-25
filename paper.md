# 论文囊(bu)括(wan)计划
[torch_geometric.nn](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html) 囊括了许多有关图上深度学习方法的代码，本人计划补完这些论文，这里是囊(bu)括(wan)计划的进度跟踪文档。

- graph convolution
    - Semi-Supervised Classification with Graph Convolutional Networks
    - [arxiv 1609.02907](https://arxiv.org/abs/1609.02907)
    - 图卷积最经典的论文。


- chebyshev spectral graph convolution
    - Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering
    - [arxiv 1606.09375](https://arxiv.org/abs/1606.09375)
    - 相较于上面的图卷积，在利用 Chebyshev 多项式逼近卷积的时候使用了更高的阶数。

- GraphSAGE
    - Inductive Representation Learning on Large Graphs
    - [arxiv  1706.02216](https://arxiv.org/abs/1706.02216)
    - 大概就是把所有近邻节点的特征平均得到自身的新特征，感觉不适合deep learning。