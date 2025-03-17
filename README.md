# StyleMind-GNN
Building a graph representation for outfit generation using GNNs and Polyvore dataset


## StyleMind-GNN: Outfit Generation Using Graph Neural Networks
### Abstract
In this project, we propose a two-phase approach to outfit generation using Graph Neural Networks (GNNs). The goal is to represent garments in an embedding space where garments that frequently appear together in outfits are mapped closer to each other. In Phase 1, we train a GNN-based model using the Polyvore dataset, employing both homogeneous GNNs (such as GCN and GAT) and more complex heterogeneous GNNs (such as R-GNN, HAN, and other H-GNNs). The performance of these models will be evaluated using a Graph Autoencoder (GAE) with a reconstruction loss. In Phase 2, we extend the learned garment embeddings to an inductive setting by constructing a garment graph using image-based embeddings extracted from a ResNet model. We then apply a GNN model to learn garment representations and predict outfit compatibility in the absence of predefined outfit relationships. Finally, we propose improvements by incorporating hierarchical learning to refine the graph construction process. Our work aims to advance automated outfit generation by leveraging GNNs for both transductive and inductive learning.
