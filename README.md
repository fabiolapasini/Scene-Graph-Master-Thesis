# Scene-Graph-Master-Thesis

Master's Thesis of Fabiola Pasini \
University of Parma  &
Technical University of Munich 

# Summary
Graph Neural Networks have become an interesting field of reaserch in the last years and diffent variants of the original GNN architecture have been proposed. The most famous works  focused mainly on the nodes and their features, not considering the edges between them. 
Scene graphs are particular kind of graph in which the nodes are objects visible inside the scene and the edges are the relationships between them. 
Scene graphs play an important role in scene understanding and scene representation, so in my thesis I implemented a GNN architecture that can learn not only node features but  also dege features, in order to be able to predict graphs that fully describe a scene considering also the links between the objects.

# Main libraries used:
PyTorch \
Pytorch Geometric \
Trimesh \
pyximport 
