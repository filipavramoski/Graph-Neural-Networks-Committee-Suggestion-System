This thesis presents an automated system for recommending a thesis committee using a graph neural network (GNN). The system addresses the challenge of identifying a mentor and two other committee members (C2 and C3) while paying attention to their field of work, previous thesis mentorships, and personal research work to ensure that they are relevant to the given thesis. The approach creates a heterogeneous graph that represents the connections between professors and theses, using a relational graph convolutional network (R-GCN) implemented in PyTorch Geometric. The node features for a professor include the professor’s field of work, while the nodes for a given thesis include the thesis’ title and description. The model uses a three-layer R-GCN architecture with 3 “output heads” corresponding to the three types of output offered by the model: mentor suggestion, second member, and third member suggestion for the committee. When a student enters a thesis title, the system adds that thesis as a new node and connects it to professors in the graph network, creating edges that enable efficient message transmission. The GNN learns to predict compatibility scores between thesis and potential mentors by analyzing patterns in the academic network.

Graph neural networks (GNNs) are a class of deep neural networks specialized for processing data structured as graphs. Unlike traditional neural networks that work with regular structures (images, sequences), GNNs can work with irregular, unstructured graphs where the number of vertices and edges varies between different examples.

The system is trained on two datasets that use real academic data -> Committee dataset and Research dataset

The graph contains two types of nodes:
  · Professor nodes: 98 nodes - Representing professors at FINKI - Each professor has an embedding vector of 384 dimensions - 80 professors have content-based embeddings, while 18 have an embedding obtained based on their mentored thesis’.
  · Thesis nodes: 5,858 nodes - Represent theses and research papers - Each thesis has an embedding vector of 384 dimensions - Generated from title + abstract with SentenceTransformer
  
The committee recommendation system is based on Relational Graph Convolutional Networks (R-GCN) with a multi-task learning architecture. The main components are:
  · Heterogeneous graph with 5 types of relations (mentor, C2, C3, research, collaboration)
  · R-GCN layers for message passing through different types of connections
  · Three separate prediction heads for each role (mentor, C2, C3)
  · Edge masking strategy for realistic evaluation of generalizatio

  This system aims to provide a practical solution that will facilitate the process of forming committees at faculties, while guaranteeing high quality of evaluation, fair distribution of the burden among professors, and a better match between the expertise of the committee and the topic of the thesis.
