# kb2vec/supervised 

This project provides an alternative use of graph embeddings in Entity Disambiguation. The input of the feedforward neural network is the concatenation of context vector, span vector, entity graph embeddings, and long abstract (of corresponding entity) vector. 

Installation
-----------

```
cd supervised
pip install -r requirements.txt
```

Set up
-----------

1 -  Creating entity graph embeddings:

From DBpedia datasets (https://wiki.dbpedia.org/develop/datasets/downloads-2016-10), long 
abstracts, labels, and page links files are downloaded. Using `../construct_graph.py`, the graph is contructed.
Page links are the inputs of DeepWalk  algorithm (https://github.com/phanein/deepwalk) to create entity graph embeddings.

2 - FFNN ablation test:

For generating negative samples use negative_sampling.py and for training nn.py.The relative inputs are commented out for ablation test.

