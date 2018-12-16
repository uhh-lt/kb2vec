import construct_graph
import preprocess


# for subgraph
graph = construct_graph.Graph(logfile='../datasets/subset/construct_graph.log')
subnodes = preprocess.read_list(path='../datasets/subset/1000_nodelist_url.txt')

graph.create_nodes_from_db(longabsdb_path='../datasets/subset/1000_long_abstracts.db',
                            labelsdb_path='../datasets/subset/1000_labels.db',
                            lookupdb_path='../datasets/subset/1000_nodes_lookup.db',
                            subnodes=subnodes)
print('nodes created..')

edges = preprocess.read_edges(path='../datasets/subset/1000_edgelist.txt')
print('edges are read...')

graph.create_edges_from_list(edges=edges)
print('edges are created...')

graph.write_graph(path='../datasets/subset/1000_graph_sub.gpickle')
print('graph is written...')

graph.draw()

