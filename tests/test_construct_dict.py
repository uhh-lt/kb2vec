import construct_graph
import preprocess
'''
url_lookup = preprocess.read_lookup(path='/Users/sevgili/Ozge-PhD/DBpedia-datasets/outputs/en/intersection_nodes_lookup.txt')
urls = url_lookup.keys()
print('url lookup is read...', len(urls))

labels_dict = preprocess.read_dict(path='/Users/sevgili/Ozge-PhD/DBpedia-datasets/labels/en/labels_en_dict.txt')
print('labels are read...', len(labels_dict.keys()))

filtered_labels = dict(filter(lambda i:i[0] in urls, labels_dict.items()))
print('labels are filtered...', len(filtered_labels.keys()))
del labels_dict

longabs_dict = preprocess.read_dict(path='/Users/sevgili/Ozge-PhD/DBpedia-datasets/long-abstracts/en/long_abstracts_en_dict.txt')
print('long abstracts are read...', len(longabs_dict.keys()))

filtered_longabs = dict(filter(lambda i:i[0] in urls, longabs_dict.items()))
print('longabs are filtered...', len(filtered_longabs.keys()))
del longabs_dict


graph = construct_graph.Graph(logfile='/Users/sevgili/Ozge-PhD/DBpedia-datasets/outputs/en/construct_graph.log')
graph.create_nodes_from_dict(url_longabstracts=filtered_longabs, url_labels=filtered_labels, url_ids=url_lookup)
print('nodes are created...')

del url_lookup, filtered_longabs, filtered_labels
print('dictionaries are deleted...')


graph.write_graph(path='/Users/sevgili/Ozge-PhD/DBpedia-datasets/outputs/en/intersection_graph_nodes.gpickle')
print('graph is written...')

edges = preprocess.read_edges(path='/Users/sevgili/Ozge-PhD/DBpedia-datasets/outputs/en/intersection_edges_ids.txt')
print('edges are read...')

#graph.create_edges_from_file(path='/Users/sevgili/Ozge-PhD/DBpedia-datasets/outputs/en/intersection_edges_ids.txt')
#print('edges are created...')

graph.write_graph(path='/Users/sevgili/Ozge-PhD/DBpedia-datasets/outputs/en/intersection_graph.gpickle')
print('graph is written...')

graph.draw()
'''
# for subgraph
graph = construct_graph.Graph(logfile='/Users/sevgili/Ozge-PhD/DBpedia-datasets/outputs/en/subset/construct_graph.log')
subnodes = preprocess.read_list(path='/Users/sevgili/Ozge-PhD/DBpedia-datasets/outputs/en/subset/1000_nodelist_url.txt')

graph.create_nodes_from_db(longabsdb_path='/Users/sevgili/Ozge-PhD/DBpedia-datasets/outputs/databases/long_abstracts.db',
                            labelsdb_path='/Users/sevgili/Ozge-PhD/DBpedia-datasets/outputs/databases/labels.db',
                            lookupdb_path='/Users/sevgili/Ozge-PhD/DBpedia-datasets/outputs/databases/intersection_nodes_lookup.db',
                            subnodes=subnodes)
print('nodes created..')

edges = preprocess.read_edges(path='/Users/sevgili/Ozge-PhD/DBpedia-datasets/outputs/en/subset/1000_edgelist.txt')
print('edges are read...')

graph.create_edges_from_list(edges=edges)
print('edges are created...')

graph.write_graph(path='/Users/sevgili/Ozge-PhD/DBpedia-datasets/outputs/en/subset/1000_graph.gpickle')
print('graph is written...')

graph.draw()

