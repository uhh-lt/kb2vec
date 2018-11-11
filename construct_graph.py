import networkx as nx
import matplotlib.pyplot as plt
import logging
import codecs
from sqlitedict import SqliteDict


class Graph:
    def __init__(self, logfile='output.log'):
        self._G = nx.Graph()
        # create logger
        self._logger = logging.getLogger('construct_graph')
        self._logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(logfile)
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self._logger.addHandler(fh)

    # takes three dictionaries:
    # url_ids - keys contain the urls, values are the unique ids of these urls.
    # url_longabstracts - keys contain the urls, values are the long abstracts (texts) of them.
    # url_labels - keys, again, are the urls, values contain the title of them.
    # unique ids are used to create node and other properties are used as the attributes of the nodes.
    def create_nodes_from_dict(self, url_longabstracts, url_labels, url_ids):
        urls = url_ids.keys()

        count = 0
        for url in urls:
            # long abstract is the list of tokens.
            long_abstract = url_longabstracts[url]
            # title is the list of tokens/token.
            title = url_labels[url]
            # node id is the integer value.
            node_id = url_ids[url]

            # id, url, long abstract (text), and title are attributes.
            self._G.add_node(node_id, id=node_id, url=url, long_abstract=long_abstract, title=title)
            if count % 100000 == 0:
                self._logger.info(str(count) + ' nodes are processed..')

            count += 1

    # subnodes is the list of nodes, it is used to create nodes from sublist and have a subgraph.
    def create_nodes_from_db(self, longabsdb_path, labelsdb_path, lookupdb_path, subnodes=False):
        longabsdb = SqliteDict(longabsdb_path, autocommit=False)
        labelsdb = SqliteDict(labelsdb_path, autocommit=False)
        lookupdb = SqliteDict(lookupdb_path, autocommit=False)

        if subnodes:
            urls = subnodes
        else:
            urls = lookupdb.keys()

        count = 0
        for url in urls:
            # long abstract is string.
            long_abstract = longabsdb[url]
            # title is string.
            title = labelsdb[url]
            # node id is the integer value.
            node_id = int(lookupdb[url])

            # id, url, long abstract (text), and title are attributes.
            self._G.add_node(node_id, id=node_id, url=url, long_abstract=long_abstract, title=title)
            if count % 100000 == 0:
                self._logger.info(str(count) + ' nodes are processed..')

            count += 1

    # takes file a parameter:
    # file contains edge at each line, like (1, 2).
    def create_edges_from_file(self, path):
        count = 0

        file = codecs.open(path, 'r')
        line = file.readline()

        while line != '':
            nodes = line.split()
            line = file.readline()

            self._G.add_edge(int(nodes[0]), int(nodes[1]))

            if count % 100000 == 0:
                self._logger.info(str(count) + ' edges are processed..')

            count += 1

    def create_edges_from_list(self, edges):
        self._G.add_edges_from(edges)

    def write_graph(self, path):
        nx.write_gpickle(self._G, path)

    def load_graph(self, path):
        self._G = nx.read_gpickle(path)

    def draw(self):
        nx.draw(self._G, with_labels=True, font_weight='bold')
        plt.show()







