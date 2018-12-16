import codecs
from rdflib import Graph


def open_triples(path):
    return codecs.open(path, "r", "utf-8")


def read_triples(path):
    return codecs.open(path, "r", "utf-8").read()


# takes the path of the .ttl file and returns the dictionary
# whose keys are the subject and values are the object.
def read_triples_manuel(path):
    result = dict()
    file = codecs.open(path, "r", "utf-8")

    line = file.readline()
    while line != '':
        if line.startswith('<'):
            splitted_line = line.split()
            subject = splitted_line[0][1:-1]
            object = splitted_line[2][1:-1]

            result[subject] = object
        line = file.readline()

    return result


def parse_triples(input_triple, input_format='n3'):
    g = Graph()
    return g.parse(data=input_triple, format=input_format)


# takes the rdflib graph and writes its subject and object
# to the given file.
def write_triple(input_triple, path):
    file = open(path, 'w')

    count = 0
    print('writing is started...')
    print(len(input_triple))
    for subj, pred, obj in input_triple:
        file.write(str(subj) + ' ' + str(obj) + '\n')
        if count%100000 == 0:
            print(count, 'nodes are written..')

        count += 1
    print('count', count)
    file.close()


def triple2dict(input_triple):
    result = dict()
    for subj, pred, obj in input_triple:
        result[str(subj)] = str(obj)

    return result


# nodes_ids is dictionary: keys are urls, values are ids of them.
# edges is list of tuples where two nodes have an edge.
def filter_edges_by_nodes(nodes_ids, edges):

    filtered_edges = list()
    filtered_edges_ids = list()

    for nodes in edges:
        node1, node2 = nodes[0], nodes[1]

        try:
            id1, id2 = nodes_ids[node1], nodes_ids[node2]
        except KeyError:
            continue

        filtered_edges.append((node1, node2))
        filtered_edges_ids.append((id1, id2))

    return filtered_edges, filtered_edges_ids


def read_dict(path):
    result = dict()

    file = codecs.open(path, 'r')
    line = file.readline()

    while line != '':
        splitted = line.split()
        line = file.readline()

        try:
            result[str(splitted[0])] = splitted[1:]
        except IndexError:
            continue

    return result


def read_lookup(path):
    result = dict()

    file = codecs.open(path, 'r')
    line = file.readline()

    while line != '':
        splitted = line.split()
        line = file.readline()

        try:
            result[str(splitted[0])] = int(splitted[1])
        except IndexError:
            continue

    return result


def read_edges(path):
    edges = list()

    file = codecs.open(path, 'r')
    line = file.readline()

    while line != '':
        splitted = line.split()
        line = file.readline()

        edges.append((int(splitted[0]), int(splitted[1])))

    return edges


def write_edges(edgelist, path):
    file = codecs.open(path, 'w')

    for edge in edgelist:
        file.write(str(edge[0]) + ' ' + str(edge[1]) + '\n')

    file.close()

def read_list(path):
    data = list()

    file = codecs.open(path, 'r')
    line = file.readline()

    while line != '':
        splitted = line.split()
        data.append(splitted[0])
        line = file.readline()

    return data


