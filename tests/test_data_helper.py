import data_helper

# look up
data_helper.create_dictdb_from_file(file_path='/Users/sevgili/Ozge-PhD/DBpedia-datasets/outputs/en/intersection_nodes_lookup.txt',
                                    db_path='/Users/sevgili/Ozge-PhD/DBpedia-datasets/outputs/databases/intersection_nodes_lookup.db')

# long abstracts
data_helper.create_dictdb_from_file(file_path='/Users/sevgili/Ozge-PhD/DBpedia-datasets/long-abstracts/en/long_abstracts_en_dict.txt',
                                    db_path='/Users/sevgili/Ozge-PhD/DBpedia-datasets/outputs/databases/long_abstracts.db')

# labels
data_helper.create_dictdb_from_file(file_path='/Users/sevgili/Ozge-PhD/DBpedia-datasets/labels/en/labels_en_dict.txt',
                                    db_path='/Users/sevgili/Ozge-PhD/DBpedia-datasets/outputs/databases/labels.db')

data_helper.create_db_from_dictdb(lookup_db_path='/Users/sevgili/Ozge-PhD/DBpedia-datasets/outputs/databases/intersection_nodes_lookup.db',
                                  longabs_db_path='/Users/sevgili/Ozge-PhD/DBpedia-datasets/outputs/databases/long_abstracts.db',
                                  labels_db_path='/Users/sevgili/Ozge-PhD/DBpedia-datasets/outputs/databases/labels.db',
                                  db_name='/Users/sevgili/Ozge-PhD/DBpedia-datasets/outputs/databases/graph.db')
