from supervised import negative_sampling

import codecs


def check_written_file(contexts_r, phrases_r, contexts, phrases):
    is_equal = True

    for phrase in phrases:
        entity, beg, end, ref_context, url = phrase[0], phrase[1], phrase[2], phrase[3], phrase[4]
        try:
            context = contexts[ref_context]
            context_r = contexts_r[entity+str(beg)+str(end)+url+context]
            phrase_r = phrases_r[entity+str(beg)+str(end)+url+context]

            is_equal &= (context == context_r) & (entity == phrase_r[0]) & (beg == phrase_r[1]) & (end == phrase_r[2])
            if not is_equal:
                print(entity, url, beg, end)
                break
        except KeyError:
            print("Warning: not found", ref_context)

    return is_equal

'''
# creating negative samples 
contexts_r, phrases_r = negative_sampling.read_samples('/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/csv/positive_samples.tsv')
print('positive samples are read..')
negative_samples = negative_sampling.create_negative_samples_with_positive_samples(urls_db='/Users/sevgili/Ozge-PhD/DBpedia-datasets/outputs/databases/intersection_nodes_lookup.db',
                                           contexts=contexts_r, phrases=phrases_r)

print(len(negative_samples))
print('Writing started..')
negative_sampling.write_negative_samples_with_positive_samples(positive_negatives=negative_samples,
                       path='/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/csv/negative_samples_with_positives_V2.tsv')

'''
# check samples
#positive_negatives, count = negative_sampling.\
#    read_negative_samples_with_positive_samples(path='/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/csv/negative_samples_with_positives.tsv')

#print(len(positive_negatives), count)
''' '''

# closest sampling
positive_negatives = negative_sampling.\
    read_negative_samples_with_positive_samples(path='/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/csv/negative_samples_with_positives_V2.tsv')
filtered_samples = negative_sampling.\
    filter_negative_samples_closest(positives_negatives=positive_negatives,
                                                   url_db='/Users/sevgili/Ozge-PhD/DBpedia-datasets/outputs/databases/intersection_nodes_lookup.db',
                                                   pagerank_db='/Users/sevgili/Ozge-PhD/DBpedia-datasets/outputs/databases/pagerank.db', n=3)
print('starts to write')
negative_sampling.write_negative_samples_with_positive_samples(positive_negatives=filtered_samples,
                                             path='/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/csv/negative_samples_filtered_closest_3.tsv')
''' 
# random sampling

positive_negatives = negative_sampling.read_negative_samples_with_positive_samples(path='/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/csv/negative_samples_with_positives_V2.tsv')
filtered_samples = negative_sampling.filter_negative_samples_randomly(positives_negatives=positive_negatives,
                                                   url_db='/Users/sevgili/Ozge-PhD/DBpedia-datasets/outputs/databases/intersection_nodes_lookup.db',
                                                    n=5)
print('starts to write')
negative_sampling.write_negative_samples_with_positive_samples(positive_negatives=filtered_samples,
                                             path='/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/csv/negative_samples_filtered_randomly_5.tsv')

'''
#input_ttl_fpaths = ["/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/ttl/dbpedia-spotlight-nif.ttl"]
#input_ttl_fpaths = ["/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/ttl/kore50-nif.ttl"]
#input_ttl_fpaths = ["/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/ttl/Reuters-128.ttl"]

'''
input_ttl_fpaths = ["/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/ttl/dbpedia-spotlight-nif.ttl",
                    "/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/ttl/kore50-nif.ttl",
                    "/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/ttl/Reuters-128.ttl"]

for input_ttl_fpath in input_ttl_fpaths:
    in_ttl = codecs.open(input_ttl_fpath, "r", "utf-8")

    input_ttl = in_ttl.read()
    graph, contexts, phrases = parse_d2kb_ttl(input_ttl)

    print(phrases)
    print(contexts)

    #write_positive_samples(contexts=contexts, phrases=phrases,
    #                       path='/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/csv/positive_samples.tsv')
'''

