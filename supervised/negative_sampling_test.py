from supervised import negative_sampling
import ttl
import codecs
from sqlitedict import SqliteDict

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


def get_statistics_true_url(positives_negatives, urls_db):
    db = SqliteDict(urls_db, autocommit=False)
    urls = list(db.keys())

    file = codecs.open('candidates_without_true_name1_.tsv', 'a')
    count_exist = 0
    count_all = 0
    count_not_included = 0

    for positive_negative in positives_negatives:
        entity, beg, end, true_url, context, negative_samples = positive_negative

        samples = list()
        for negative_sample in negative_samples:
            samples.append(negative_sample.strip())

        if true_url in samples:
            count_exist += 1
        elif true_url in urls:
            file.write(str(entity) + '\t' + str(true_url) + '\n')
        else:
            count_not_included += 1

        count_all += 1

    print(count_exist)
    print(count_all)
    print(count_not_included)
    return float(count_exist)/count_all


''' 
# creating negative samples 
contexts_r, phrases_r = negative_sampling.read_samples('/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/csv/positive_samples_new.tsv')
print('positive samples are read..')
negative_samples = negative_sampling.create_negative_samples_with_positive_samples(urls_db='/Users/sevgili/Ozge-PhD/DBpedia-datasets/outputs/databases/intersection_nodes_lookup.db',
                                           contexts=contexts_r, phrases=phrases_r)

print(len(negative_samples))
print('Writing started..')
negative_sampling.write_negative_samples_with_positive_samples(positive_negatives=negative_samples,
                       path='/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/csv/negative_samples_with_positives_new.tsv')
'''
''' '''
# creating candidates
contexts_r, phrases_r = negative_sampling.read_samples('/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/csv/positive_samples_new.tsv')
print('positive samples are read..')
negative_samples = negative_sampling.create_candidates(urls_db='/Users/sevgili/Ozge-PhD/DBpedia-datasets/outputs/databases/intersection_nodes_lookup.db',
                                           contexts=contexts_r, phrases=phrases_r)

print(len(negative_samples))
print('Writing started..')
negative_sampling.write_negative_samples_with_positive_samples(positive_negatives=negative_samples,
                       path='/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/candidates/candidate1_big.tsv')


''' 
# get statistics
positive_negatives = negative_sampling.\
    read_negative_samples_with_positive_samples(path='/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/candidates/candidate1.tsv')
print(get_statistics_true_url(positive_negatives, urls_db='/Users/sevgili/Ozge-PhD/DBpedia-datasets/outputs/databases/intersection_nodes_lookup.db'))
'''

# check samples
#positive_negatives, count = negative_sampling.\
#    read_negative_samples_with_positive_samples(path='/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/csv/negative_samples_with_positives.tsv')

#print(len(positive_negatives), count)
''' 
# closest sampling
positive_negatives = negative_sampling.\
    read_negative_samples_with_positive_samples(path='/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/csv/negative_samples_with_positives.tsv')
filtered_samples = negative_sampling. \
    filter_negative_samples_closest_with_scores(positives_negatives=positive_negatives,
                                                   url_db='/Users/sevgili/Ozge-PhD/DBpedia-datasets/outputs/databases/intersection_nodes_lookup.db',
                                                   pagerank_db='/Users/sevgili/Ozge-PhD/DBpedia-datasets/outputs/databases/pagerank.db', n=10)
print('starts to write')
negative_sampling.write_negative_samples_with_positive_samples_with_scores(positive_negatives=filtered_samples,
                                             path='/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/csv/with_scores/negative_samples_filtered_closest_10.tsv')
'''
''' 
# random sampling

positive_negatives = negative_sampling.read_negative_samples_with_positive_samples(path='/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/csv/negative_samples_with_positives_new.tsv')
filtered_samples = negative_sampling.filter_negative_samples_randomly(positives_negatives=positive_negatives,
                                                   url_db='/Users/sevgili/Ozge-PhD/DBpedia-datasets/outputs/databases/intersection_nodes_lookup.db',
                                                    n=10)
print('starts to write')
negative_sampling.write_negative_samples_with_positive_samples(positive_negatives=filtered_samples,
                                             path='/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/csv/negative_samples_filtered_randomly_10_big.tsv')

'''
''' 
# completely random
contexts_r, phrases_r = negative_sampling.read_samples('/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/csv/positive_samples.tsv')
samples = negative_sampling.create_completely_random(urls_db='/Users/sevgili/Ozge-PhD/DBpedia-datasets/outputs/databases/intersection_nodes_lookup.db',
                                           contexts=contexts_r, phrases=phrases_r, n=5)

print('starts to write')
negative_sampling.write_negative_samples_with_positive_samples(positive_negatives=samples,
                                             path='/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/csv/negative_samples_completely_random_5.tsv')
'''
''' 
# closest sampling with scores and similarity
positive_negatives = negative_sampling.\
    read_negative_samples_with_positive_samples(path='/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/csv/negative_samples_with_positives_new.tsv')

sims_scores = negative_sampling.get_negative_samples_similarity_and_scores(positives_negatives=positive_negatives,
                                                                           url_db='/Users/sevgili/Ozge-PhD/DBpedia-datasets/outputs/databases/intersection_nodes_lookup.db',
                                                                           graphembed='/Users/sevgili/Ozge-PhD/DBpedia-datasets/outputs/nodes.embeddings',
                                                                           pagerank_db='/Users/sevgili/Ozge-PhD/DBpedia-datasets/outputs/databases/pagerank.db')

print('starts to write')
negative_sampling.write_negative_samples_with_positive_samples_with_scores(positive_negatives=sims_scores,
                                             path='/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/csv/with_scores/negative_samples_sims_scores_new.tsv')
'''
''' 
# prune closest

positive_negatives = negative_sampling.read_negative_samples_with_positive_samples_with_scores(path='/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/csv/with_scores/negative_samples_sims_scores_new.tsv')
print('positive_negatives is read')
pruned_samples = negative_sampling.prune_most_closest(positives_negatives=positive_negatives, n=10)
print('samples is pruned')
negative_sampling.write_negative_samples_with_positive_samples(positive_negatives=pruned_samples,
                                             path='/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/csv/negative_samples_filtered_closest_pruned_10_big.tsv')
'''


def ttl2csv(list_of_paths, write_path):
    for input_ttl_fpath in list_of_paths:
        in_ttl = codecs.open(input_ttl_fpath, "r", "utf-8")

        input_ttl = in_ttl.read()
        graph, contexts, phrases = negative_sampling.parse_d2kb_ttl(input_ttl)

        print(phrases)
        print(contexts)

        negative_sampling.write_positive_samples(contexts=contexts, phrases=phrases,
                                                 path=write_path)


input_ttl_fpaths = ["/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/ttl/dbpedia-spotlight-nif.ttl",
                    "/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/ttl/kore50-nif.ttl",
                    "/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/ttl/Reuters-128.ttl"]

new_input_ttl_fpaths = ["/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/ttl/RSS-500.ttl",
                    "/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/ttl/News-100.ttl"]

#ttl2csv(new_input_ttl_fpaths,
#        write_path="/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/csv/positive_samples_new.tsv")

