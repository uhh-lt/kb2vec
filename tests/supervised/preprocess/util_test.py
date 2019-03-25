from supervised.preprocess.util import FetchFilteredCoreferencedCandEntities, load_url2graphid
from nltk.tokenize import word_tokenize
from supervised.negative_sampling import parse_d2kb_ttl
import codecs


# not_match_entity = 13, total = 331 spotlight
# not_match_entity = 22, total = 1000 rss-500
# not_match_entity = 57, total = 880 reuters
# not_match_entity = 3, total = 144 kore50-nif
# not_match_entity = 75, total = 1655 News-100
def test_index_span():

    context = "In the first study, intended to measure a person’s short-term emotional reaction to gossiping, " \
                  "140 men and women, primarily undergraduates, were asked to talk about a fictional person either " \
                  "positively or negatively."
    beg = 124
    end = 138
    entity = 'undergraduates'
    chunk_words = word_tokenize(context)
    left = chunk_words.index(entity)
    right = left + len(word_tokenize(entity))
    print(' '.join(chunk_words[left:right]))

    input_ttl_fpath = "/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/ttl/News-100.ttl"
    in_ttl = codecs.open(input_ttl_fpath, "r", "utf-8")

    input_ttl = in_ttl.read()
    graph, contexts, phrases = parse_d2kb_ttl(input_ttl)
    phrases = set(phrases)

    not_match_entity = 0
    for phrase in phrases:
        entity, beg, end, ref_context, url = phrase

        try:
            context = contexts[ref_context]
        except KeyError:
            print('KeyError', phrase)

        chunk_words = word_tokenize(context)

        try:
            left = chunk_words.index(entity)
            right = left + len(word_tokenize(entity))
        except ValueError:
            left = len(word_tokenize(context[:beg]))
            right = len(word_tokenize(context[:end]))

        span_text = ' '.join(chunk_words[left:right])

        if span_text != entity:
            print('ERROR:', 'span:', span_text, 'entity:', entity, 'beg-end:', context[beg:end], 'context:', context)
            not_match_entity += 1

    print(not_match_entity, len(phrases))


#  - nones: 14  - # of phrases: 331 331  - not include: 12  - # of total cand: 6124  - # of except 26 - spotlight
#  - nones: 3  - # of phrases: 144 144  - not include: 8  - # of total cand: 3498  - # of except 17 - kore50-nif
#  - nones: 318  - # of phrases: 880 880  - not include: 42  - # of total cand: 9622  - # of except 318 - Reuters-128
#  - nones: 359  - # of phrases: 1000 1000  - not include: 48  - # of total cand: 9485  - # of except 538 - RSS-500
#  - nones: 309  - # of phrases: 1655 1655  - not include: 18  - # of total cand: 20772  - # of except 1623 - News-100
def test_cand_list():
    fetch_filtered_entities = FetchFilteredCoreferencedCandEntities()
    url2graphid = load_url2graphid()

    input_ttl_fpath = "/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/ttl/News-100.ttl"
    in_ttl = codecs.open(input_ttl_fpath, "r", "utf-8")

    input_ttl = in_ttl.read()
    graph, contexts, phrases = parse_d2kb_ttl(input_ttl)
    phrases = set(phrases)

    count_nones = 0
    count = 0
    count_not_include = 0
    cand_number = 0
    key_error = 0
    for phrase in phrases:
        entity, beg, end, ref_context, url = phrase
        try:
            id = url2graphid[url]
        except KeyError:
            id = -1
            key_error += 1
        try:
            context = contexts[ref_context]
        except KeyError:
            print('KeyError', phrase)

        chunk_words = word_tokenize(context)

        try:
            left = chunk_words.index(entity)
            right = left + len(word_tokenize(entity))
        except ValueError:
            left = len(word_tokenize(context[:beg]))
            right = len(word_tokenize(context[:end]))

        cand, score = fetch_filtered_entities.process(left, right, chunk_words)

        if cand is None:
            count_nones += 1
        else:
            cand_number += len(cand)

        if cand is not None and id != -1:
            if int(id) not in cand:
                count_not_include += 1

        count += 1

    print(' - nones:', count_nones, ' - # of phrases:', count, len(phrases), ' - not include:', count_not_include,
          ' - # of total cand:', cand_number, ' - # of except', key_error)


#test_index_span()
test_cand_list()

input_ttl_fpaths = ["/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/ttl/dbpedia-spotlight-nif.ttl",
                    "/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/ttl/kore50-nif.ttl",
                    "/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/ttl/Reuters-128.ttl",
                    "/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/ttl/RSS-500.ttl",
                    "/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/ttl/News-100.ttl"]