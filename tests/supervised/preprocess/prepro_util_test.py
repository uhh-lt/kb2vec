from supervised.preprocess.prepro_util import *
from supervised.preprocess.util import load_url2graphid
from supervised.negative_sampling import parse_d2kb_ttl

'''

generator = InputSamplesGenerator()
samples = generator.process('/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/ttl/dbpedia-spotlight-nif.ttl', ttl=True)
not_include = 0
total = 0
except_ = 0
for sample in samples:
    chunk_id, chunk_words, entity, begin_gm, end_gm, ground_truth, cand_entities, cand_entities_scores = sample

    for index in range(len(entity)):
        try:
            print(entity[index], ground_truth[index], cand_entities[index])
            if int(ground_truth[index]) not in cand_entities[index]:
                not_include += 1
        except:
            except_ += 1
        total += 1


print(not_include, total, except_)
'''
# len phrase 660 - spotlight
# 288 - kore50-nif
# 880 - Reuters-128
# 1000 - RSS-500
# 1655 - News-100
def test_chunker_parse_d2kb():
    chunker = Chunker()

    input_ttl_fpath = "/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/ttl/News-100.ttl"
    in_ttl = codecs.open(input_ttl_fpath, "r", "utf-8")

    input_ttl = in_ttl.read()

    _, contexts, phrases = chunker.parse_d2kb_ttl(input_ttl)
    print('CONT:', len(contexts.keys()), 'PHR:', len(phrases.keys()))

    for key in contexts.keys():
        if key not in phrases.keys():
            print(key)

    for key in phrases.keys():
        if key not in contexts.keys():
            print(key)

    if contexts.keys() == phrases.keys():
        print("YESSS")

    _, contexts_, phrases_ = parse_d2kb_ttl(input_ttl)

    if len(set(phrases_)) != len(phrases):
        print('len original:', len(set(phrases_)) , 'len chunker:', len(phrases.keys()))

    if len(contexts) != len(contexts_):
        print('len original:', len(contexts_), 'len chunker:', len(contexts))

    if set(contexts_.keys()).difference(set(contexts.keys())):
        print('not the same context keys')

    contexts_keys = contexts.keys()
    len_phrase = 0
    for context in contexts_keys:
        try:
            phrase_contexts = phrases[context]
            for phrase in phrase_contexts:
                span, beg, end, ind_ref = phrase
                if (span, beg, end, context, ind_ref) not in phrases_:
                    print((span, beg, end, context, ind_ref))
                    return
            len_phrase += len(phrase_contexts)
        except KeyError:
            # only one context ref in spotlight, the problem in the dataset!
            # http://www.nytimes.com/2010/10/11/arts/design/11chaos.html?ref=arts_sentence2
            print('KEY ERROR:', context)
    print(len_phrase)


# number_phrases = 608 + 52 ground truth error - spotlight
# number_phrases = 254 + 34 ground truth error - kore50-nif
# number_phrases = 562 + 318 ground truth error - Reuters-128
# number_phrases = 462 + 538 ground truth error - RSS-500
# number_phrases = 32 + 1623 ground truth error - News-100
def test_process_ttl():
    chunker = Chunker()

    input_ttl_fpath = "/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/ttl/News-100.ttl"
    url2graphid = load_url2graphid()
    count = 0
    number_phrases = 0
    for chunk in chunker.process_ttl(input_ttl_fpath, url2graphid):
        #print(chunk)
        chunk_id, chunk_words, begin_gm, end_gm, ground_truth = chunk
        number_phrases += len(begin_gm)
        count += 1
    print(count, number_phrases)


# 608 11366 - dbpedia-spotlight-nifspotlight
# 254 6180 - kore50-nif
# 562 7474 - Reuters-128
# 462 6389 - RSS-500
# 32 46 - News-100
def test_chunk2sample():
    input_generator = InputSamplesGenerator()

    input_ttl_fpath = "/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/ttl/News-100.ttl"
    url2graphid = load_url2graphid()

    number_cand = 0
    number_phrases = 0


    for chunk in input_generator.chunker.process_ttl(input_ttl_fpath, url2graphid):
        chunk_id, chunk_words, begin_gm, end_gm, ground_truth, \
        cand_entities, cand_entities_scores = input_generator.chunk2sample(chunk)

        if len(begin_gm) != len(end_gm) or len(begin_gm) != len(ground_truth) or len(begin_gm) != len(cand_entities):
            print(chunk_id, begin_gm, end_gm, ground_truth, cand_entities, cand_entities_scores)
        number_phrases += len(begin_gm)

        for index in range(len(begin_gm)):
            candidates = cand_entities[index]
            number_cand += len(candidates)

    print(number_phrases, number_cand)


# 57 - dbpedia-spotlight-nifspotlight
# 50 - kore50-nif
# 107 - Reuters-128
# 334 - RSS-500
# 14 - News-100
def test_InputSampleGenerate_process():
    input_generator = InputSamplesGenerator()
    input_ttl_fpath = "/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/ttl/News-100.ttl"
    number_samples = 0
    for sample in input_generator.process(input_ttl_fpath, ttl=True):
        number_samples += 1

    print(number_samples)

print('Called')
#test_chunker_parse_d2kb()
#test_process_ttl()
#test_chunk2sample()
test_InputSampleGenerate_process()
print('Finished')
input_ttl_fpaths = ["/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/ttl/dbpedia-spotlight-nif.ttl",
                    "/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/ttl/kore50-nif.ttl",
                    "/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/ttl/Reuters-128.ttl",
                    "/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/ttl/RSS-500.ttl",
                    "/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/ttl/News-100.ttl"]