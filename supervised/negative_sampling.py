from rdflib import URIRef, Graph
import codecs
from sqlitedict import SqliteDict
from nltk.stem import WordNetLemmatizer
from random import shuffle

A = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
PHRASE = "#Phrase"
CONTEXT = "#Context"
STRING = "#isString"
ANCOR = "#anchorOf"
BEG = "#beginIndex"
END = "#endIndex"
REFCONTEXT = "#referenceContext"
INDREF = "#taIdentRef"
CLASS_URI = URIRef("http://www.w3.org/2005/11/its/rdf#taClassRef")
LINK_URI = URIRef("http://www.w3.org/2005/11/its/rdf#taIdentRef")
NONE_URI = URIRef("http://dbpedia.org/nonsense")


# NONE_URI = URIRef("http://dbpedia.org/page/Thing")


def parse_d2kb_ttl(input_ttl):
    g = Graph()
    result = g.parse(data=input_ttl, format="n3")
    contexts, phrases = get_phrases(g)

    return g, contexts, phrases


def get_phrases(g):
    """ Collect the context and phrases """

    contexts = dict()
    phrases = list()

    for subj, pred, obj in g:
        p = str(pred)
        s = str(subj)
        o = str(obj)

        # catch the context
        if o.endswith(CONTEXT):
            for pred_s, obj_s in g.predicate_objects(subj):
                if pred_s.strip().endswith(STRING):
                    contexts[s] = str(obj_s)

        # catch the phrases to disambiguate
        if o.endswith(PHRASE) or p.endswith(ANCOR):
            phrase = ""
            end = -1
            beg = -1
            ref_context = ""
            ind_ref = ""
            for pred_s, obj_s in g.predicate_objects(subj):
                ps = pred_s.strip()
                if ps.endswith(ANCOR):
                    phrase = str(obj_s)
                elif ps.endswith(BEG):
                    beg = int(obj_s)
                elif ps.endswith(END):
                    end = int(obj_s)
                elif ps.endswith(REFCONTEXT):
                    ref_context = str(obj_s)
                elif ps.endswith(INDREF):
                    ind_ref = str(obj_s)

            if phrase == "" or beg == -1 or end == -1:
                print("Warning: bad phrase", subj, pred, obj)
            else:
                phrases.append((phrase, beg, end, ref_context, ind_ref))

    return contexts, phrases


def write_positive_samples(contexts, phrases, path):

    with codecs.open(path, "a", "utf-8") as file:
        for phrase in phrases:
            entity, beg, end, ref_context, url = phrase[0], phrase[1], phrase[2], phrase[3], phrase[4]
            try:
                context = contexts[ref_context].strip()
                file.write(entity + '\t' + str(beg) + '\t' + str(end) + '\t' + url + '\t' + context.replace("\n", " ") + '\t' + '1' + '\n')
            except KeyError:
                print("Warning: not found", ref_context)


def write_negative_samples(phrases, path):
    with codecs.open(path, "a", "utf-8") as file:
        for phrase in phrases:
            entity, beg, end, url, context, is_positive = phrase[0], phrase[1], phrase[2], phrase[3], phrase[4], phrase[5]

            file.write(
                    entity + '\t' + str(beg) + '\t' + str(end) + '\t' + url + '\t' + context + '\t' + str(is_positive) + '\n')


def write_negative_samples_with_positive_samples(positive_negatives, path):
    with codecs.open(path, "a", "utf-8") as file:
        for positive_negative in positive_negatives:
            entity, beg, end, true_url, context = positive_negative[0]
            negatives = positive_negative[1:]

            str2write = entity + '\t' + str(beg) + '\t' + str(end) + '\t' + true_url + '\t' + context.strip()
            for negative_url in negatives:
                str2write += '\t' + negative_url.strip()

            file.write(str2write + '\n')


def write_negative_samples_with_positive_samples_with_scores(positive_negatives, path):
    with codecs.open(path, "a", "utf-8") as file:
        for positive_negative in positive_negatives:
            entity, beg, end, true_url, context = positive_negative[0]
            negatives = positive_negative[1:]

            str2write = entity + '\t' + str(beg) + '\t' + str(end) + '\t' + true_url + '\t' + context.strip()
            for negative_url in negatives:
                if len(negative_url) > 1:
                    url, score = negative_url[0], negative_url[1]
                    str2write += '\t' + url.strip() + '\t' + str(score)
                else:
                    print(true_url)
                    str2write += '\t' + negative_url.strip()

            file.write(str2write + '\n')


def read_negative_samples_with_positive_samples(path):
    positives_negatives = list()

    with codecs.open(path, "r", "utf-8") as file:
        lines = file.readlines()

    for line in lines:
        splitted = line.split('\t')
        entity, beg, end, true_url, context, negative_samples = splitted[0], int(splitted[1]), int(splitted[2]), \
                                            splitted[3], splitted[4], splitted[5:]

        positives_negatives.append([entity, beg, end, true_url, context, negative_samples])

    return positives_negatives


def read_samples(path):
    contexts = dict()
    phrases = dict()

    with codecs.open(path, "r", "utf-8") as file:
        lines = file.readlines()

    for line in lines:
        splitted = line.split('\t')

        entity, beg, end, url, context, is_positive = splitted[0], int(splitted[1]), int(splitted[2]), \
                                                        splitted[3], splitted[4], int(splitted[5])

        phrases[entity+str(beg)+str(end)+url+context] = (entity, beg, end, url)
        contexts[entity+str(beg)+str(end)+url+context] = context

    return contexts, phrases


def create_keywords_from_url(url_db):
    url_keywords = dict()

    db = SqliteDict(url_db, autocommit=False)
    urls = db.keys()
    lemmatizer = WordNetLemmatizer()

    for url in urls:
        entity = url.split('/')[-1]

        if "–" in entity:
            words = list()
            first_words = entity.split('_')
            for word in first_words:
                words.extend(word.split('–'))
        elif "-" in entity:
            words = list()
            first_words = entity.split('_')
            for word in first_words:
                words.extend(word.split('-'))
        else:
            words = entity.split('_')

        keywords = set()
        for word in words:
            prepro = word.strip(',').strip('.').strip('(').strip(')').lower()
            # keywords.add(prepro)
            keywords.add(lemmatizer.lemmatize(prepro))

        url_keywords[url] = keywords

    db.close()

    return url_keywords


def create_negative_samples(urls_db, contexts, phrases):
    url_keywords = create_keywords_from_url(urls_db)
    negative_samples = list()

    urls = url_keywords.keys()
    keys = phrases.keys()
    for key in keys:
        entity, beg, end, true_url = phrases[key]

        for candidate_url in urls:
            keywords = url_keywords[candidate_url]

            words = entity.split()
            for word in words:
                if word.lower() in keywords and candidate_url != true_url:

                    # entity, beginning, ending, negative url, context, is_positive
                    negative_samples.append((entity, beg, end, candidate_url, contexts[key], 0))
                    break

    return negative_samples


def create_negative_samples_with_positive_samples(urls_db, contexts, phrases):
    url_keywords = create_keywords_from_url(urls_db)
    positive_negatives = list()
    lemmatizer = WordNetLemmatizer()

    urls = url_keywords.keys()
    keys = phrases.keys()
    count = 0

    print(len(keys))

    for key in keys:
        if count % 100 == 0:
            print(count)
        count += 1
        entity, beg, end, true_url = phrases[key]
        negatives = [(entity, beg, end, true_url, contexts[key])]

        negative_samples = list()
        for candidate_url in urls:
            keywords = url_keywords[candidate_url]

            words = entity.split()

            if "–" in entity:
                words = entity.split("–")
            elif "-" in entity:
                words = entity.split("-")

            for word in words:
                word = lemmatizer.lemmatize(word.lower())
                if word in keywords and candidate_url != true_url:

                    negative_samples.append(candidate_url)
                    break

        negatives.extend(negative_samples)
        if len(negative_samples) == 0:
            print(entity)
        positive_negatives.append(negatives)

    return positive_negatives


def filter_negative_samples_randomly(positives_negatives, url_db, n=10):
    filtered_samples = list()
    db = SqliteDict(url_db, autocommit=False)

    keys = db.keys()
    urls = [key for key in keys]

    for positive_negative in positives_negatives:
        entity, beg, end, true_url, context, negative_samples = positive_negative[0], int(positive_negative[1]), int(positive_negative[2]), \
                                                                positive_negative[3], positive_negative[4], positive_negative[5:][0]

        try:
            # check if db contains this url or not.
            id = db[true_url]
        except KeyError:
            # if not, skip it.
            continue

        addition = list()

        # if it does not have negative samples, then completely random samples are created.
        length = len(negative_samples)
        if length == 0:
            negative_samples = urls
        elif length < n:
            shuffle(urls)
            size = n - len(negative_samples)
            addition = urls[:size]

        shuffle(negative_samples)

        filtered_sample = [(entity, beg, end, true_url, context)]
        negative_samples.extend(addition)
        filtered_sample.extend(negative_samples[:n])
        filtered_samples.append(filtered_sample)

    return filtered_samples


def filter_negative_samples_closest(positives_negatives, url_db, pagerank_db, n=10):
    filtered_samples = list()
    url_keywords = create_keywords_from_url(url_db)
    url_pagerank = SqliteDict(pagerank_db, autocommit=False)
    db = SqliteDict(url_db, autocommit=False)

    keys = db.keys()
    urls = [key for key in keys]

    for positive_negative in positives_negatives:
        negativeurl_score = dict()
        entity, beg, end, true_url, context, negative_samples = positive_negative[0], int(positive_negative[1]), int(positive_negative[2]),\
                                                                positive_negative[3], positive_negative[4], positive_negative[5:][0]
        try:
            true_keywords = url_keywords[true_url]
        except KeyError:
            continue

        union = set(entity.split()).union(set(true_keywords))

        for negative_url in negative_samples:
            negative_url = negative_url.strip()
            try:
                keywords = url_keywords[negative_url]
            except KeyError:
                continue

            number_intersection = len(union.intersection(set(keywords)))
            length = len(keywords)

            page_rank = url_pagerank[negative_url]

            negativeurl_score[negative_url] = number_intersection * page_rank / length

        sorted_samples = [url for url in sorted(negativeurl_score, key=negativeurl_score.get, reverse=True)]

        addition = list()

        # if it does not have negative samples, then completely random samples are created.
        length = len(negative_samples)
        if length == 0:
            sorted_samples = urls
            shuffle(sorted_samples)
        elif length < n:
            shuffle(urls)
            size = n - len(negative_samples)
            addition = urls[:size]

        filtered_sample = [(entity, beg, end, true_url, context)]
        sorted_samples.extend(addition)
        filtered_sample.extend(sorted_samples[:n])
        filtered_samples.append(filtered_sample)

    return filtered_samples


def filter_negative_samples_closest_with_scores(positives_negatives, url_db, pagerank_db, n=10):
    filtered_samples = list()
    url_keywords = create_keywords_from_url(url_db)
    url_pagerank = SqliteDict(pagerank_db, autocommit=False)
    db = SqliteDict(url_db, autocommit=False)

    keys = db.keys()
    urls = [key for key in keys]

    for positive_negative in positives_negatives:
        negativeurl_score = dict()
        entity, beg, end, true_url, context, negative_samples = positive_negative[0], int(positive_negative[1]), int(positive_negative[2]),\
                                                                positive_negative[3], positive_negative[4], positive_negative[5:][0]
        try:
            true_keywords = url_keywords[true_url]
        except KeyError:
            continue

        union = set(entity.split()).union(set(true_keywords))

        for negative_url in negative_samples:
            negative_url = negative_url.strip()
            try:
                keywords = url_keywords[negative_url]
            except KeyError:
                continue

            number_intersection = len(union.intersection(set(keywords)))
            length = len(keywords)

            page_rank = url_pagerank[negative_url]

            negativeurl_score[negative_url] = number_intersection * page_rank / length

        sorted_samples_ = [url for url in sorted(negativeurl_score, key=negativeurl_score.get, reverse=True)]
        sorted_samples = list()
        for sample in sorted_samples_:
            sorted_samples.append((sample, negativeurl_score[sample]))

        addition = list()

        # if it does not have negative samples, then completely random samples are created.
        length = len(negative_samples)
        if length == 0:
            sorted_samples = urls
            shuffle(sorted_samples)
        elif length < n:
            shuffle(urls)
            size = n - len(negative_samples)
            addition = urls[:size]

        filtered_sample = [(entity, beg, end, true_url, context)]
        sorted_samples.extend(addition)
        filtered_sample.extend(sorted_samples[:n])
        filtered_samples.append(filtered_sample)

    return filtered_samples


def create_completely_random(urls_db, contexts, phrases, n=10):
    samples = list()

    db = SqliteDict(urls_db, autocommit=False)
    url_keys = db.keys()
    urls = [url for url in url_keys]

    keys = phrases.keys()
    count = 0

    print(len(keys))

    for key in keys:
        if count % 100 == 0:
            print(count)
        count += 1
        entity, beg, end, true_url = phrases[key]
        negatives = [(entity, beg, end, true_url, contexts[key])]

        shuffle(urls)

        negative_samples = urls[:n]
        negatives.extend(negative_samples)
        samples.append(negatives)

    return samples

