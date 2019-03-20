from collections import defaultdict
import time
import sys
import os, codecs
from sqlitedict import SqliteDict


def load_url2graphid(path=None):
    if path is None:
        path='/Users/sevgili/Ozge-PhD/DBpedia-datasets/outputs/databases/intersection_nodes_lookup.db'
    return SqliteDict(path, autocommit=False)


def load_wiki2graph(path=None):
    if path is None:
        path='preprocess/idmaps/wikiid2graphid.txt'
    wiki2graphmap = dict()

    with open(path) as fin:
        for line in fin:
            id1, id2 = line.split('\t')
            wiki2graphmap[int(id1)] = int(id2)

    return wiki2graphmap


def load_graph2wiki(path=None):
    if path is None:
        path='preprocess/idmaps/graphid2wikiid.txt'
    graph2wikimap = dict()
    count = 0
    multiple_references = set()
    with open(path) as fin:
        for line in fin:
            id1, id2 = line.split('\t')
            id1, id2 = int(id1), int(id2)
            try:
                graph2wikimap[id1].add(id2)
                multiple_references.add(id1)
                count += 1
            except:
                graph2wikimap[id1] = set()
                graph2wikimap[id1].add(id2)

    print(count, len(multiple_references))
    return graph2wikimap, multiple_references


# original https://github.com/dalab/end2end_neural_el
def load_wiki_name_id_map(lowercase=False,
                          filepath=None):
    if filepath is None:
        filepath="/Users/sevgili/PycharmProjects/end2end_neural_el/data/basic_data/wiki_name_id_map.txt"
    wall_start = time.time()
    wiki_name_id_map = dict()
    wiki_id_name_map = dict()
    wiki_name_id_map_errors = 0
    duplicate_names = 0    # different lines in the doc with the same title
    duplicate_ids = 0      # with the same id

    with codecs.open(filepath, 'r', encoding='utf8') as fin:
        for line in fin:
            line = line.rstrip()
            try:
                wiki_title, wiki_id = line.split("\t")

                if lowercase:
                    wiki_title = wiki_title.lower()

                if wiki_title in wiki_name_id_map:
                    duplicate_names += 1
                if wiki_id in wiki_id_name_map:
                    duplicate_ids += 1

                wiki_name_id_map[wiki_title] = wiki_id
                wiki_id_name_map[wiki_id] = wiki_title
            except ValueError:
                wiki_name_id_map_errors += 1
    print("load wiki_name_id_map. wall time:", (time.time() - wall_start)/60, " minutes")
    print("wiki_name_id_map_errors: ", wiki_name_id_map_errors)
    print("duplicate names: ", duplicate_names)
    print("duplicate ids: ", duplicate_ids)
    return wiki_name_id_map, wiki_id_name_map



class WikiNameIdMap(object):
    def __init__(self):
        pass

    def init_compatible_ent_id(self):
        self.wiki_name_id_map, self.wiki_id_name_map = load_wiki_name_id_map(lowercase=False)

    # original https://github.com/dalab/end2end_neural_el
    def is_valid_wiki_id(self, wiki_id):
        return wiki_id in self.wiki_id_name_map


# original https://github.com/dalab/end2end_neural_el
def custom_p_e_m(cand_ent_num=30, allowed_entities_set=None,
                 lowercase_p_e_m=False, filedict=None):
    """Args:
    cand_ent_num: how many candidate entities to keep for each mention
    allowed_entities_set: restrict the candidate entities to only this set. for example
    the most frequent 1M entities. First this restiction applies and then the cand_ent_num."""
    wall_start = time.time()
    if filedict is None:
        filedict='/Users/sevgili/PycharmProjects/end2end_neural_el/data/basic_data/prob_yago_crosswikis_wikipedia_p_e_m.txt'
    p_e_m = dict()  # for each mention we have a list of tuples (wiki_id, score)
    mention_total_freq = dict() # for each mention of the p_e_m we store the total freq
                                # this will help us decide which cand entities to take
    p_e_m_errors = 0

    wikiNameIdMap = WikiNameIdMap()
    wikiNameIdMap.init_compatible_ent_id()

    incompatible_ent_ids = 0
    with codecs.open(filedict, 'r', encoding='utf8') as fin:
        duplicate_mentions_cnt = 0
        clear_conflict_winner = 0  # both higher absolute frequency and longer cand list
        not_clear_conflict_winner = 0  # higher absolute freq but shorter cand list
        for line in fin:
            line = line.rstrip()
            try:
                temp = line.split("\t")
                mention, entities = temp[0],  temp[2:]
                absolute_freq = int(temp[1])
                res = []
                for e in entities:
                    if len(res) >= cand_ent_num:
                        break
                    wiki_id, score, entity_name = map(str.strip, e.split(',', 2))
                    #print(wiki_id, score)
                    if not wikiNameIdMap.is_valid_wiki_id(wiki_id):
                        incompatible_ent_ids += 1
                    elif allowed_entities_set is not None and \
                                    wiki_id not in allowed_entities_set:
                        pass
                    else:
                        res.append((wiki_id, float(score)))
                if res:
                    if mention in p_e_m:
                        duplicate_mentions_cnt += 1
                        #print("duplicate mention: ", mention)
                        if absolute_freq > mention_total_freq[mention]:
                            if len(res) > len(p_e_m[mention]):
                                clear_conflict_winner += 1
                            else:
                                not_clear_conflict_winner += 1
                            p_e_m[mention] = res
                            mention_total_freq[mention] = absolute_freq
                    else:
                        p_e_m[mention] = res    # for each mention we have a list of tuples (wiki_id, score)
                        mention_total_freq[mention] = absolute_freq

            except Exception as esd:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                p_e_m_errors += 1
                print("error in line: ", repr(line))

    print("duplicate_mentions_cnt: ", duplicate_mentions_cnt)
    print("end of p_e_m reading. wall time:", (time.time() - wall_start)/60, " minutes")
    print("p_e_m_errors: ", p_e_m_errors)
    print("incompatible_ent_ids: ", incompatible_ent_ids)

    if not lowercase_p_e_m:   # do not build lowercase dictionary
        return p_e_m, None, mention_total_freq

    wall_start = time.time()
    # two different p(e|m) mentions can be the same after lower() so we merge the two candidate
    # entities lists. But the two lists can have the same candidate entity with different score
    # we keep the highest score. For example if "Obama" mention gives 0.9 to entity Obama and
    # OBAMA gives 0.7 then we keep the 0.9 . Also we keep as before only the cand_ent_num entities
    # with the highest score
    p_e_m_lowercased = defaultdict(lambda: defaultdict(int))

    for mention, res in p_e_m.items():
        l_mention = mention.lower()
        # if l_mention != mention and l_mention not in p_e_m:
        #   the same so do nothing      already exist in dictionary
        #   e.g. p(e|m) has Obama and obama. So when i convert Obama to lowercase
        # I find that obama already exist so i will prefer this.
        if l_mention not in p_e_m:
            for r in res:
                wiki_id, score = r
                p_e_m_lowercased[l_mention][wiki_id] = max(score, p_e_m_lowercased[l_mention][wiki_id])

    print("end of p_e_m lowercase. wall time:", (time.time() - wall_start)/60, " minutes")

    import operator
    p_e_m_lowercased_trim = dict()
    for mention, ent_score_map in p_e_m_lowercased.items():
        sorted_ = sorted(ent_score_map.items(), key=operator.itemgetter(1), reverse=True)
        p_e_m_lowercased_trim[mention] = sorted_[:cand_ent_num]
    print(p_e_m_lowercased_trim)
    return p_e_m, p_e_m_lowercased_trim, mention_total_freq


class FetchCandidateEntities(object):
    """takes as input a string or a list of words and checks if it is inside p_e_m
    if yes it returns the candidate entities otherwise it returns None.
    it also checks if string.lower() inside p_e_m and if string.lower() inside p_e_m_low"""
    def __init__(self):
        self.lowercase_spans = 30
        self.lowercase_p_e_m = False
        self.p_e_m, self.p_e_m_low, self.mention_total_freq = custom_p_e_m(
            cand_ent_num=30,
            lowercase_p_e_m=False)


    def process(self, span):
        """span can be either a string or a list of words"""
        if isinstance(span, list):
            span = ' '.join(span)
        title = span.title()
        # 'obama 44th president of united states'.title() # 'Obama 44Th President Of United States'
        title_freq = self.mention_total_freq[title] if title in self.mention_total_freq else 0
        span_freq = self.mention_total_freq[span] if span in self.mention_total_freq else 0

        if title_freq == 0 and span_freq == 0:
            if self.lowercase_spans and span.lower() in self.p_e_m:

                return map(list, zip(*self.p_e_m[span.lower()]))
            elif self.lowercase_p_e_m and span.lower() in self.p_e_m_low:
                return map(list, zip(*self.p_e_m_low[span.lower()]))
            else:
                return None, None

        else:
            if span_freq > title_freq:
                return map(list, zip(*self.p_e_m[span]))
            else:
                return map(list, zip(*self.p_e_m[title]))

                # from [('ent1', 0.4), ('ent2', 0.3), ('ent3', 0.3)] to
                # ('ent1', 'ent2', 'ent3')  and (0.4, 0.3, 0.3)
                # after map we have lists i.e. ['ent1', 'ent2', 'ent3']   , [0.4, 0.3, 0.3]


class FetchFilteredCoreferencedCandEntities(object):
    def __init__(self):
        self.fetchCandidateEntities = FetchCandidateEntities()
        self.wiki2graph = load_wiki2graph(None)
        #self.keyerror = 0

    # takes all context (chunk_words) and location of the span (begining(left) - ending(right) indexes of span)
    def process(self, left, right, chunk_words):
        span_text = ' '.join(chunk_words[left:right])
        cand_ent, scores = self.fetchCandidateEntities.process(span_text)
        # changing wiki to graph
        cand_ent_ = None
        if cand_ent is not None:
            cand_ent_ = list()
            for cand in cand_ent:
                try:
                    cand_ent_.append(self.wiki2graph[int(cand)])
                except:
                    #self.keyerror += 1
                    continue

        #if self.keyerror:
        #    print('candidate entities map, key error: ', self.keyerror, ' span_text: ', span_text)
        #    self.keyerror = 0
        return cand_ent_, scores

    '''
    def find_corefence_person(self, span_text, left_right_words):
        """if span_text is substring of another person's mention found before. it should be
        substring of words. so check next and previous characters to be non alphanumeric"""
        if len(span_text) < 3:
            return None
        if left_right_words:  # this check is only for allspans mode not for gmonly.
            if left_right_words[0] and left_right_words[0][0].isupper() or \
                            left_right_words[1] and left_right_words[1][0].isupper():
                # if the left or the right word has uppercased its first letter then do not search for coreference
                # since most likely it is a subspan of a mention.
                # This condition gives no improvement to Gerbil results even a very slight decrease (0.02%)
                return None
        for mention in reversed(self.persons_mentions_seen):
            idx = mention.find(span_text)
            if idx != -1:
                if len(mention) == len(span_text):
                    continue   # they are identical so no point in substituting them
                if idx > 0 and mention[idx-1].isalpha():
                    continue
                if idx + len(span_text) < len(mention) and mention[idx+len(span_text)].isalpha():
                    continue
                #print("persons coreference, before:", span_text, "after:", mention)
                return mention
        return None
    '''

if __name__ == "__main__":
    p_e_m, p_e_m_lowercased_trim, mention_total_freq = custom_p_e_m()
    print(p_e_m_lowercased_trim)
    print(len(p_e_m.keys()), len(p_e_m_lowercased_trim.keys()), len(mention_total_freq.keys()))