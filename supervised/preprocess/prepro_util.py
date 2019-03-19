import preprocess.util as util
from collections import namedtuple
import codecs
from rdflib import URIRef, Graph
from nltk.tokenize import word_tokenize

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
DBPEDIA_PREFIX = "http://dbpedia.org/resource/"


class Chunker(object):
    def __init__(self):
        self.separator = "per_document"
        self.chunk_ending = {'DOCEND'}
        if self.separator == "per_paragraph":
            self.chunk_ending.add('*NL*')
        if self.separator == "per_sentence":
            self.chunk_ending.add('.')
            self.chunk_ending.add('*NL*')
        self.parsing_errors = 0
        self.ground_truth_errors = 0
        self.wiki2graph = util.load_wiki2graph(None)

    def new_chunk(self):
        self.chunk_words = []
        self.begin_gm = []          # the starting positions of gold mentions
        self.end_gm = []            # the end positions of gold mentions
        self.ground_truth = []      # list with the correct entity ids

    def compute_result(self, docid):
        chunk_id = docid
        if self.separator == "per_paragraph":
            chunk_id = chunk_id + "&*" + str(self.par_cnt)
        if self.separator == "per_sentence":
            chunk_id = chunk_id + "&*" + str(self.par_cnt) + "&*" + str(self.sent_cnt)
        result = (chunk_id, self.chunk_words, self.begin_gm, self.end_gm, self.ground_truth)

        # correctness checks. not necessary
        no_errors_flag = True
        if len(self.begin_gm) != len(self.end_gm) or \
            len(self.begin_gm) != len(self.ground_truth):
            no_errors_flag = False
        for b, e in zip(self.begin_gm, self.end_gm):
            if e <= b or b >= len(self.chunk_words) or e > len(self.chunk_words):
                no_errors_flag = False

        self.new_chunk()
        if no_errors_flag == False:
            self.parsing_errors += 1
            print("chunker parse error: ", result)
            return None
        else:
            return result

    def process(self, path):
        with open(path) as fin:
            self.new_chunk()
            docid = ""
            # paragraph and sentence counter are not actually useful. only for debugging purposes.
            self.par_cnt = 0      # paragraph counter (useful if we work per paragraph)
            self.sent_cnt = 0      # sentence counter (useful if we work per sentence)
            for line in fin:
                line = line.rstrip()     # omit the '\n' character
                if line in self.chunk_ending:
                    if len(self.chunk_words) > 0:  # if we have continues *NL* *NL* do not return empty chunks
                        temp = self.compute_result(docid)
                        if temp is not None:
                            yield temp
                    # do not add the chunk separator, no use
                    if line == '*NL*':
                        self.par_cnt += 1
                        self.sent_cnt = 0
                    if line == '.':
                        self.sent_cnt += 1
                elif line == '*NL*':
                    self.par_cnt += 1
                    self.sent_cnt = 0
                    # do not add this in our words list
                elif line == '.':
                    self.sent_cnt += 1
                    self.chunk_words.append(line)
                elif line.startswith('MMSTART_'):
                    ent_id = line[8:]   # assert that ent_id in wiki_name_id_map
                    # convert ent_id to graph_id, if no graph_id, -1 is assigned
                    # self.ground_truth.append(ent_id)
                    try:
                        self.ground_truth.append(self.wiki2graph[int(ent_id)])
                    except:
                        self.ground_truth.append(-1)
                        self.ground_truth_errors += 1

                    self.begin_gm.append(len(self.chunk_words))
                elif line == 'MMEND':
                    self.end_gm.append(len(self.chunk_words))
                elif line.startswith('DOCSTART_'):
                    docid = line[9:]
                    self.par_cnt = 0
                    self.sent_cnt = 0
                else:
                    self.chunk_words.append(line)

        print(path, " chunker parsing errors: ", self.parsing_errors, "ground truth errors:", self.ground_truth_errors)
        self.parsing_errors = 0
        self.ground_truth_errors = 0

    def parse_d2kb_ttl(self, input_ttl):
        g = Graph()
        result = g.parse(data=input_ttl, format="n3")
        contexts, phrases = self.get_phrases(g)

        return g, contexts, phrases

    def get_phrases(self, g):
        """ Collect the context and phrases """

        contexts = dict()
        phrases = dict()

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
                    try:
                        phrases[ref_context].append((phrase, beg, end, ind_ref))
                    except KeyError:
                        phrases[ref_context] = list()
                        phrases[ref_context].append((phrase, beg, end, ind_ref))

        return contexts, phrases

    def process_ttl(self, path, url2graphid):
        in_ttl = codecs.open(path, "r", "utf-8")
        input_ttl = in_ttl.read()
        _, contexts, phrases = self.parse_d2kb_ttl(input_ttl)
        # TODO: look at the keys and see the reason of the difference
        #print(contexts.keys(), phrases.keys())
        #print(set(contexts.keys()) == set(phrases.keys()))
        #return 0
        self.new_chunk()
        count_except = 0
        count_except_url = 0

        context_urls = contexts.keys()
        for context_url in context_urls:
            context = contexts[context_url]
            self.chunk_words = word_tokenize(context)
            try:
                context_phrases = phrases[context_url]

                for phrase in context_phrases:
                    span, beg, end, ind_ref = phrase
                    try:
                        graph_id = url2graphid[ind_ref]
                        self.ground_truth.append(graph_id)
                        self.begin_gm.append(len(word_tokenize(context[:beg])))
                        self.end_gm.append(len(word_tokenize(context[:end])))
                    except KeyError:
                        self.ground_truth_errors += 1
                        continue
            except KeyError:
                count_except_url += 1
                continue
            yield (context_url, self.chunk_words, self.begin_gm, self.end_gm, self.ground_truth)

        print(path, " chunker parsing errors: ", self.parsing_errors, "ground truth errors:", self.ground_truth_errors,
              'not found url', count_except_url)
        self.parsing_errors = 0
        self.ground_truth_errors = 0


Sample = namedtuple("Sample",
                          ["chunk_id", "chunk_words", 'begin_gm', "end_gm",
                          "ground_truth", "cand_entities", "cand_entities_scores"])


class InputSamplesGenerator(object):
    def __init__(self):
        self.chunker = Chunker()
        self.fetchFilteredCoreferencedCandEntities = util.FetchFilteredCoreferencedCandEntities()
        self.url2graphid = util.load_url2graphid(None)
        self.wiki2graph = util.load_wiki2graph(None)

    def chunk2sample(self, chunk):
        cand_entities = []  # list of lists     candidate entities
        cand_entities_scores = []
        chunk_id, chunk_words, begin_gm, end_gm, ground_truth = chunk

        for left, right, gt in zip(begin_gm, end_gm, ground_truth):
            cand_ent, scores = self.fetchFilteredCoreferencedCandEntities.process(left, right, chunk_words)

            if cand_ent is None:
                cand_ent, scores = [], []

            cand_entities.append(cand_ent)
            cand_entities_scores.append(scores)

        return chunk_id, chunk_words, begin_gm, end_gm, ground_truth, cand_entities, cand_entities_scores

    def process(self, path, ttl=False):
        if ttl:
            for chunk in self.chunker.process_ttl(path, self.url2graphid):
                chunk_id, chunk_words, begin_gm, end_gm, ground_truth, \
                                        cand_entities, cand_entities_scores = self.chunk2sample(chunk)

                if begin_gm:  #not emtpy
                    yield Sample(chunk_id, chunk_words, begin_gm, end_gm, ground_truth,
                                       cand_entities, cand_entities_scores)
        else:
            for chunk in self.chunker.process(path):
                chunk_id, chunk_words, begin_gm, end_gm, ground_truth, \
                        cand_entities, cand_entities_scores = self.chunk2sample(chunk)

                if begin_gm:  # not emtpy
                    yield Sample(chunk_id, chunk_words, begin_gm, end_gm, ground_truth,
                                 cand_entities, cand_entities_scores)


if __name__ == "__main__":
    generator = InputSamplesGenerator()
    samples = generator.process('/Users/sevgili/PycharmProjects/end2end_neural_el/data/new_datasets/ace2004.txt')
    #samples = generator.process('/Users/sevgili/Ozge-PhD/DBpedia-datasets/training-datasets/ttl/RSS-500.ttl',
    #                            ttl=True)
    #print(len(samples), len(context_excepts.keys()))

    #contexts = context_excepts.keys()
    #for context in contexts:
    #    print(context, len(context_excepts[context]), len(context_cands[context]))
    count = 0
    for sample in samples:
        #print('***** AAAAAAAA****', sample)
        count += 1
    print(count)