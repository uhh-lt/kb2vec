import re
from rdflib import URIRef, Graph
import codecs
from candidate import Phrase


A = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
PHRASE = "#Phrase"
CONTEXT = "#Context"
STRING = "#isString"
ANCOR = "#anchorOf"
BEG = "#beginIndex"
END = "#endIndex"
CLASS_URI = URIRef("http://www.w3.org/2005/11/its/rdf#taClassRef")
LINK_URI = URIRef("http://www.w3.org/2005/11/its/rdf#taIdentRef")
NONE_URI = URIRef("http://dbpedia.org/nonsense")
# NONE_URI = URIRef("http://dbpedia.org/page/Thing")


class DatasetBuilder(object):
    def __init__(self, dataset_fpath):
        self._dataset_fpath = dataset_fpath
        with codecs.open(self._dataset_fpath, "a", "utf-8") as ttl_f:
            ttl_f.write("targets\tcontext\n")

    def add_to_dataset(self, input_ttl):
        graph, context, phrases = parse_d2kb_ttl(input_ttl)
        with codecs.open(self._dataset_fpath, "a", "utf-8") as ttl_f:
            phrases_str = ", ".join(p.text for p in phrases)
            ttl_f.write("{}\t{}\n".format(phrases_str, context))


def parse_d2kb_ttl(input_ttl):
    g = Graph()
    result = g.parse(data=input_ttl, format="n3")
    contexts, phrases = get_phrases(g)

    return g, contexts, phrases


def get_phrases(g):
    """ Collect the context and phrases """
    
    contexts = []
    phrases = []
    
    for subj, pred, obj in g:
        p = str(pred)
        s = str(subj)
        o = str(obj)

        # catch the context 
        if o.endswith(CONTEXT):
            for pred_s, obj_s in g.predicate_objects(subj):
                if pred_s.strip().endswith(STRING):
                    contexts.append(obj_s)

        # catch the phrases to disambiguate 
        if o.endswith(PHRASE):
            phrase = ""
            end = -1
            beg = -1
            for pred_s, obj_s in g.predicate_objects(subj):
                ps = pred_s.strip()
                if ps.endswith(ANCOR): phrase = str(obj_s)
                elif ps.endswith(BEG): beg = int(obj_s)
                elif ps.endswith(END): end = int(obj_s)

            if phrase == "" or beg == -1 or end == -1:
                print("Warning: bad phrase", subj, pred, obj)
            else:
                phrases.append(Phrase(phrase, beg, end, subj))

    return contexts, phrases


def add_nonsense_response(input_ttl):
    graph, context, phrases = parse_d2kb_ttl(input_ttl)
    
    # add new triples that correspond to the links of the disambiguation links
    print("# triples input:", len(graph))
    for phrase in phrases:
        graph.add( (phrase.subj, CLASS_URI, NONE_URI) )
        graph.add( (phrase.subj, LINK_URI, NONE_URI) )
    print("# triples output:", len(graph))

    output_ttl = str(graph.serialize(format='n3', encoding="utf-8"), "utf-8")
    
    return output_ttl


def remove_classref(text):
    output = []
    for line in text.split("\n"):
        upd_line = re.sub(r"itsrdf:taClassRef     <[^;]*> ;",
                          "itsrdf:taClassRef     <nonsense> ;",
                          line)
        output.append(upd_line)
        
    return "\n".join(output)

