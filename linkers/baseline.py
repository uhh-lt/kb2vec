from converter import URIConverter
from math import log
import json
from utils import truncated_log, overlap
from candidate import Candidate
from diffbot_api import CachedQuery, EL_POL_ENTITY_TYPES
from ttl import parse_d2kb_ttl, CLASS_URI, LINK_URI, NONE_URI
from rdflib import URIRef


class TTLinker(object):
    def link_ttl(self, input_ttl):
        graph, context, phrases = parse_d2kb_ttl(input_ttl)
        input_len = len(graph)

        for phrase, candidate in self.link(context, phrases):
            if candidate and candidate.link:
                graph.add( (phrase.subj, LINK_URI, URIRef(candidate.link)) )
                graph.add( (phrase.subj, CLASS_URI, URIRef(candidate.link)) )
            else:
                print("Warning: no candidates for the phrase {}".format(phrase))
                graph.add( (phrase.subj, LINK_URI, NONE_URI) )
                graph.add( (phrase.subj, CLASS_URI, NONE_URI) )

        print("# triples input:", input_len)
        print("# triples output:", len(graph))
        output_ttl = str(graph.serialize(format='n3', encoding="utf-8"), "utf-8")
        
        return output_ttl


class BaselineLinker(TTLinker):
    def __init__(self, use_overlap = True, verbose = True):
        self._cq = CachedQuery()
        self._conv = URIConverter()
        self._use_overlap = use_overlap
        self._verbose = verbose
        
    def __del__(self):
        self.close()
        
    def close(self):
        try:
            self._cq.close()
            self._conv.close()
        except:
            print("Warning: trying to close a closed object.") 
    
    def _find_wiki_uri(self, uris):
        for uri in uris:
            if "wikipedia.org" in uri:
                return uri
        return "" 

    def _get_dbpedia_uri(self, wiki_uri, uris):
        dbpedia_uri = ""
        
        if wiki_uri != "":
            dbpedia_uri = self._conv.wikipedia2dbpedia(wiki_uri)
        else:
            for uri in uris:
                dbpedia_uri = self._conv.wikidata2dbpedia(uri)
                if dbpedia_uri != "": break

        return dbpedia_uri

    def _link_db_query(self, target, diffbot_query_response, use_overlap=True):
        candidates = []
        if "data" not in diffbot_query_response:
            return candidates
        else:
            data = diffbot_query_response["data"]

        for hit in data:
            uris = set(hit["allUris"])
            if "origin" in hit: uris.add( hit["origin"] )
            if "origins" in hit: uris.union( set(hit["origins"]) )
            if "wikipediaUri" in hit:
                uris.add( hit["wikipediaUri"] )

            if "importance" in hit:
                name = hit["name"]
                importance = float(hit["importance"])
                score = truncated_log(importance) * overlap(name, target) if use_overlap else importance
                wiki_uri = self._find_wiki_uri(uris)
                dbpedia_uri = self._get_dbpedia_uri(wiki_uri, uris)

                c = Candidate(score,
                              name,
                              dbpedia_uri,
                              wiki_uri,
                              hit["types"],
                              hit["allNames"],
                              uris)
                candidates.append(c)

                print(">>> {}=log({})*{}".format(score, importance, overlap(name,target)), c)
                print
            else:
                print("Warning: Skipping a hit without importance value.")

        return sorted(candidates, reverse=True)

    def link(self, context, phrases):
        linked_phrases = []
        for phrase in phrases:
            candidates = []
            for entity_type in EL_POL_ENTITY_TYPES:
                r = self._cq.make_query('type:{} name:"{}"'.format(entity_type, phrase.text))
                db_response = json.loads(r.content)
                candidates += self._link_db_query(phrase.text, db_response, use_overlap=self._use_overlap) 
            candidates = set(candidates)

            if len(candidates) > 0:
                best = sorted(candidates, reverse=True)[0]
            else:
                best = None
            linked_phrases.append( (phrase, best) )

        if len(linked_phrases) != len(phrases):
            print("Warning: length of output is not equal to length of input {} != {}".format(len(best), len(phrases)))
        
        return linked_phrases
   
