from linkers.baseline import BaselineLinker
from collections import defaultdict
from diffbot_api import EL_POL_ENTITY_TYPES
import json
from candidate import Candidate
from langid import classify
from candidate import Phrase
import re
from tqdm import tqdm


class ContextAwareLinker(BaselineLinker):
    """ A base class for linkers that make use of textual representations of entities. """

    def __init__(self):
        BaselineLinker.__init__(self)
        self._re_contains_alpha = re.compile(r"[a-z]+", re.U|re.I)
        self._re_newlines = re.compile(r"[\n\r]+")
        self._sep = " . "

    def get_db_entry(diffbot_uri):
        """ Gets an entity like https://www.diffbot.com/entity/AcZTRPXDrY9 and 
        returns a json by https://www.diffbot.com/entity/AcZTRPXDrY9.json """
        
        raise NotImplementedError 
        return {}

    def _is_english(self, text):
        lang, conf = classify(text)
        return lang == "en"

    def _is_alpha(self, text):
        return self._re_contains_alpha.search(text)
    
    def _get_en_names(self, hit):
        names = []
        
        if "allNames" in hit:
            for name in hit["allNames"]:
                if self._is_alpha(name) and self._is_english(name):                    
                    names.append(name)
        
        return names 
    
    def _get_name(self, hit):
        if "name" in hit:
            return hit["name"]
        else:
            return ""
        
    def _get_record_texts(self, hit):
        texts = [ self._get_name(hit) ]        
        texts += self._get_en_names(hit)

        if "isPartOf" in hit:
            for is_part_of in hit["isPartOf"]:
                if "name" in is_part_of:
                    texts.append(is_part_of["name"])
                             
        if "description" in hit:
            texts.append(hit["description"])

        texts_str = self._sep.join(texts)

        return texts_str
    
    def _get_wiki_texts(self, wiki_uri):
        # access from a cached (?) wikipedia dump
        return ""
    
    def _get_uri_texts(self, uris):
        # access the uris
        return ""
 
    def get_candidates(self, phrases):
        phrase2candidates = defaultdict(set)  
        i = 0
        for phrase in tqdm(phrases):
            for entity_type in EL_POL_ENTITY_TYPES:
                r = self._cq.make_query('type:{} name:"{}"'.format(entity_type, phrase.text))
                db_response = json.loads(r.content)
            
                if "data" not in db_response: continue
                else: data = db_response["data"]

                for hit in data:
                    uris = self._get_uris(hit)
                    wiki_uri = self._get_wikipedia_uri(hit, uris)  
                    
                    texts_record = self._get_record_texts(hit)
                    texts_wiki = self._get_wiki_texts(wiki_uri) 
                    texts_uris = self._get_uri_texts(uris)                    
                    texts = self._sep.join([texts_record, texts_wiki, texts_uris])
                    texts = self._re_newlines.sub(self._sep, texts)

                    score = float(hit["importance"])
                    link = self._get_dbpedia_uri(wiki_uri, uris) 
                    c = Candidate(score,
                                  self._get_name(hit),
                                  link,
                                  wiki_uri,
                                  hit["types"],
                                  self._get_en_names(hit),
                                  uris,
                                  texts)
                    i += 1
                    phrase2candidates[phrase].add(c)
        
        return phrase2candidates

