from wikidata.client import Client
from traceback import format_exc
from sqlitedict import SqliteDict
from traceback import format_exc


WIKIDATA_DOMAIN = "wikidata.org"
WIKIPEDIA_DOMAIN = "wikipedia.org"
WIKIDATA_PREFIX = "wikidata.org/wiki/"
DBPEDIA_PREFIX = "http://dbpedia.org/resource/"
WIKIPEDIA_PREFIX = "wikipedia.org/wiki/"
CACHED_WIKI2DBPEDIA_DB = "wikidata2dbpedia-cache.sqlite"
verbose = False
  

class URIConverter(object):
    def __init__(self, cache_fpath=CACHED_WIKI2DBPEDIA_DB):
        self._cache = SqliteDict(cache_fpath, autocommit=True)
        self._client = Client()

    def __del__(self):
        try:
            self._cache.close()
        except:
            if verbose: print("Warning: trying to close a closed cache.")

    def close(self):
        self._cache.close()

    def get_postfix(self, string, prefix):
        """ Given a string and a prefix returns postfix. If not found 
        then returns None. """

        beg_index = string.find(prefix)
        if beg_index != -1:
            end_index = beg_index + len(prefix)
            return string[end_index:]
        else:
            return None

    def get_fuzzy_postfix(self, string, prefix):
        if prefix in string:
            parts = string.split("/")
            if len(parts) > 1:
                return parts[-1]
            else:
                return None
        
    def wikipedia2dbpedia(self, wikipedia_uri):
        article_name = self.get_fuzzy_postfix(wikipedia_uri, prefix=WIKIPEDIA_DOMAIN)

        if article_name is None:
            if verbose: print("Warning: cannot convert to DBpedia URI '{}'".format(wikipedia_uri))
            return ""
        else:
            return DBPEDIA_PREFIX + article_name                

    def wikidataid2wikipedia(self, wikidata_q_id="Q42"):
        try:
            if wikidata_q_id in self._cache:
                return self._cache[wikidata_q_id]
            else:
                entity = self._client.get(wikidata_q_id, load=True)
                can_get = ("sitelinks" in entity.attributes and
                           "enwiki" in entity.attributes["sitelinks"] and
                           "url" in entity.attributes["sitelinks"]["enwiki"])
                if can_get:
                    wikipedia_uri = entity.attributes["sitelinks"]["enwiki"]["url"]
                    self._cache[wikidata_q_id] = wikipedia_uri
                    return wikipedia_uri
                else:
                    wiki_links = []
                    for key in entity.attributes["sitelinks"]:
                        if key.endswith("wiki"):
                            if "url" in entity.attributes["sitelinks"][key]:
                                wiki_links.append(entity.attributes["sitelinks"][key]["url"])
                    
                    if len(wiki_links) > 0:
                        print("Warning: no links to English Wiki found, but found {} links to other Wikis".format(len(wiki_links)))
                        self._cache[wikidata_q_id] = wiki_links[0]
                        return wiki_links[0]
                    else:
                        self._cache[wikidata_q_id] = ""
                        return ""

        except KeyboardInterrupt:
            raise KeyboardInterrupt()
        except:
            print("Warning: cannot process '{}'".format(wikidata_q_id))
            print(format_exc())
            return ""

    def get_wikidata_id(self, wikidata_uri):
        wikidata_id = self.get_fuzzy_postfix(wikidata_uri, prefix=WIKIDATA_DOMAIN)
        if wikidata_id is None:
            if verbose: print("Warning: cannot extract WikiData ID '{}'".format(wikidata_uri))
            return ""
        else:
            return wikidata_id

    def wikidata2wikipedia(self, wikidata_uri):
        wikidata_id = self.get_wikidata_id(wikidata_uri)
        if wikidata_id != "":
            wikipedia_uri = self.wikidataid2wikipedia(wikidata_id)
            return wikipedia_uri
        else:
            if verbose: print("Warning: cannot extract DBpedia URI from a Wikidata URI")
            return ""


    def wikidata2dbpedia(self, wikidata_uri):
        return self.wikipedia2dbpedia(self.wikidata2wikipedia(wikipedia_uri))

