import requests
import json
import codecs
import grequests 
from sqlitedict import SqliteDict
from utils import ROOT_DIR
from os.path import join


endpoint_diffbot = "http://kg.diffbot.com/kg/dql_endpoint"

ENTITY_TYPES = ["AdministrativeArea", "Article", "Corporation",
                    "DegreeEntity", "EducationMajorEntity", "EducationalInstitution",
                   "EmploymentCategory", "Image", "Intangible", "Landmark", "LocalBusiness",
                   "Miscellaneous", "Organization", "Person", "Place", "Product", "Role",
                   "Skill", "Video"]

EL_ENTITY_TYPES = ["AdministrativeArea", "Corporation", "EducationalInstitution",
                   "Landmark", "LocalBusiness", "Miscellaneous", "Organization", 
                   "Person", "Place", "Product"]

EL_POL_ENTITY_TYPES = ["AdministrativeArea", "Corporation", "EducationalInstitution",
                   "Landmark", "LocalBusiness", "Organization", 
                   "Person", "Place", "Product"]

CACHED_QUERY_DB =  join(join(ROOT_DIR, "cache"), "diffbot-query-cache.sqlite")


class CachedQuery(object):
    def __init__(self, cache_fpath=CACHED_QUERY_DB):
        self._cache = SqliteDict(cache_fpath, autocommit=True)
        
    def __del__(self):
        try:
            self._cache.close()
        except:
            print("Warning: trying to close a closed cache.")
            
    def make_query(self, query):
        if query in self._cache:
            return self._cache[query]
        else:
            response = make_query(query)
            self._cache[query] = response
            return response
        
    def close(self):
        self._cache.close()

    def response2dict(self, response):
        return json.loads(response.content)

    def get_entity(self, db_uri):
        if db_uri in self._cache:
            return self._cache[db_uri]
        else:
            response = self._get_entity(db_uri)
            self._cache[db_uri] = response
            return response

    def _get_entity(self, db_uri):
        """ Takes as input URI like http://www.diffbot.com/entity/CQSNBJBdRL7 and returns
        and entity. """

        db_uri = db_uri.replace("https:", "http:")

        data = {
            "token": get_token(),
            "query": "diffbotUri:{}".format(db_uri),
            "type": "query"}

        r = requests.get(endpoint_diffbot, params=data)

        return self.response2dict(r)


# https://dev.kg.diffbot.com/kg/dql_endpoint?type=query&token=token&query=id:OIZzlT1rihy
# https://www.diffbot.com/entity/OIZzlT1rihy


token = None
def get_token():
    global token    
    if token:
        return token
    else:
        with open("dbt", "r") as f:
            token = f.read().strip()
            return token


def make_queries(queries, parallel=32):    
    rs = []
    for query in queries:
        data = {
            "token": get_token(),
            "query": query,
            "type": "query"}

        rs.append(grequests.get(endpoint_diffbot, params=data))
    
    return grequests.map(rs, size=parallel)


def make_query(query):
    data = {
        "token": get_token(),
        "query": query,
        "type": "query"}
    r = requests.get(endpoint_diffbot, params=data)

    return r 


def save2json(output_fpath, r):
    with codecs.open(output_fpath, "w", "utf-8") as out:   
        out.write(json.dumps(json.loads(r.content)))
    print(output_fpath)
 

def query_and_save(query, output_fpath):
    r = make_query(query)
    save2json(output_fpath, r)





