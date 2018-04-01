import requests
from pprint import pprint
import json 
import codecs


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


def dbpedia2wikipedia(url, to_en=True):
    
    url = url.replace("https://", "")
    url = url.replace("http://", "")
    
    if to_en:
        wiki_domain = "en.wikipedia.org/wiki/"
    else:
        wiki_domain = "wikipedia.org/wiki/"
        
    new_url = url.replace("dbpedia.org/resource/", wiki_domain)
    if new_url == url:
        new_url = url.replace("dbpedia.org/page/", wiki_domain)
        
    return new_url


def get_token():
    with open("../.diffbot-token", "r") as f:
        return f.read().strip()


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
    
   
