from glob import glob
from os.path import join 
from diffbot_api import dbpedia2wikipedia, make_query
from traceback import format_exc
import json


def get_hits(diffbot_query_response):
    hits_num = diffbot_query_response["hits"]
    data = diffbot_query_response["data"]

    types = []
    for i, hit in enumerate(diffbot_query_response["data"]):
        types.append(hit["type"])

        
    return hits_num, types


def generate_absent_datasets(datasets_fpaths):
    saved = None

    for dataset_fpath in glob(datasets_fpaths):
        print(dataset_fpath)
        total_hits = 0
        total_urls = 0
        total_absent = 0

        with open(dataset_fpath, "r") as in_f, open(dataset_fpath + ".absent", "w") as out_f:
            for url in in_f:
                try:
                    url = dbpedia2wikipedia(url.strip())
                    query = 'origins:"{}"'.format(url)
                    r = make_query(query)
                    db_response = json.loads(r.content)

                    hits_num, types = get_hits(db_response)
                    if url == "en.wikipedia.org/wiki/Russians": saved = db_response
                    print(".", end="")  
                    total_urls += 1
                    if hits_num == 0:
                        total_absent += 1
                        out_f.write("{}\n".format(url))
                    total_hits += hits_num

                except KeyboardInterrupt:
                    break
                except:
                    print(url, "error")
                    print(format_exc())
            print("\n")

        print("Absent urls:", total_absent)
        print("Total urls:", total_urls)
        print("Hits total for all urls:", total_hits)
        print("Avg. hits per url: {:.2f}".format(float(total_hits)/total_urls))


datasets_fpaths = "datasets/*txt"
generate_absent_datasets(datasets_fpaths)
