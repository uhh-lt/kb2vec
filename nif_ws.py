from flask import Flask, request, Response
import logging
import requests
import codecs
from os.path import join
from time import time
from ttl import remove_classref, add_nonsense_response, DatasetBuilder
from linkers.baseline import BaselineLinker
from linkers.sparse import SparseLinker
from linkers.dense import DenseLinker
from linkers.supertagger import SuperTagger


endpoint = "http://localhost:8080/spotlight"
data_dir = "data/"
no_classref = False
save_ttl_data = True
ds = DatasetBuilder(join(data_dir, "dataset.csv"))

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger("nif_ws.py")


def save_data(prefix, req_data, resp_data):
    if save_ttl_data:
        fid = prefix + "-" + str(time()).replace(".","")
        request_fpath = join(data_dir, fid + "-request.ttl")
        with codecs.open(request_fpath, "w", "utf-8") as req:
            req.write(str(req_data, "utf-8"))

        response_fpath = join(data_dir, fid + "-response.ttl")
        with codecs.open(response_fpath, "w", "utf-8") as res:
            res.write(str(resp_data, "utf-8"))


@app.route("/proxy", methods=['POST'])
def proxy():
    h = {key: value for key, value in request.headers}
    r = requests.post(endpoint, headers=h, data=request.data)

    resp = Response()
    if r.status_code == 200:
        for header_name, header_value in r.headers.items():
            resp.headers[header_name] = header_value
        
        r_content = str(r.content, "utf-8")
        resp_data = remove_classref(r_content) if no_classref else r_content
        resp.data = resp_data
        save_data("proxy", request.data, resp_data)
        ds.add_to_dataset(request.data)
    else:
        log.info("Warning: server returned an error")
        log.info(r)

    return resp


@app.route("/trivial", methods=['POST'])
def trivial():
    h = {key: value for key, value in request.headers}
    
    resp_data = add_nonsense_response(request.data)

    resp = Response()
    for header_name, header_value in request.headers.items():
        resp.headers[header_name] = header_value
    resp.data = resp_data
    save_data("trivial", request.data, resp_data)
    ds.add_to_dataset(request.data)
    
    return resp


overlap_importance_linker = BaselineLinker(use_overlap=True, use_importance=True)

@app.route("/overlap_importance", methods=['POST'])
def overlap_importance():
    response = Response()
    
    for header_name, header_value in request.headers.items():
        response.headers[header_name] = header_value
    response.data = overlap_importance_linker.link_ttl(request.data)

    save_data("overlap_importance", request.data, response.data)
    
    return response


importance_linker = BaselineLinker(use_overlap=False, use_importance=True)

@app.route("/importance", methods=['POST'])
def importance():
    response = Response()
    
    for header_name, header_value in request.headers.items():
        response.headers[header_name] = header_value
    response.data = importance_linker.link_ttl(request.data)

    save_data("importance", request.data, response.data)
    
    return response


overlap_linker = BaselineLinker(use_overlap=True, use_importance=False, lower=True)

@app.route("/overlap", methods=['POST'])
def overlap():
    response = Response()
    
    for header_name, header_value in request.headers.items():
        response.headers[header_name] = header_value
    response.data = overlap_linker.link_ttl(request.data)

    save_data("overlap", request.data, response.data)
    
    return response


overlap_linker_case = BaselineLinker(use_overlap=True, use_importance=False, lower=False)

@app.route("/overlap_case", methods=['POST'])
def overlap_case():
    response = Response()

    for header_name, header_value in request.headers.items():
        response.headers[header_name] = header_value
    response.data = overlap_linker_case.link_ttl(request.data)

    save_data("overlap_case", request.data, response.data)

    return response


random_linker = BaselineLinker(use_overlap=False, use_importance=False)

@app.route("/random", methods=['POST'])
def random():
    response = Response()
    
    for header_name, header_value in request.headers.items():
        response.headers[header_name] = header_value
    response.data = random_linker.link_ttl(request.data)

    save_data("random", request.data, response.data)
    
    return response

dense_linker = DenseLinker("data/count-stopwords-3", "data/wiki-news-300d-1M.vec")
# dense_linker = DenseLinker("data/count-stopwords-3-cc", "data/crawl-300d-2M.vec")

@app.route("/dense_overlap", methods=['POST'])
def dense_overlap():
    params = {"tfidf": False, "use_overlap": True}
    dense_linker.set_params(params)

    response = Response()

    for header_name, header_value in request.headers.items():
        response.headers[header_name] = header_value
    response.data = dense_linker.link_ttl(request.data)

    save_data("dense_overlap", request.data, response.data)

    return response


# sparse_linker = SparseLinker("data/all0")
# sparse_linker = SparseLinker("data/tfidf-stopwords-2")
sparse_linker = dense_linker # SparseLinker("data/count-stopwords-3")

@app.route("/sparse", methods=['POST'])
def sparse():
    params = {"tfidf": True, "use_overlap": False}
    sparse_linker.set_params(params)

    response = Response()
    
    for header_name, header_value in request.headers.items():
        response.headers[header_name] = header_value
    response.data = sparse_linker.link_ttl(request.data)

    save_data("sparse", request.data, response.data)
    
    return response



@app.route("/sparse_overlap", methods=['POST'])
def sparse_overlap():
    params = {"tfidf": True, "use_overlap": True}
    sparse_linker.set_params(params)

    response = Response()
    
    for header_name, header_value in request.headers.items():
        response.headers[header_name] = header_value
    response.data = sparse_linker.link_ttl(request.data)

    save_data("sparse_overlap", request.data, response.data)
    
    return response


super_linker = SuperTagger()
@app.route("/supertagger", methods=['POST'])
def supertagger():
    response = Response()

    for header_name, header_value in request.headers.items():
        response.headers[header_name] = header_value
    response.data = super_linker.link_ttl(request.data)

    save_data("supertagger", request.data, response.data)

    return response


if __name__ == "__main__":
    app.run(host="127.0.0.1", threaded=True)