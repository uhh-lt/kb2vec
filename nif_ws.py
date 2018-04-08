from flask import Flask, request, Response
import logging
import requests
import codecs
from os.path import join
from time import time
import re
from ttl import remove_classref, add_nonsense_response, DatasetBuilder


endpoint = "http://localhost:8080/spotlight"
data_dir = "data/"
no_classref = False
ds = DatasetBuilder(join(data_dir, "dataset.csv"))

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger("nif_ws.py")


def save_data(prefix, req_data, resp_data):
    fid = prefix + "-" + str(time()).replace(".","")
    request_fpath = join(data_dir, fid + "-request.ttl") 
    with codecs.open(request_fpath, "w", "utf-8") as req:
        req.write(str(req_data, "utf-8"))

    response_fpath = join(data_dir, fid + "-response.ttl") 
    with codecs.open(response_fpath, "w", "utf-8") as res:
        res.write(resp_data)


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


@app.route("/mfs", methods=['POST'])
def mfs():
    h = {key: value for key, value in request.headers}
    
    resp_data = mfs(request.data)

    resp = Response()
    for header_name, header_value in request.headers.items():
        resp.headers[header_name] = header_value
    resp.data = resp_data
    save_data("mfs", request.data, resp_data)
    ds.add_to_dataset(request.data)
    
    return resp

