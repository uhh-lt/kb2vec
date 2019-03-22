from flask import Flask, request, Response
import logging
import requests
import codecs
from os.path import join
from time import time
from ttl import remove_classref, DatasetBuilder
from linkers.nn_graph import NNLinker

endpoint = "http://localhost:8080/spotlight"
data_dir = "data/"
no_classref = False
save_ttl_data = False
ds = DatasetBuilder(join(data_dir, "dataset.csv"))

app = Flask(__name__)
app.debug = False


# logging.basicConfig(level=logging.DEBUG)
# log = logging.getLogger("nif_ws.py")


def save_data(prefix, req_data, resp_data):
    if save_ttl_data:
        fid = prefix + "-" + str(time()).replace(".", "")
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


nn_linker = NNLinker()

@app.route("/nngraph", methods=['POST'])
def nngraph():
    response = Response()

    for header_name, header_value in request.headers.items():
        response.headers[header_name] = header_value
    response.data = nn_linker.link_ttl(request.data)

    save_data("nngraph", request.data, response.data)

    return response


if __name__ == "__main__":
    app.run(host="127.0.0.1", threaded=True)