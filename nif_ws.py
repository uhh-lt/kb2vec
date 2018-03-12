from flask import Flask, request, Response
import logging
import requests
import codecs
from os.path import join
from time import time
import re
    

endpoint = "http://localhost:8181/spotlight"
data_dir = "data/"
no_classref = True


app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger("nif_ws.py")


def remove_classref(text):
    output = []
    for line in text.split("\n"):
        upd_line = re.sub(r"itsrdf:taClassRef     <[^;]*> ;",
                          "itsrdf:taClassRef     <nonsense> ;",
                          line)
        output.append(upd_line)
        
    return "\n".join(output)


@app.route("/proxy", methods=['POST'])
def proxy():
    h = {key: value for key, value in request.headers}
    r = requests.post(endpoint, headers=h, data=request.data)

    resp = Response()
    if r.status_code == 200:
        for header_name, header_value in r.headers.items():
            resp.headers[header_name] = header_value
        resp_data = remove_classref(r.content) if no_classref else r.content
        resp.data = resp_data

        fid = str(time()).replace(".","")
        request_fpath = join(data_dir, fid + "-request.ttl") 
        with codecs.open(request_fpath, "w", "utf-8") as req:
            req.write(request.data.decode("utf-8"))

        response_fpath = join(data_dir, fid + "-response.ttl") 
        with codecs.open(response_fpath, "w", "utf-8") as res:
            res.write(resp_data.decode("utf-8"))

    else:
        log.info("Warning: server returned an error")
        log.info(r)

    return resp

