from flask import Flask, request, Response
import logging
import requests


endpoint = "http://localhost:8181/spotlight"


app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger("proxy.py")


@app.route("/", methods=['POST'])
def hello():
    h = {key: value for key, value in request.headers}
    r = requests.post(endpoint, headers=h, data=request.data)

    resp = Response()
    if r.status_code == 200:
        for header_name, header_value in r.headers.items():
            resp.headers[header_name] = header_value
        resp.data = r.content
    else:
        print("Warning: server returned an error")
        print(r)

    return resp

