from flask import Flask, request, Response
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger("proxy.py")


@app.route("/", methods=['POST'])
def hello():
    log.info(request.headers)
    log.info(request.data)


    # add here the original headers
    resp = Response("Foo bar baz")
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.data = request.data
    return resp

