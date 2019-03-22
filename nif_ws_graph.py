from flask import Flask, request, Response
from linkers.nn_graph import NNLinker


host = "127.0.0.1"

app = Flask(__name__)
app.debug = False

nn_linker = NNLinker()


@app.route("/nngraph", methods=['POST'])
def nngraph():
    response = Response()

    for header_name, header_value in request.headers.items():
        response.headers[header_name] = header_value
    response.data = nn_linker.link_ttl(request.data)

    return response


if __name__ == "__main__":
    app.run(host=host, threaded=False)