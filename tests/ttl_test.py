from ttl import parse_d2kb_ttl
import codecs

input_ttl_fpaths = ["../datasets/kore50.ttl", "../datasets/n3-reuters-128.ttl", "../datasets/dbpedia.ttl"]

for input_ttl_fpath in input_ttl_fpaths:
    in_ttl = codecs.open(input_ttl_fpath, "r", "utf-8")
    output_ttl_fpath = input_ttl_fpath + ".tsv"
    out_ttl = codecs.open(output_ttl_fpath, "w", "utf-8")

    input_ttl = in_ttl.read()
    graph, context, phrases = parse_d2kb_ttl(input_ttl)


    out_ttl.write("{}\t{}".format(
        ", ".join([p.text for p in phrases]),
        context
    ))
    in_ttl.close()
    out_ttl.close()

    print("Output:", output_ttl_fpath)

