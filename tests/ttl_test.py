from ttl import parse_d2kb_ttl
import codecs

input_ttl_fpaths = ["../datasets/kore50.ttl", "../datasets/n3-reuters-128.ttl", "../datasets/dbpedia.ttl"]

for input_ttl_fpath in input_ttl_fpaths:
    in_ttl = codecs.open(input_ttl_fpath, "r", "utf-8")
    phrases_fpath = input_ttl_fpath + ".phrases.tsv"
    contexts_fpath = input_ttl_fpath + ".contexts.tsv"

    phrases_ttl = codecs.open(phrases_fpath, "w", "utf-8")
    contexts_ttl = codecs.open(contexts_fpath, "w", "utf-8")

    input_ttl = in_ttl.read()
    graph, contexts, phrases = parse_d2kb_ttl(input_ttl)

    for phrase in phrases:
        phrases_ttl.write("{}\t    \n".format(phrase.text))

    for context in contexts:
        contexts_ttl.write("    \t{}\n".format(context))

    in_ttl.close()
    phrases_ttl.close()
    contexts_ttl.close()

    print("Output:", phrases_fpath)
    print("Output:", contexts_fpath)