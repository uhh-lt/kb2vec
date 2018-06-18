from linkers.baseline import BaselineLinker
from candidate import Phrase
from pandas import read_csv

dataset_fpath = "../datasets/dbpedia.tsv"

df = read_csv(dataset_fpath, sep="\t", encoding="utf-8")
bl = BaselineLinker()

for i, row in df.iterrows():
    phrases = [Phrase(phrase.strip(), 1, len(phrase.strip()), "http://" + phrase.strip())
               for phrase in row.targets.split(",")]

    print("\n\n{}\n".format(row.context))

    for phrase, candidate in bl.link(row.context, phrases):
        link = candidate.link if candidate else ""
        print(phrase.text, link)