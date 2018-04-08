from pandas import read_csv
from glob import glob 
import re
from traceback import format_exc


def dataset_stat(dataset_fpath):
    try:
        df = read_csv(dataset_fpath, sep="\t", encoding="utf-8")
        df.targets
    except AttributeError:
        df = read_csv(dataset_fpath, sep="\t", encoding="utf-8", names=["targets","context"])
        
    targets = set()
    for i, row in df.iterrows():
        for t in str(row.targets).split(","):
            ts = t.strip()
            if len(ts) > 0: targets.add(ts)

    print("# of contexts:", len(df))
    print("# of targets:", len(targets))


def format_urls(url_fpaths):
    url = re.compile(r"<([^>]+)>")
    for url_fpath in glob(url_fpaths):
        print(url_fpath)
        with open(url_fpath, "r") as in_f, open(url_fpath + ".out", "w") as out_f:
            for line in in_f:
                match = url.search(line)
                if match:
                    out_f.write("{}\n".format(match.groups(0)[0]))


datasets_fpath = "/home/panchenko/kb2vec/datasets/*.tsv"
for dataset_fpath in glob(datasets_fpath):
    
    print(dataset_fpath)
    dataset_stat(dataset_fpath)
    

format_urls(url_fpaths = "datasets/*txt")
