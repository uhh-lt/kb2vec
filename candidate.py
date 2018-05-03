from collections import namedtuple
from namedlist import namedlist
import codecs


Phrase = namedtuple("Phrase", "text beg end subj")

CandidateBase = namedlist("CandidateBase", "score name link wiki types names uris text db_uri")


def make_phrases(str_phrases):
    """ From a list of strings generates a list of phrases (e.g. for tests)""" 

    return [Phrase(phrase.strip(), 1, len(phrase.strip()), "http://" + phrase.strip())
                   for phrase in str_phrases]


class Candidate(CandidateBase):
    def __init__(self, score=0.0, name="", link="", wiki="", types=[], names=[], uris=[], text="", db_uri=""):
       CandidateBase.__init__(self, score, name, link, wiki, types, names, uris, text, db_uri)

    def get_hash(self):
        uris = "".join(self.uris) if self.uris is not None else ""
        types = "".join(self.types) if self.types is not None else ""
        hash_str = str(self.score) + self.name + uris + types
        if hash_str is None:
            print("Warning: hash string is none.")

        return hash(hash_str)

    def __hash__(self):
        return self.get_hash()
    
    def __eq__(self, other):
        return self.get_hash() == other.get_hash()

    #def __gt__(self, other):
    #    return self.age > other.age

    #def __lt__(self, other):
    #    return self.age < other.age


def save_candidates_text(output_fpath="data/sf-candidates.txt"):
    re_newlines = re.compile(r"[\n\r]+")

    with codecs.open(output_fpath, "w", "utf-8") as c_f:
        for phrase in c:
            for candidate in c[phrase]:
                text = candidate.text
                c_f.write("{}\t{}\t{}\n".format(
                    phrase.text,
                    candidate.name,
                    text.strip()))
                
    print(output_fpath)
