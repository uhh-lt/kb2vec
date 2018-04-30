from collections import namedtuple
from namedlist import namedlist


Phrase = namedtuple("Phrase", "text beg end subj")

CandidateBase = namedlist("CandidateBase", "score name link wiki types names uris text")

class Candidate(CandidateBase):
    def __init__(self, score=0.0, name="", link="", wiki="", types=[], names=[], uris=[], text=""):
       CandidateBase.__init__(self, score, name, link, wiki, types, names, uris, text)

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
