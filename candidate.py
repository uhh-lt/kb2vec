from collections import namedtuple

Phrase = namedtuple('Phrase', ['text', 'beg', 'end', 'subj'])

CandidateBase = namedtuple("Candidate", ["score", "name", "link", "wiki", "types", "names", "uris"])

class Candidate(CandidateBase):
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

