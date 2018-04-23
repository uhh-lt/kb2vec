from collections import namedtuple

CandidateBase = namedtuple("Candidate", ["score", "name", "wiki", "types", "names", "uris"])

class Candidate(CandidateBase):
    def __hash__(self):
        return hash(self.score) * hash(self.name) * hash("".join(uris) * hash("".join(types)))

