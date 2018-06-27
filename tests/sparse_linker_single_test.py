from linkers.sparse import SparseLinker
from candidate import make_phrases


dataset_fpaths = ["../datasets/singleton.tsv"]

sl = SparseLinker("../data/single")
sl.train(dataset_fpaths)
context = "Richard Stallman, often known by his initials, rms — is an American free software movement activist and programmer. He campaigns for software to be distributed in a manner such that its users receive the freedoms to use, study, distribute and modify that software."
phrases = ["Richard Stallman"]
linked_phrases = sl.link(context, make_phrases(phrases))
print(linked_phrases)