from linkers.sparse import SparseLinker
from candidate import make_phrases


dataset_fpaths = ["../datasets/singleton.tsv"]

sl = SparseLinker("../data/single5")
# sl.train(dataset_fpaths)
context = "Richard Stallman, often known by his initials, rms — is an American free software movement activist and programmer. He campaigns for software to be distributed in a manner such that its users receive the freedoms to use, study, distribute and modify that software."
phrases = ["Richard Stallman"]
linked_phrases = sl.link(context, make_phrases(phrases))
print(linked_phrases)

context = "Linus Benedict Torvalds (/ˈliːnəs ˈtɔːrvɔːldz/;[5] Swedish: [ˈliːn.ɵs ˈtuːr.valds] (About this sound listen); born December 28, 1969) is a Finnish-American software engineer[2][6] who is the creator, and historically, the principal developer of the Linux kernel, which became the kernel for operating systems such as the Linux operating systems, Android, and Chrome OS."
phrases = ["Linus Torvalds"]
linked_phrases = sl.link(context, make_phrases(phrases))
print(linked_phrases)