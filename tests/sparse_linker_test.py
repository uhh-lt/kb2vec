from linkers.sparse import SparseLinker
from candidate import make_phrases

dataset_fpaths = ["../datasets/dbpedia.ttl.phrases.tsv",
                  "../datasets/kore50.ttl.phrases.tsv",
                  "../datasets/n3-reuters-128.ttl.phrases.tsv"]


dataset_fpaths = ["../datasets/test.phrases.tsv"]

sl = SparseLinker("../data/count-stopwords-11-related", stop_words=True, tfidf=False, related_entities=True)
sl.train(dataset_fpaths)

context = "Madonna is a great music  signer and lives near West Holywood in LA. adonna Louise Ciccone (/tʃɪˈkoʊni/; born August 16, 1958) is an American singer, songwriter, actress, and businesswoman. Referred to as the Queen of Pop since the 1980s, Madonna is known for pushing the boundaries of lyrical content in mainstream popular music, as well as visual imagery in music videos and on stage. She has also frequently reinvented both her music and image while maintaining autonomy within the recording industry. Besides sparking controversy, her works have bee "
phrases = ["Madonna"]
linked_phrases = sl.link(context, make_phrases(phrases))
print(linked_phrases)

context = "Richard Stallman, often known by his initials, rms — is an American free software movement activist and programmer. He campaigns for software to be distributed in a manner such that its users receive the freedoms to use, study, distribute and modify that software."
phrases = ["Richard Stallman"]
linked_phrases = sl.link(context, make_phrases(phrases))
print(linked_phrases)

context = "Linus Benedict Torvalds (/ˈliːnəs ˈtɔːrvɔːldz/;[5] Swedish: [ˈliːn.ɵs ˈtuːr.valds] (About this sound listen); born December 28, 1969) is a Finnish-American software engineer[2][6] who is the creator, and historically, the principal developer of the Linux kernel, which became the kernel for operating systems such as the Linux operating systems, Android, and Chrome OS."
phrases = ["Linus Torvalds"]
linked_phrases = sl.link(context, make_phrases(phrases))
print(linked_phrases)
