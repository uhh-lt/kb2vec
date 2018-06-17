from linkers.supertagger import SuperTagger
from candidate import Phrase
import codecs


st = SuperTagger()


def make_positional_phrases(word_beg_ends):
    phrases = []
    for word, beg, end in word_beg_ends:
        phrases.append(Phrase(word, beg, end, "http://www.{}.com".format(word)))
    return phrases

request_fpath = "../data/supertagger-1529250101365435-request.ttl"
with codecs.open(request_fpath, "r", "utf-8") as ttl:
    input_ttl = ttl.read()

output_ttl = st.link_ttl(input_ttl)
with codecs.open(request_fpath + ".response", "w", "utf-8") as ttl:
    ttl.write(output_ttl)

context = "Prokhorov said the visit would serve as a cornerstone for future interaction between players and coaches from the Nets and young Russians, with the aim of developing basketball in Russia, where the sport is a distant third in popularity behind soccer and hockey."
phrases = make_positional_phrases([["Russia", 180, 186],
                                  ["sport", 198, 203],
                                  ["basketabll", 166, 176],
                                  ["Russians", 129, 137],
                                  ["Prokhorov", 0, 9]])
linked_phrases = st.link(context, phrases)
print(linked_phrases)

context = "Madonna is a great music  signer and lives near West Holywood in Los Angeles. adonna Louise Ciccone (/tʃɪˈkoʊni/; born August 16, 1958) is an American singer, songwriter, actress, and businesswoman. Referred to as the Queen of Pop since the 1980s, Madonna is known for pushing the boundaries of lyrical content in mainstream popular music, as well as visual imagery in music videos and on stage. She has also frequently reinvented both her music and image while maintaining autonomy within the recording industry. Besides sparking controversy, her works have bee "
phrases = [Phrase("Madonna", 0, 6, "http://madonna.com"),
           Phrase("West Holywood", 48, 62, "http://westholy.com"),
           Phrase("Los Angeles", 65, 76, "http://la.com")]

linked_phrases = st.link(context, phrases)
print(linked_phrases)

