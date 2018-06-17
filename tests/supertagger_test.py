from linkers.supertagger import SuperTagger
from candidate import Phrase

st = SuperTagger()
context = "Madonna is a great music  signer and lives near West Holywood in Los Angeles. adonna Louise Ciccone (/tʃɪˈkoʊni/; born August 16, 1958) is an American singer, songwriter, actress, and businesswoman. Referred to as the Queen of Pop since the 1980s, Madonna is known for pushing the boundaries of lyrical content in mainstream popular music, as well as visual imagery in music videos and on stage. She has also frequently reinvented both her music and image while maintaining autonomy within the recording industry. Besides sparking controversy, her works have bee "
phrases = [Phrase("Madonna", 0, 6, "http://madonna.com"),
           Phrase("West Holywood", 48, 62, "http://westholy.com"),
           Phrase("Los Angeles", 65, 76, "http://la.com")]

linked_phrases = st.link(context, phrases)
print(linked_phrases)