from linkers.baseline import BaselineLinker
from candidate import Phrase

context = "San Francisco said the visit would serve as a cornerstone for future interaction between players and coaches from the Nets and young Russians, with the aim of developing basketball in Russia, where the sport is a distant third in popularity behind soccer and hockey."
phrases = "San Francisco"

phrases =  [Phrase(phrase.strip(), 0, len(phrase.strip()), "http://" + phrase.strip())
                   for phrase in phrases.split(",")]
bl = BaselineLinker()

for phrase, candidate in bl.link(context, phrases):
    print(phrase.text, candidate)