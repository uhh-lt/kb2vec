from linkers.baseline import BaselineLinker
from candidate import Candidate
from supervised.evaluate import Evaluator


class NNLinker(BaselineLinker):
    def __init__(self):
        BaselineLinker.__init__(self)
        self.evaluator = Evaluator()

    def link(self, context, phrases):

        linked_phrases = list()

        #file = open('/Users/sevgili/Desktop/context-phrase-nif.txt', 'a')

        for phrase in phrases:
            #file.write(str(context) + '\t' + str(phrase.text) + '\t' + str(phrase.beg) + '\t' + str(phrase.end) + '\n')
            score, predicted_url = self.evaluator.get_best_pred(context, phrase)
            print('******', context, phrase, score, predicted_url)
            c = Candidate(score=score, link=predicted_url)

            linked_phrases.append((phrase, c))

        return linked_phrases


class CandidateRandom(NNLinker):
    def __init__(self):
        NNLinker.__init__(self)

    def link(self, context, phrases):

        linked_phrases = list()

        for phrase in phrases:
            score, predicted_url = self.evaluator.get_random_pred(context, phrase)

            c = Candidate(score=score, link=predicted_url)

            linked_phrases.append((phrase, c))

        return linked_phrases
