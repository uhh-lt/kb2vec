from supervised.train import Trainer
from supervised.preprocess.prepro_train import InputVecGenerator
from nltk.tokenize import word_tokenize
from supervised.preprocess.prepro_util import Sample
from collections import namedtuple
from operator import itemgetter
import random


Phrase = namedtuple("Phrase", "text beg end subj")


class Evaluator(object):
    def __init__(self):
        trainer = Trainer()
        trainer.restore_sess()
        self.sess = trainer.sess
        self.pred = trainer.model.pred
        self.X = trainer.model.X
        self.keep_prob = trainer.model.keep_prob
        self.input_generator = InputVecGenerator()
        self.fetchFilteredCoreferencedCandEntities = self.input_generator.sample_generator.fetchFilteredCoreferencedCandEntities
        self.graphid2url = self.input_generator.graphid2url

    def derive_prediction_scores(self, context, phrase):
        chunk_words = word_tokenize(context)

        begin = len(word_tokenize(context[:phrase.beg]))
        end = len(word_tokenize(context[:phrase.end]))
        candidates, scores = self.fetchFilteredCoreferencedCandEntities.process(begin, end, chunk_words)
        #print(candidates)
        if candidates is None:
            return []
        # "chunk_id", "chunk_words", 'begin_gm', "end_gm", "ground_truth", "cand_entities",
        # "cand_entities_scores"
        sample = Sample(0, chunk_words, begin, end, [0], candidates, scores)

        predictions = list()
        for input_vec, cand in self.input_generator.create_input_vec_for_evaluation(sample):
            input_vec.T
            pred = self.pred.eval(session=self.sess, feed_dict={self.X:input_vec, self.keep_prob:1})
            predictions.append((pred[0][0], cand))

        return predictions

    def get_best_pred(self, context, phrase):
        predictions = self.derive_prediction_scores(context, phrase)
        if len(predictions) > 0:
            sorted(predictions, key=itemgetter(0), reverse=True)
            return predictions[0][0], self.graphid2url[predictions[0][1]]

        return -1, ""

    def get_random_pred(self, context, phrase):
        chunk_words = word_tokenize(context)
        span = phrase.text

        try:
            left = chunk_words.index(span)
            right = left + len(word_tokenize(span))
        except ValueError:
            left = len(word_tokenize(context[:phrase.beg]))
            right = len(word_tokenize(context[:phrase.end]))

        candidates, scores = self.fetchFilteredCoreferencedCandEntities.process(left, right, chunk_words)

        if candidates is not None:
            upper_bound = len(candidates)-1
            index = random.randint(0, upper_bound)
            return scores[index], self.graphid2url[candidates[index]]

        return - 1, ""


if __name__ == "__main__":

    default_context = "In the first study, intended to measure a person’s short-term emotional reaction to gossiping, " \
              "140 men and women, primarily undergraduates, were asked to talk about a fictional person either " \
              "positively or negatively."

    phrase = Phrase("undergraduates", 124, 138, "")

    evaluator = Evaluator()


    #print(evaluator.derive_prediction_scores(context=default_context, phrase=phrase))
    print(evaluator.get_best_pred(context=default_context, phrase=phrase))

