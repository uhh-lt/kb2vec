import numpy as np
from utils import overlap
from linkers.context_aware import ContextAwareLinker
from collections import defaultdict
from candidate import Candidate
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from candidate import Phrase, make_phrases
import re
from pandas import read_csv
from time import time
from os.path import join
from utils import ensure_dir
from sklearn.externals import joblib
import json
from os.path import exists
import codecs
from numpy import dot, argmax
from collections import namedtuple
from traceback import format_exc
import requests


Tag = namedtuple("Tag", "id score text offsets uris")


class SuperTagger(ContextAwareLinker):
    def __init__(self):
        ContextAwareLinker.__init__(self)
        self._endpoint_supertagger = "https://supertagger.diffbot.com/el?token=sam&includeKG&confidence=0.5&maxTags=10&lang=en&text={}title="

    def _entity_link(self, text, verbose=True):
        nothing = {}

        uri = self._endpoint_supertagger.format(text)
        r = requests.get(uri)
        content = json.loads(r.content)

        if "all-tags" not in content:
            if verbose: print("Warning: no 'all-tag' found.")
            return nothing

        tags = content["all-tags"]
        result = []
        for i, tag in enumerate(tags):
            try:
                if "kgEntity" not in tag:
                    print("Warning: no 'kgEntity' found.")
                    return nothing
                kg = tag["kgEntity"]

                if "allUris" not in kg:
                    print("Warning: no 'allUris' found.")
                    return nothing

                id = tag["diffbotEntityId"]
                uris = kg["allUris"]
                tag_text = tag["label"]
                offsets = tag["offsets"]["text"]
                score = tag["overallRelevanceScore"]

                result.append(Tag(id, score, tag_text, offsets, uris))
            except:
                print(format_exc())

        return result

    def link(self, context, phrases):
        linked_phrases = []

        for phrase in phrases:
            if True:
                linked_phrases.append((phrase, best_candidate))
            else:
                linked_phrases.append((phrase, Candidate()))

        return linked_phrases

