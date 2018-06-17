from linkers.context_aware import ContextAwareLinker
from candidate import Candidate
import json
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
        # link
        tags = self._entity_link(context)

        # assign tags to phrases
        linked_phrases = []
        for phrase in phrases:

            # try to assign the phrase from the tagged output
            assigned_phrase = False
            for tag in tags:
                for tag_beg, tag_end in tag.offsets:
                    if phrase.beg >= tag_end:
                        intersect = phrase.beg - tag_beg < tag_end - tag_beg
                    else:
                        intersect = tag_beg - phrase.beg < phrase.end - phrase.beg

                    if intersect:
                        wiki_uri = self._find_wiki_uri(tag.uris)
                        c = Candidate(tag.score,
                                      tag.text,
                                      wiki_uri,
                                      wiki_uri,
                                      [],[],
                                      tag.uris,
                                      tag.text,
                                      tag.id)
                        linked_phrases.append((phrase, c))
                        assigned_phrase = True

            # if nothing found assign to the phrase something still
            if not assigned_phrase:
                linked_phrases.append((phrase, Candidate()))

        return linked_phrases