from math import log
import os 
from difflib import SequenceMatcher


# This is the project root directory assuming that utils.py is in the root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def ensure_dir(dir_path):
    if not os.path.exists(dir_path): os.makedirs(dir_path)


def dbpedia2wikipedia(url, to_en=True):
    """ Convert a dbpedia to wikipedia url. """

    url = url.replace("https://", "")
    url = url.replace("http://", "")

    if to_en:
        wiki_domain = "en.wikipedia.org/wiki/"
    else:
        wiki_domain = "wikipedia.org/wiki/"
        
    new_url = url.replace("dbpedia.org/resource/", wiki_domain)
    if new_url == url:
        new_url = url.replace("dbpedia.org/page/", wiki_domain)
        
    return new_url


def longest_common_substring(s1, s2, lower=True):
    if lower:
        s1 = s1.lower()
        s2 = s2.lower()

    match = SequenceMatcher(None, s1, s2).find_longest_match(0, len(s1), 0, len(s2))
    substring = s1[match.a: match.a + match.size]

    return substring


def overlap(s1, s2, lower=True):
    direct = longest_common_substring(s1, s2, lower)
    inverse = longest_common_substring(s2, s1, lower)
    max_overlap = float(max(len(direct), len(inverse)))
    if max_overlap < 3:
        return 0.0
    else:
        max_len = float(max(len(s1), len(s2)))
        return max_overlap / max_len


def truncated_log(x):
    if x > 0: return log(x)
    else: return 0.0    
