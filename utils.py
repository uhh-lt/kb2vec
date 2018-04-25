from math import log


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


def longest_common_substring(s1, s2):
    m = [[0] * (1 + len(s2)) for i in range(1 + len(s1))]
    longest, x_longest = 0, 0
    for x in range(1, 1 + len(s1)):
        for y in range(1, 1 + len(s2)):
            if s1[x - 1] == s2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
                else:
                    m[x][y] = 0
    return s1[x_longest - longest: x_longest]


def overlap(s1, s2):
    direct = longest_common_substring(s1, s2)
    inverse = longest_common_substring(s2, s1)
    max_overlap = float(max(len(direct), len(inverse)))
    max_len = float(max(len(s1), len(s2)))

    return max_overlap / max_len


def truncated_log(x):
    if x > 0: return log(x)
    else: return 0.0    
