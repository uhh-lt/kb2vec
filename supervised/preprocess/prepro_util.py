
class Chunker(object):
    def __init__(self):
        self.separator = "per_document"
        self.chunk_ending = {'DOCEND'}
        if self.separator == "per_paragraph":
            self.chunk_ending.add('*NL*')
        if self.separator == "per_sentence":
            self.chunk_ending.add('.')
            self.chunk_ending.add('*NL*')
        self.parsing_errors = 0

    def new_chunk(self):
        self.chunk_words = []
        self.begin_gm = []          # the starting positions of gold mentions
        self.end_gm = []            # the end positions of gold mentions
        self.ground_truth = []      # list with the correct entity ids

    def compute_result(self, docid):
        chunk_id = docid
        if self.separator == "per_paragraph":
            chunk_id = chunk_id + "&*" + str(self.par_cnt)
        if self.separator == "per_sentence":
            chunk_id = chunk_id + "&*" + str(self.par_cnt) + "&*" + str(self.sent_cnt)
        result = (chunk_id, self.chunk_words, self.begin_gm, self.end_gm, self.ground_truth)

        # correctness checks. not necessary
        no_errors_flag = True
        if len(self.begin_gm) != len(self.end_gm) or \
            len(self.begin_gm) != len(self.ground_truth):
            no_errors_flag = False
        for b, e in zip(self.begin_gm, self.end_gm):
            if e <= b or b >= len(self.chunk_words) or e > len(self.chunk_words):
                no_errors_flag = False

        self.new_chunk()
        if no_errors_flag == False:
            self.parsing_errors += 1
            print("chunker parse error: ", result)
            return None
        else:
            return result

    def process(self, filepath):
        with open(filepath) as fin:
            self.new_chunk()
            docid = ""
            # paragraph and sentence counter are not actually useful. only for debugging purposes.
            self.par_cnt = 0      # paragraph counter (useful if we work per paragraph)
            self.sent_cnt = 0      # sentence counter (useful if we work per sentence)
            for line in fin:
                line = line.rstrip()     # omit the '\n' character
                if line in self.chunk_ending:
                    if len(self.chunk_words) > 0:  # if we have continues *NL* *NL* do not return empty chunks
                        temp = self.compute_result(docid)
                        if temp is not None:
                            yield temp
                    # do not add the chunk separator, no use
                    if line == '*NL*':
                        self.par_cnt += 1
                        self.sent_cnt = 0
                    if line == '.':
                        self.sent_cnt += 1
                elif line == '*NL*':
                    self.par_cnt += 1
                    self.sent_cnt = 0
                    # do not add this in our words list
                elif line == '.':
                    self.sent_cnt += 1
                    self.chunk_words.append(line)
                elif line.startswith('MMSTART_'):
                    ent_id = line[8:]   # assert that ent_id in wiki_name_id_map
                    self.ground_truth.append(ent_id)
                    self.begin_gm.append(len(self.chunk_words))
                elif line == 'MMEND':
                    self.end_gm.append(len(self.chunk_words))
                elif line.startswith('DOCSTART_'):
                    docid = line[9:]
                    self.par_cnt = 0
                    self.sent_cnt = 0
                else:
                    self.chunk_words.append(line)

        print(filepath, " chunker parsing errors: ", self.parsing_errors)
        self.parsing_errors = 0


class InputSamplesGenerator(object):
    def __init__(self):
        self.chunker = Chunker()

    def process(self, path):
        count = 0
        for chunk in self.chunker.process(path):
            cand_entities = []  # list of lists     candidate entities
            cand_entities_scores = []
            chunk_id, chunk_words, begin_gm, end_gm, ground_truth = chunk
            print(chunk_id, chunk_words, begin_gm, end_gm, ground_truth)
            print(chunk_words[begin_gm[0]:end_gm[0]])
            return 0



if __name__ == "__main__":
    generator = InputSamplesGenerator()
    generator.process('/Users/sevgili/PycharmProjects/end2end_neural_el/data/new_datasets/ace2004.txt')
