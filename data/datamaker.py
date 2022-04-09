import os
from random import shuffle
import json
from nltk import sent_tokenize, word_tokenize

class MakeData : 
    """
    """

    def __init__(self, out_dirname, merge_len=None, sequences_type="sents") : 
        self.out_dirname = out_dirname
        self.merge_len = merge_len
        if not os.path.exists(out_dirname):
            os.makedirs(out_dirname)
        
        self.sequences_type = {
                                "sents" : self.sentences,
                                "gpt" : self.gpt_sequences
                            }[sequences_type]
    def sentences(self, loaded_json) :
        for data in loaded_json :
            for sent in sent_tokenize(data["text"]) :
                yield sent

    def gpt_sequences(self, loaded_json) :
        for data in loaded_json :
            sentences = [" ".join(word_tokenize(sent)) for sent in sent_tokenize(data["text"])]
            len_tokens_seq = sum(1 for sentence in sentences for word in sentence.split(" "))
            if len_tokens_seq < self.merge_len : continue # skip data (document) which length is inferior to merge_len
            gpt_sequence = ""
            for sent in sentences :
                gpt_sequence = " ".join([gpt_sequence, sent])
                if self.merge_len <= len(gpt_sequence.split(" ")) < self.merge_len + 20 : # 20 for max difference
                    yield gpt_sequence
                    gpt_sequence = ""
    
    def make_files(self, data, out_filename) :
        out_file = open(f"{self.out_dirname}/{out_filename}", "w")
        for sequence in data :
            out_file.write(sequence + "\n")

    def __call__(self, filename, out_filename, train_pct=0.90, test_pct=0.05, dev_pct=0.05) :
        loaded_json = json.load(open(filename))
        sequences = set()
        for sequence in self.sequences_type(loaded_json) :
            if sequence in sequences : continue
            sequences.add(sequence)
        sequences = list(sequences)
        nb_sequences = len(sequences)
        shuffle(sequences)
        train_part = int(nb_sequences * train_pct)
        test_part = int(nb_sequences * test_pct)
        train = sequences[:train_part]
        test = sequences[train_part:train_part + test_part]
        dev = sequences[train_part + test_part:]
        self.make_files(train, out_filename=out_filename + ".train")
        self.make_files(test, out_filename=out_filename + ".test")
        self.make_files(dev, out_filename=out_filename + ".dev")