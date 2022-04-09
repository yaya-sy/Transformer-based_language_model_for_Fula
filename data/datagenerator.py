"""This module contains a class that enable to process \
   the data. This class can be used to generate batches, \
   to encode sentences, and to generate prompts.
"""
from random import randrange
from random import shuffle
from typing import List, Iterator

PAD_ID = 3

class DataGenerator :
    """
    This class implements a various utilities for \
    processing the data : batch generator, string encoding \
    prompting, etc.

    Atributes
    ---------
    - bpe_model : 
        Subword sentencepiece model
    - padding_idx : int
        The index of the padding
    - vocab_size : int
        The size of the sentencepiece subwords vocabulary
    - self.encoded_examples : list
        This list of all sentences encoded into vectors
    - size : int
        The number of sentence (or sequences if GPT-type sequences are used)
    """

    def __init__(self, dataset_filename: str, bpe_model) :
        self.bpe_model = bpe_model
        self.padding_idx = bpe_model.pad_id()
        self.vocab_size = bpe_model.vocab_size()

        self.encoded_examples = self.generate_examples(open(dataset_filename))
        shuffle(self.encoded_examples)
        self.encoded_examples.sort(key=lambda utterance: len(utterance[0])) # sort by length for GPU efficient use
        self.size = len(self.encoded_examples)
    
    def examples(self, utterance: str) :
        """
        """
        return self.encode(utterance, add_bos=True, add_eos=False), self.encode(utterance, add_eos=True, add_bos=False)
    
    def generate_examples(self, utterances) :
        """
        """
        return [self.examples(utterance.rstrip()) for utterance in utterances if len(utterance.split(" ")) > 5]

    def encode(self, utterance: str, add_eos=True, add_bos=True) -> List[int] :
        """
        """
        return self.bpe_model.encode(utterance, add_eos=add_eos, add_bos=add_bos)

    def decode(self, utterance: List[int]) -> str :
        """
        """
        return self.bpe_model.decode(utterance)
    
    def id_to_piece(self, utterance: List[int]) -> str :
        return self.bpe_model.id_to_piece(utterance)

    def pad(self, utterances: List[list], max_utterance_size) :
        """
        """
        for utterance in utterances :
            utterance.extend([self.padding_idx] * (max_utterance_size - len(utterance)))

    def __getitem__(self, idx) :
        """
        """
        return [self.encoded_examples[idx][0]], [self.encoded_examples[idx][1]]
    
    def prompt(self) :
        founded_prompt = []
        while len(founded_prompt) < 2 or len(founded_prompt) > 20 :
            random_prompt = randrange(self.size)
            prompt_sent, _ = self[random_prompt]
            prompt_sent = prompt_sent[0]
            prompt_size = randrange(len(prompt_sent) // 2)
            founded_prompt = prompt_sent[:prompt_size]
        return founded_prompt

    
    def __call__(self, batch_size, shuffling=False) :
        """
        """
        return self.batchify(batch_size, shuffling=shuffling)

    def batchify(self, batch_size, shuffling=False) -> Iterator[List[List[int]]] :
        """
        """
        if shuffling :
            shuffle(self.encoded_examples)
        for step in range(0, self.size, batch_size) :
            X, Y = zip(*self.encoded_examples[step:step + batch_size])
            lengths = [len(seq) for seq in X]
            max_utterance_size = max(lengths)
            self.pad(X, max_utterance_size); self.pad(Y, max_utterance_size)
            yield list(X), list(Y), lengths