from collections import defaultdict
from math import log

class CustomLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.BigramCounts = defaultdict(int)
    self.UnigramCounts = defaultdict(int)
    self.TrigramCounts = defaultdict(int)
    self.total = 0
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
        Trigram Stupid Backoff Langage Model
    """  
    for sentence in corpus.corpus:
        pre_prev_token = '<s>'
        prev_token = '<s>'
        for datum in sentence.data:

            token = datum.word
            bigram = (prev_token, token)
            trigram = (pre_prev_token, prev_token, token)

            self.UnigramCounts[token] += 1
            self.BigramCounts[bigram] += 1
            self.TrigramCounts[trigram] += 1
            self.total += 1

            pre_prev_token = prev_token
            prev_token = token

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    score = 0.0
    pre_prev_token = '<s>'
    prev_token = "<s>"
    for token in sentence:
        bigram = (prev_token, token)
        trigram = (pre_prev_token, prev_token, token)

        countTrigram = self.TrigramCounts[trigram]
        countBigram = self.BigramCounts[bigram]
        countBigram_prev = self.BigramCounts[(pre_prev_token, prev_token)]
        countUnigram = self.UnigramCounts[token]

        V = len(self.UnigramCounts)
        if countTrigram > 0 and countBigram_prev > 0:
            score += log(countTrigram)
            score -= log(countBigram_prev)            
        if countBigram > 0:
            score += log(countBigram)
            #score += log(0.4)
            score -= log(self.UnigramCounts[prev_token])
        else:
            score += log(countUnigram + 1)
            #score += log(0.4)
            score -= log(self.total + V)
    return score