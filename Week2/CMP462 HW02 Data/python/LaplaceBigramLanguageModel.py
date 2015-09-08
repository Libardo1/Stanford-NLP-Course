from collections import defaultdict
from math import log

class LaplaceBigramLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.BigramCounts = defaultdict(int)
    self.UnigramCounts = defaultdict(int)
    self.total = 0
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """  
    for sentence in corpus.corpus:
        prev_token = '<s>'
        for datum in sentence.data:
            token = datum.word
            bigram = (prev_token, token)
            self.UnigramCounts[token] += 1
            self.BigramCounts[bigram] += 1
            self.total += 1
            prev_token = token

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    prev_token = '<s>'
    V = len(self.BigramCounts)
    score = 0.0
    for token in sentence:
        bigram = (prev_token, token)
        score += log(self.BigramCounts[bigram] + 1)
        score -= log(self.UnigramCounts[prev_token] + V)
        prev_token = token
    return score
