from collections import defaultdict
from math import log

class LaplaceUnigramLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.UnigramCounts = defaultdict(int)
    self.total = 0
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """  
    for sentence in corpus.corpus:
        for datum in sentence.data:
            token = datum.world
            self.UnigramCounts[token] += 1
            self.total += 1

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
        Plus one smoothing.
    """
    score = 0.0
    V = len(self.UnigramCounts)
    for token in sentence:
        count = self.UnigramCounts[token]
        score += log(count + 1)
        score -= log(self.total + V)
    return score
