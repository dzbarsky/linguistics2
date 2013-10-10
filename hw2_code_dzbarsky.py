# David Zbarsky: dzbarsky@wharton.upenn.edu
# Yaou Wang: yaouwang@wharton.upenn.edu

from nltk.corpus import PlaintextCorpusReader
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer
import math
import string
import random

'''
homework 2 by David Zbarsky and Yaou Wang
'''

def sent_transform(sent_string):
    stemmer = PorterStemmer()
    tokens = word_tokenize(sent_string)
    tokens = [stemmer.stem(token.lower()) for token in tokens]
    tokens = ['num' if string.translate(token, None, ",.").isdigit() else token for token in tokens]
    return tokens

def make_ngram_tuples(samples, n):
    ngrams = []
    for i in range(len(samples)+1):
        l = []
        for j in range(i-n+1, i):
            if j < 0:
                l.append('<s>')
            else:
                l.append(samples[j])
        if i < len(samples):
            ngrams.append((tuple(l),samples[i]))
        else:
            ngrams.append((tuple(l),'</s>'))
    
    return ngrams

#from hw1 -> get all files from the training directory
def get_all_files(directory):
    if directory.find('.') < 0:
        return PlaintextCorpusReader(directory, '.*').fileids()
    #if directory is a file return the file in a list
    return [directory]

#from hw1 -> loads the sentences of a file
def load_file_sentences(filepath):
    index = filepath.rfind('/')
    dir = filepath[:index]
    filepath = filepath[index + 1:]
    return sent_tokenize(PlaintextCorpusReader(dir, filepath).raw())

#from hw1 -> given list of files, loads all the sentences for all the files
def load_collection_sentences(files, directory):
    sentences = []
    for file in files:
        sentences.extend(load_file_sentences(directory + '/' + file))
    return sentences

class NGramModel:
    ngram_freq = dict()
    context_freq = dict()
    events = set()
    
    def __init__(self, trainfiles, n):
        sentences = load_collection_sentences(trainfiles, 'data')
        for sentence in sentences:
            l = make_ngram_tuples(sent_transform(sentence), n)
            for p in l:
                if p in self.ngram_freq:
                    self.ngram_freq[p] += 1
                else:
                    self.ngram_freq[p] = 1
                if p[0] in self.context_freq:
                    self.context_freq[p[0]] += 1
                else:
                    self.context_freq[p[0]] = 1
                if p[1] not in self.events:
                    self.events.add(p[1])
        

    def logprob(self, context, event):
        ngram = (context, event)
        num = self.ngram_freq[ngram] if ngram in self.ngram_freq else 0
        denom = self.context_freq[context] if context in self.context_freq else 0
        prob = (float(num) + 1)/(float(denom) + len(self.events))
        return math.log(prob)

    def prob_unsmooth(self, context, event):
        ngram = (context, event)
        num = self.ngram_freq[ngram] if ngram in self.ngram_freq else 0
        denom = self.context_freq[context] if context in self.context_freq else len(self.events)
        return float(num)/denom

    def get_events(self):
        return self.events

def gen_rand_text(bigrammodel, n, wordlimit):
    events = bigrammodel.get_events()
    #print events
    string = '<s>'
    context = ('<s>',)
    sen_num = 0
    for i in range(wordlimit):
        prob = random.uniform(0.0,1.0)
        interval = 0.0
        word = ''
        for event in events:
            #for simplicity we are using an unsmoothed probability model to generate words
            interval += bigrammodel.prob_unsmooth(context, event)
            if interval >= prob:
                word += event
                break
        string = string + ' ' + word
        context = (word,)
        if word == '</s>':
            sen_num += 1
            if sen_num == n:
                return string
            else:
                string = string + ' <s>'
                context = ('<s>',)
                i += 1

    return string

def main():
    print sent_transform('The puppy circled it 34,123.397 times.')
    print make_ngram_tuples(sent_transform('She eats happily'), 2)
    trainfiles = get_all_files('data')
    model = NGramModel(trainfiles, 2)
    print model.logprob(('h.264',), 'h.264')
    print gen_rand_text(model, 2, 100)

if __name__ == "__main__":
    main()
