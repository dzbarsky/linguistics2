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
    tokens = ['num' if string.translate(token, None, ",.-").isdigit() else token for token in tokens]
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

    #initializes 2 dicts: one with just the context (literals before the word)
    #and one with the ngrams and counts their frequencies
    #if it's the first time we see a word, we replace it with the '<UNK>' symbol
    #also creates a set of all seen words
    def __init__(self, trainfiles, n):
        sentences = load_collection_sentences(trainfiles, 'data')
        self.events.add('<UNK>')
        for sentence in sentences:
            tokens = sent_transform(sentence)
            #goes and replaces first occurence of word with <UNK>
            #by doing it at this stage, we will not count the first
            #occurences of <s> and </s> as <UNK>: these are just special symbols
            #and in our opinion aren't words
            for i in range(len(tokens)):
                if tokens[i] not in self.events:
                    self.events.add(tokens[i])
                    tokens[i] = '<UNK>'
            l = make_ngram_tuples(tokens, n)
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
        if event not in self.events:
            event = '<UNK>'
        if context not in self.context_freq:
            context = ('<UNK>',)
        ngram = (context, event)
        num = self.ngram_freq[ngram] if ngram in self.ngram_freq.keys() else 0
        denom = self.context_freq[context] if context in self.context_freq.keys() else 0
        prob = (float(num) + 1)/(float(denom) + len(self.events))
        return math.log(prob)

    #the probability for each word given its context
    #this is for random text generator purposes: it is unsmoothed
    #and prob is not logarithmic
    def prob_randtext(self, context, event):
        if event not in self.events:
            event = '<UNK>'
        if context not in self.context_freq:
            context = ('<UNK>',)
        ngram = (context, event)
        num = self.ngram_freq[ngram] if ngram in self.ngram_freq else 0
        denom = self.context_freq[context] if context in self.context_freq else len(self.events)
        return float(num)/(float(denom))

    #gives the set of words in the training corpus
    def get_events(self):
        return self.events

def gen_rand_text_helper(bigrammodel, events, prob, context):
    interval = 0.0
    for event in events:
        #for simplicity we are using an unsmoothed probability model to generate words
        interval += bigrammodel.prob_randtext(context, event)
        #if random probability falls in the interval of word
        if interval >= prob:
            return event

def gen_rand_text(bigrammodel, n, wordlimit):
    events = bigrammodel.get_events()
    string = '<s>'
    context = ('formerli',)
    sen_num = 0
    #generates at max the word limit
    i = 0
    while i < wordlimit:
        prob = random.uniform(0.0,1.0)
        word = gen_rand_text_helper(bigrammodel, events, prob, context)
        #if word is <UNK> we regenerate
        #we do not want the token <UNK> to be in our generated string
        if word == '<UNK>':
            i -= 1
        else:
            string = string + ' ' + word
            context = (word,)
        #if the only token given the context is <UNK>
        #we regenerate a new token given the context <UNK>
        if bigrammodel.prob_randtext(context, '<UNK>') == 1:
            context = ('<UNK>',)
        #if it's the end of a sentence
        if word == '</s>':
            sen_num += 1
            #stops when reaches desired number of sentences
            if sen_num == n:
                return string
            else:
                string = string + ' <s>'
                context = ('<s>',)
                i = i + 1
        i += 1

    return string

'''
Here are the 4 sentences randomly generated:
<s> equiti stake in the compani initi valu avail from continu improv coffe lover the compani report second quarter end march num million , num , or $ num million with the silkworm num , inc. report with it market-lead depend on gaap net incom befor . </s> <s> all produc and chief execut director . </s> <s> updat to articl of rang of manag for evalu a quarterli cash flow . </s> <s> the brocad commun system inc. achiev book valu per dilut share on the present call . </s>

'''

def main():
    print sent_transform('The puppy circled it 34,123.397 times.')
    print make_ngram_tuples(sent_transform('She eats happily'), 2)
    trainfiles = get_all_files('data')
    model = NGramModel(trainfiles, 2)
    print model.logprob(('.',), '</s>')
    print gen_rand_text(model, 4, 200)


if __name__ == "__main__":
    main()
