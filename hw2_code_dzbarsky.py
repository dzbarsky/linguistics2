# David Zbarsky: dzbarsky@wharton.upenn.edu
# Yaou Wang: yaouwang@wharton.upenn.edu

from nltk.corpus import PlaintextCorpusReader
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer
import math
import string
import random
import fileinput
import os

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
        if file.rfind('/') < 0:
            sentences.extend(load_file_sentences(directory + '/' + file))
        else:
            sentences.extend(load_file_sentences(file))
    return sentences

class NGramModel:

    #initializes 2 dicts: one with just the context (literals before the word)
    #and one with the ngrams and counts their frequencies
    #if it's the first time we see a word, we replace it with the '<UNK>' symbol
    #also creates a set of all seen words
    def __init__(self, trainfiles, n):
        self.ngram_freq = dict()
        self.context_freq = dict()
        self.events = set()
        self.n = n
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

    #gives the log probability of event|context with add1 smoothing
    def logprob(self, context, event):
        if event not in self.events:
            event = '<UNK>'
        ngram = (context, event)
        num = self.ngram_freq[ngram] if ngram in self.ngram_freq.keys() else 0
        denom = self.context_freq[context] if context in self.context_freq.keys() else 0
        prob = (float(num) + 1)/(float(denom) + len(self.events))
        return math.log(prob, 2)

    #the probability for each word given its context
    #this is for random text generator purposes: it is unsmoothed
    #and prob is not logarithmic
    def prob_randtext(self, context, event):
        if event not in self.events:
            event = '<UNK>'
        #this change only deals with unknown context for bigrams
        if context not in self.context_freq.keys():
            context = ('<UNK>',)
        ngram = (context, event)
        num = self.ngram_freq[ngram] if ngram in self.ngram_freq else 0
        denom = self.context_freq[context] if context in self.context_freq else len(self.events)
        return float(num)/(float(denom))

    #gives the set of words in the training corpus
    def get_events(self):
        return self.events

    #returns the log perplexity of given file
    #this implementation assumes each sentence is independent
    #and hence their log probabilities are added together
    def getppl(self, testfile):
        ppl = 0.0
        t = 0.0
        sentences = load_file_sentences(testfile)
        for sentence in sentences:
            tokens = sent_transform(sentence)
            l = make_ngram_tuples(tokens, self.n)
            for p in l:
                t += 1
                ppl += self.logprob(p[0], p[1])
        return math.pow(2, -ppl/t)

#helper for gen_rand_text function
def gen_rand_text_helper(bigrammodel, events, prob, context):
    interval = 0.0
    for event in events:
        #for simplicity we are using an unsmoothed probability model to generate words
        interval += bigrammodel.prob_randtext(context, event)
        #if random probability falls in the interval of word
        if interval >= prob:
            return event

#generates random text given bigram model
def gen_rand_text(bigrammodel, n, wordlimit):
    events = bigrammodel.get_events()
    string = '<s>'
    context = ('<s>',)
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
<s> equiti stake in the compani initi valu avail from continu improv coffe lover the compani report second quarter end march num million , num , or $ num million with the silkworm num , inc. report with it market-lead depend on gaap net incom befor . </s>
<s> all produc and chief execut director . </s>
<s> updat to articl of rang of manag for evalu a quarterli cash flow . </s>
<s> the brocad commun system inc. achiev book valu per dilut share on the present call . </s>

'''

def get_files_listed(corpusroot, filelist):
    lowd = dict()
    highd = dict()
    files = get_all_files(corpusroot)
    index = filelist.rfind('/')
    if index < 0:
        tokens = word_tokenize(PlaintextCorpusReader('.', filelist).raw())
    else:
        tokens = word_tokenize(PlaintextCorpusReader(filelist[:index], filelist[index+1:]).raw())
    i = 0
    while i < len(tokens):
        if float(tokens[i+1]) <= 5.0 and tokens[i] in files:
            lowd[tokens[i]] = float(tokens[i+1])
        if float(tokens[i+1]) >= 5.0 and tokens[i] in files:
            highd[tokens[i]] = float(tokens[i+1])
        i += 2

    return (lowd, highd)

def lm_predict(trainfileshigh, trainfileslow, testfiledict):
    results_high = set()
    bench_high = set()
    results_low  = set()
    bench_low = set()
    lm_high = NGramModel(trainfileshigh, 2)
    lm_low = NGramModel(trainfileslow, 2)

    for testfile in testfiledict.keys():
        if testfile.rfind('/') < 0:
            p_high = lm_high.getppl('test_data/' + testfile)
            p_low = lm_low.getppl('test_data/' + testfile)
        else:
            p_high = lm_high.getppl(testfile)
            p_low = lm_low.getppl(testfile)
        if p_low < p_high:
            results_low.add(testfile)
        else:
            results_high.add(testfile)
        if testfiledict[testfile] > 0.0:
            bench_high.add(testfile)
        else:
            bench_low.add(testfile)
    #we evaluate the big merged text here since the language models have already been built in this function
    print 'merged texts evaluation accuracy =' + str(lm_predict_merged(lm_high, lm_low, './merged_high.txt', './merged_low.txt'))
    pres = len(results_high.intersection(bench_high))/float(len(results_high))
    recall = len(results_high.intersection(bench_high))/float(len(bench_high))
    accu = (len(results_high.intersection(bench_high))+len(results_low.intersection(bench_low)))/float(len(results_high)+len(results_low))
    return (pres, recall, accu)

#helper to merge the high/low files into one file each
def merge_files(fileshigh, fileslow, testfilehigh, testfilelow):
    with open(testfilehigh, 'w') as outfile:
        for testfile in fileshigh:
            with open('test_data/' + testfile) as infile:
                for line in infile:
                    outfile.write(line)
    with open(testfilelow, 'w') as outfile:
        for testfile in fileslow:
            with open('test_data/' + testfile) as infile:
                for line in infile:
                    outfile.write(line)

#function for evaluating the two merged files
def lm_predict_merged(lm_high, lm_low, testfilehigh, testfilelow):
    accuracy = 0.0
    p_high1 = lm_high.getppl(testfilehigh)
    p_low1 = lm_low.getppl(testfilehigh)
    p_high2 = lm_high.getppl(testfilelow)
    p_low2 = lm_low.getppl(testfilelow)
    if p_high1 < p_low1:
        accuracy += 0.5
    if p_low2 < p_high2:
        accuracy += 0.5
    return accuracy

'''
On evaluation of our language model:
  We found that our language model is not very accurate in terms of predicting high vs. low returns.
  The precision, recall, accuracy values work out to be (0.54, 0.54, 0.54), which are only a little
  bit better than random chance. Fundamentally, this is because of data sparsity. We have many cases
  where the context or event has never been seen before by the models. Curiously, if we eliminate the
  <UNK> substitution (deleting the following in the logprob function:
         if event not in self.events:
            event = '<UNK>')
  the precision, recall, accuracy increases to (0.589, 0.66, 0.60). This suggests that rather than
  guessing occurences of <UNK> words it would be better to just assign unknown words the probability
  of chance. However, the fact that changing the implementation of seeing the unknown word changes
  the probability is also a reflection of how sparse our data is.

  On testing the merged texts, we are able to arrive at an accuracy of 1.0. This shows that if the test
  file is long enough, then there are enough clues for our language models to pick up to accurately predict
  the file's associated returns.

'''

def print_sentences_from_files(file_names, outfilename):
    sentences = load_collection_sentences(file_names, 'data')
    with open(outfilename, 'w') as outfile:
        for sentence in sentences:
            outfile.write(sentence)


def gen_lm_from_file(input, output):
    os.system('srilm/ngram-count -text ' + input + ' -lm ' + output)

#def srilm predict(lmfilehigh, lmfilelow, testfileshigh, testfileslow):


def main():
    #print sent_transform('The puppy circled it 34,123.397 times.')
    #print make_ngram_tuples(sent_transform('She eats happily'), 2)
    #trainfiles = get_all_files('data')
    #model = NGramModel(trainfiles, 2)
    #print model.logprob(('.',), '</s>')
    #print gen_rand_text(model, 4, 200)
    lowd, highd = get_files_listed('data', 'xret_tails.txt')
    trainfileshigh = highd.keys()
    trainfileslow = lowd.keys()
    ld, hd = get_files_listed('test_data', 'xret_tails.txt')
    #merge_files(hd.keys(), ld.keys(), 'merged_high.txt', 'merged_low.txt')
    ld.update(hd)
    testfiledict = ld
    #print lm_predict(trainfileshigh, trainfileslow, testfiledict)
    #print_sentences_from_files(trainfileshigh, 'all_highd.txt')
    #print_sentences_from_files(trainfileslow, 'all_lowd.txt')
    #for file in get_all_files('test_data'):
    #    print file
    #    print_sentences_from_files(['test_data/' + file], 'srilm/' + file)
    gen_lm_from_file('all_highd.txt', 'highd_lm')
    gen_lm_from_file('all_lowd.txt', 'lowd_lm')



if __name__ == "__main__":
    main()
