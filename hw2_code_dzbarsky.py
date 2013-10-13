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
import itertools
import subprocess

'''
homework 2 by David Zbarsky and Yaou Wang
'''

#stems the sentences and replaces numbers with token <num>
def sent_transform(sent_string):
    stemmer = PorterStemmer()
    tokens = word_tokenize(sent_string)
    tokens = [stemmer.stem(token.lower()) for token in tokens]
    tokens = ['num' if string.translate(token, None, ",.-").isdigit() else token for token in tokens]
    return tokens

#makes tuples of n-grams given the samples
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

#uses filelist to group and divide the appropriate files
#in corpusroot
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

#takes two sets of training files and generates two language models
#to predict the grouping of files in testfiledict
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

def print_sentences_from_files(file_names, outfilename):
    sentences = load_collection_sentences(file_names, 'data')
    with open(outfilename, 'w') as outfile:
        for sentence in sentences:
            outfile.write(sentence)


def gen_lm_from_file(input, output):
    os.system('srilm/ngram-count -text ' + input + ' -lm ' + output)

def srilm_predict(lmfilehigh, lmfilelow, testfileshigh, testfileslow):
    results_high = set()
    bench_high = testfileshigh 
    results_low  = set()
    bench_low = testfileslow

    for testfile in itertools.chain(testfileshigh, testfileslow):
	p_high = subprocess.check_output(["srilm/ngram", "-lm", lmfilehigh, "-ppl", 'test_data/' + testfile])
	p_low = subprocess.check_output(["srilm/ngram", "-lm", lmfilelow, "-ppl", 'test_data/' + testfile])

        p_high = p_high[p_high.find('ppl') + 5:]
        p_high = float(p_high[:p_high.find(' ')])
        p_low = p_low[p_low.find('ppl') + 5:]
        p_low = float(p_low[:p_low.find(' ')])
        if p_low < p_high:
            results_low.add(testfile)
        else:
            results_high.add(testfile)
    
    pres = len(results_high.intersection(bench_high))/float(len(results_high))
    recall = len(results_high.intersection(bench_high))/float(len(bench_high))
    accu = (len(results_high.intersection(bench_high))+len(results_low.intersection(bench_low)))/float(len(results_high)+len(results_low))
    return (pres, recall, accu)

def srilm_predict_merged(lm_high, lm_low, testfilehigh, testfilelow):
    accuracy = 0.0
    p_high1 = subprocess.check_output(["srilm/ngram", "-lm", lm_high, "-ppl", testfilehigh])
    p_low1 = subprocess.check_output(["srilm/ngram", "-lm", lm_low, "-ppl", testfilehigh])

    p_high1 = p_high1[p_high1.find('ppl') + 5:]
    p_high1 = float(p_high1[:p_high1.find(' ')])
    p_low1 = p_low1[p_low1.find('ppl') + 5:]
    p_low1 = float(p_low1[:p_low1.find(' ')])

    p_high2 = subprocess.check_output(["srilm/ngram", "-lm", lm_high, "-ppl", testfilelow])
    p_low2 = subprocess.check_output(["srilm/ngram", "-lm", lm_low, "-ppl", testfilelow])

    p_high2 = p_high2[p_high2.find('ppl') + 5:]
    p_high2 = float(p_high2[:p_high2.find(' ')])
    p_low2 = p_low2[p_low2.find('ppl') + 5:]
    p_low2 = float(p_low2[:p_low2.find(' ')])
    
    if p_high1 < p_low1:
        accuracy += 0.5
    if p_low2 < p_high2:
        accuracy += 0.5

    return accuracy

'''
2.2.4

  We found that our language model is not very accurate in terms of predicting high vs. low returns.
  The precision, recall, accuracy values work out to be (0.54, 0.54, 0.54). The SRILM language model
  is only a little better, at (0.5636363636363636, 0.62, 0.57). Both these results are only a little
  bit better than random chance. This shows that language models are not good predictors for this task.
  
  (Aside: Curiously, if we eliminate the <UNK> substitution (deleting the following in the logprob function:
         if event not in self.events:
            event = '<UNK>')
  the precision, recall, accuracy increases to (0.589, 0.66, 0.60). This may suggest that rather than
  guessing occurences of <UNK> words it would be better to just assign unknown words the probability
  of chance. However, it may also just be a purely random occurence due to this test data set that we use)

  The merged text accuracy is better for our own language model, which gives an accuracy of 1.0. The
  accuracy of the SRILM model is 0.5. This suggests that our own language model got an increase in performance
  possibly because of more context queues. However, since the accuracy measure is very binary (only 2 testfiles)
  we cannot affirmatively conclude any drastic increase in performance. In terms of evaluation, the individual
  files evaluation is much more useful as it gives more testing data and hence comes up with a meaningful
  statistic.

  By a simple comparison, the SRILM model is better in terms of perplexity as we see that in the individual
  events testing this model generates perplexities smaller than our language model. Further, the perplexity values
  for SRILM in the merged test are smaller than our language model. The perplexity improvement does not translate
  very visibly into improvement in the main task. Again, both models are not good at predicting stock performances
  and the SRILM model, despite having a little higher performance, is not much better than our language model.

'''

def get_top_unigrams(lm_file, t):
    words = []
    with open(lm_file) as lm:
        for i in range(3):
            line = lm.readline()
        count = int(line[line.find('=') + 1:])
        for i in range(4):
            lm.readline()
        for i in range(count):
            line = lm.readline()
            prob = line[0:line.find('\t')]
            line = line[line.find('\t') + 1:]
            word = line[0:line.find('\t')]
            words.append((prob, word))
        lm.close()
    words.sort(key=lambda x: x[0])
    return [x[1] for x in words[:t]]

'''
2.2.5

  Here are the top unigrams for the language model trained on high return files:
  ['the', 'of', 'to', 'and', 'for', 'a', 'in', '</s>', 'million', 'Inc.', 'per', 'on',
  'or', 'quarter', 'that', 'share', 'will', 'from', 'net', 'income', 'announced', 'as', 
  'diluted', 'with', 'company', 'Earnings', 'reported', 'was', 'year', 'earnings', 'Corp.',
  'is', 'its', 'compared', 'has', 'million,', 'first', 'by', 'an', 'at', 'ended', 'results',
  'Quarter', 'Communications', 'Results', 'be', 'Corporation', 'period', '2008', 'revenue']

  Here are the top unigrams for the language model trained on low return files:
  ['the', 'of', 'to', 'and', 'for', 'in', 'a', '</s>', 'million', 'Inc.', 'on', 'per', 'quarter',
  'or', 'share', 'income', 'net', 'that', 'from', 'will', 'company', 'diluted', 'announced',
  'reported', 'its', 'with', 'Earnings', 'earnings', 'as', 'was', 'million,', 'Corp.', 'year', 'is',
  'has', 'compared', 'an', 'period', 'by', 'Quarter', 'be', 'ended', 'results', 'Communications',
  'Results', 'same', 'at', 'Corporation', 'first', '30,']

  The two are almost exactly the same. Only 2 words are different. This would explain why
  our models are bad at predicting returns since both high and low return files use very similar
  words.

'''

def get_lm_ranking(lm_file_list, test_text_file):
    lms = []
    for lm in lm_file_list:
	p_high = subprocess.check_output(["srilm/ngram", "-lm", lm, "-ppl", test_text_file])
        p_high = p_high[p_high.find('ppl') + 5:]
        p_high = float(p_high[:p_high.find(' ')])
        lms.append((p_high, lm))
  
    lms.sort(key=lambda x:x[0])
    return [x[1] for x in lms]

'''
2.3.2

  Here is the sorted list of language models from best to worst (smallest to highest perplexivity):
  ['lm_interpolated', 'lm_discount_3', 'lm_default_3', 'lm_discount_2', 'lm_default_2', 'lm_default_1',
  'lm_discount_1', 'lm_laplace_1', 'lm_laplace_2', 'lm_laplace_3']
  This shows that the 3-gram models are the best. Further, the Ney’s absolute discounting with interpolation
  smoothing method is the best and the Laplace smoothing method is the worst. 

'''

def main():
    #print sent_transform('The puppy circled it 34,123.397 times.')
    #print make_ngram_tuples(sent_transform('She eats happily'), 2)
    #trainfiles = get_all_files('data')
    #model = NGramModel(trainfiles, 2)
    #print model.logprob(('.',), '</s>')
    #print gen_rand_text(model, 4, 200)
    #lowd, highd = get_files_listed('data', 'xret_tails.txt')
    #trainfileshigh = highd.keys()
    #trainfileslow = lowd.keys()
    ld, hd = get_files_listed('test_data', 'xret_tails.txt')
    testfileslow = set(ld.keys())
    testfileshigh = set(hd.keys())
    #merge_files(hd.keys(), ld.keys(), 'merged_high.txt', 'merged_low.txt')
    #ld.update(hd)
    #testfiledict = ld
    #print lm_predict(trainfileshigh, trainfileslow, testfiledict)
    #print_sentences_from_files(trainfileshigh, 'all_highd.txt')
    #print_sentences_from_files(trainfileslow, 'all_lowd.txt')
    #for file in get_all_files('test_data'):
    #    print file
    #    print_sentences_from_files(['test_data/' + file], 'srilm/' + file)
    #gen_lm_from_file('all_highd.txt', 'highd_lm')
    #gen_lm_from_file('all_lowd.txt', 'lowd_lm')
    #print srilm_predict('highd_lm', 'lowd_lm', testfileshigh, testfileslow)
    #print srilm_predict_merged('highd_lm', 'lowd_lm', './merged_high.txt', './merged_low.txt')
    #print get_top_unigrams('highd_lm', 50)
    #print get_top_unigrams('lowd_lm', 50)
    #print srilm_predict_merged('highd_lm', 'lowd_lm', './merged_high.txt', './merged_low.txt')
    #l1 = get_top_unigrams('highd_lm', 50)
    #l2 = get_top_unigrams('lowd_lm', 50)
    #print l1
    #print l2
    #print len(set(l1).intersection(set(l2)))    
    #for order in range(3):
    #    num = str(order + 1)
    #    print num
    #    os.system('srilm/ngram-count -text all_highd.txt -order ' + num + ' -lm lm_default_' + num)
    #    os.system('srilm/ngram-count -text all_highd.txt -addsmooth 1 -order ' + num + ' -lm lm_laplace_' + num)
    #    os.system('srilm/ngram-count -text all_highd.txt -cdiscount 0.75 -order ' + num + ' -lm lm_discount_' + num)

    #os.system('srilm/ngram-count -text all_highd.txt -cdiscount 0.75 -interpolate -order 3 -lm lm_interpolated')

    lms = ['lm_default_1', 'lm_default_2', 'lm_default_3', 'lm_discount_1', 'lm_discount_2', 'lm_discount_3', 'lm_interpolated', 'lm_laplace_1', 'lm_laplace_2', 'lm_laplace_3']
    print get_lm_ranking(lms, 'all_highd.txt')

if __name__ == "__main__":
    main()
