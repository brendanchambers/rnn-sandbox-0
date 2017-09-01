# based on
# http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/

import csv
import itertools
import nltk
import numpy
import time
from utils import *
from datetime import datetime
import sys
from rnn_theano import RNNTheano, gradient_check_theano


#raw_text_path = 'full_trump_speech.txt'
raw_text_path = 'data/full_trump_speech.txt'
CORPUS_ENCODING = 'utf-16'

RETRAIN = True
HIDDEN_DIM = 2 #  80
LEARNING_RATE = 0.000001 # 0.005
N_EPOCH = 3 # 100
BATCH_SIZE = 1
MODEL_FILE = 'test'


vocabulary_size = 5
unknown_token = "Unknown_Token"
sentence_start_token = "Sentence_Start"
sentence_end_token = "Sentence_End"

########################################################################### helper function -
def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=1, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        print "epoch #" + str(epoch)
        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5
                print "Setting learning rate to %f" % learning_rate
            sys.stdout.flush()
            # ADDED! Saving model oarameters
            save_model_parameters_theano("data/rnn-theano-%d-%d-%s.npz" % (model.hidden_dim, model.word_dim, time), model)
        # For each training example...
        #for i in range(len(y_train)):
        sample_idxs = np.random.choice(len(y_train),BATCH_SIZE) # mini batches
        for i in sample_idxs:
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1


########################################################################### get corpus

# for opening .txt
print "reading datafile:  " + raw_text_path
with open(raw_text_path,'rb') as f:
    print "nltk version: " + str(nltk.__version__)

    file_contents = f.read()
    #file_contents = file_contents.lower() # might not want to do this yet right? because sent_tokenize is smart
    # todo ignore "Trump:" ignore "[applause]"
    raw_sentences = file_contents.rstrip("\n").decode(CORPUS_ENCODING)
print "tokenizing sentences..."
sentences = nltk.sent_tokenize(raw_sentences)
print np.shape(raw_sentences)
print np.shape(sentences)
# todo remove annotations (e.g. "[applause], Trump: ")

'''
# for opening .csv
with open(csv_path, 'rb') as f:
    reader = csv.reader(f, skipinitialspace=True)
    reader.next()
    print reader.next()

    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
'''
# more pre-processing
sentences = ["%s %s %s" % (sentence_start_token, x.lower(), sentence_end_token) for x in sentences] # insert markers for start and stop, for discrete sequence learning
# sanity
print "Parsed %d sentences." % (len(sentences))
#print sentences

print "tokenizing words"
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences] # lower case
print "sentence 1: "
print tokenized_sentences[0]

word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print "found %d unique words tokens" % len(word_freq.items())
print "word frequency list: "
print word_freq # todo look at histogram

# Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size-1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token) # add the special case
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
print index_to_word
print word_to_index

print "Using vocabulary size %d." % vocabulary_size
print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])

# Replace all words not in our vocabulary with the unknown token
# todo needs cleaner text preprocessing
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

# Create the training data
X_train = numpy.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = numpy.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])


######################################################################## construct RNN
print "constructing model..."  # todo try a smarter initialization - wrt vanishing gradients
model = RNNTheano(vocabulary_size, hidden_dim=HIDDEN_DIM)

gradient_check_theano(model, X_train[10], y_train[10], h=0.0000001, error_threshold=0.01)

######################################################################## train

if RETRAIN:
    # run a single step to get a feel for training time
    print "run a single step..."
    t1 = time.time()
    model.sgd_step(X_train[10], y_train[10], LEARNING_RATE)
    t2 = time.time()
    print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)

    #print " loading model parameters..."
    #if MODEL_FILE != None:
    #    load_model_parameters_theano(MODEL_FILE, model)

    print "training..."
    train_with_sgd(model, X_train, y_train, nepoch=N_EPOCH, learning_rate=LEARNING_RATE)

####################################################################### if no train, load prior model
else:
    # todo load model
    MODEL_FILE = 'data/rnn-theano-100-2000-2017-08-31-11-33-30.npz'
    print "loading model parameters..."
    model = RNNTheano(vocabulary_size, hidden_dim=HIDDEN_DIM) # re-init from scratch
    load_model_parameters_theano(MODEL_FILE, model)

############################################################################## helper function

def generate_sentence_deprecated(model):
    # We start the sentence with the start token
    new_sentence = [word_to_index[sentence_start_token]]
    # Repeat until we get an end token
    while not new_sentence[-1] == word_to_index[sentence_end_token]:
        next_word_probs = model.forward_propagation(new_sentence)
        sampled_word = word_to_index[unknown_token]
        # We don't want to sample unknown words
        while sampled_word == word_to_index[unknown_token]:
            samples = np.random.multinomial(1, next_word_probs[-1])
            sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
    sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    return sentence_str

def generate_sentence(model):

    new_sentence = [word_to_index[sentence_start_token]] # start with start token

    while not new_sentence[-1] == word_to_index[sentence_end_token]: # sample until we get an end token
        next_word_probs = model.forward_propagation(new_sentence)
        #sampled_word = word_to_index[unknown_token]
        # We don't want to sample unknown words
        #while sampled_word == word_to_index[unknown_token]:
        samples = np.random.multinomial(1, next_word_probs[-1])
        sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
    sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    return sentence_str

############################################################## generate some sentences


num_sentences = 10
senten_min_length = 3

for i in range(num_sentences):
    sent = []
    # We want long sentences, not sentences with one or two words
    while len(sent) < senten_min_length:
        sent = generate_sentence(model)
    print " ".join(sent)


######################################### get the gradient for a sample trial

#print "run a single step..."
#t1 = time.time()
#model.sgd_step(X_train[0], y_train[0], LEARNING_RATE)
#t2 = time.time()
#print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)

