#!/usr/bin/env python2.7

from collections import Counter
from tqdm import tqdm
import numpy as np

corpus_file = '/Users/calvin-is-seksy/Desktop/myProjects/CS291A/data/text8'
vocab_file = '/Users/calvin-is-seksy/Desktop/myProjects/CS291A/data/vocab.txt'
output_file = './vector.txt'
# LR = 1e-3
epoch = 10
CWS = 2 			# Context Window Size
numNS = 2 			# neg samples 
vector_dimension = 300 

'''
what do: 
LR = 0.01 


'''

def lines(infile):
    with open(infile, 'r') as fp:
        for line in fp:
            yield line

def sigmoid(x): 
    return 1/(1+np.exp(-x))

def run(corpus_file, vocab_file, output_file):

	LR = .01

	# read data from input
	allTargetWords = []
	for line in lines(vocab_file):
		temp1 = line.strip() 
		temp2 = temp1.split()[0]
		word = temp2.lower()
		allTargetWords.append(word)

	myCounter = Counter() 
	for line in lines(corpus_file):
		temp1 = line.strip()
		words = line.split()
		myCounter.update(words)

	vocab = {}
	for index, word in enumerate(allTargetWords):
		vocab[word] = (index, myCounter[word])

	vocabSize = len(vocab)

	myTokens, p = zip(*vocab.values())
	p = np.array(p)**.5
	p = p / p.sum()           
	PSamples = [] 
	TF = sum(myCounter.values()) * 1e-6
	for l in lines(corpus_file):
		words = []
		for word in l.strip().split():
			if len(word) > 1 and word.isalnum():
				words.append(word.lower())
		
		tokens = [] 
		for word in words:
			if np.random.binomial(1, min(1, (TF/myCounter[word])**(.75) )) == 1: 
				temp1 = vocab.get(word, (None, 0))
				tokens.append(temp1[0])

			# subsampling other implementation -- total words in corpus: 17005207
			# sample_rate = 0.001
			# P_w = np.dot(np.sqrt(myCounter[word] / 17005207 / sample_rate) + 1, sample_rate/myCounter[word])
			# if P_w >= 0.0010000009421554255/2: # min: 9.42155425496e-10, max: 0.001
			# 	temp1 = vocab.get(word, (None, 0))
			# 	tokens.append(temp1[0])

		for index, token in enumerate(tokens):
			if token is None: continue 
			leftEnd = max(0, index - CWS)
			rightEnd = min(index + CWS, len(tokens) - 1)
			context = tokens[leftEnd:index] + tokens[index+1:rightEnd+1]
			while None in context:
				context.remove(None)
			if not context: continue 
			PSamples.append((token, context))

	# print PSamples

	# your training algorithm
	embedding_word = np.random.uniform(-.5, .5, size=(vocabSize, vector_dimension))
	embedding_context = np.random.uniform(-.5, .5, size=(vocabSize, vector_dimension))
	# gradient_square_embedding_word = np.ones((vocabSize, vector_dimension))
	# gradient_square_embedding_context = np.ones((vocabSize, vector_dimension))

	for epo in range(epoch):
		negLL = 0.0
		np.random.shuffle(PSamples)
		for center, context in tqdm(PSamples):
			NSamples = np.random.choice(myTokens, numNS, False) # p 
			while any([s in context for s in NSamples]):
				NSamples = np.random.choice(myTokens, numNS, False) # p 
			
			for contextToken in context: 
				gradient_word = np.zeros(vector_dimension)
				targets = [(contextToken, 1)] + [(s, 0) for s in NSamples]

				for target, binary in targets: 
					sig = sigmoid(np.dot(embedding_word[center], embedding_context[target]))

					# OTHER IMP 
					error = sig - binary 
					# negLL  = np.sum(error ** 2)
					# gradient_context = embedding_word[center].T.dot(error) / embedding_word.shape[0]
					# embedding_context[target] += -LR * gradient_context 
					# gradient_word = embedding_context[target].T.dot(error) / embedding_context.shape[0]
					# embedding_word[center] += -LR * gradient_word

					# OG SHIT 
					gradient_word += error * embedding_context[target]
					gradient_context = error * embedding_word[center] 

					embedding_context -= LR * gradient_context 
					negLL -= np.log((1-binary) - (-1)**binary*sig)

				embedding_word[center] -= LR * gradient_word 

		print 'Epoch: %d \t Negative Log Likelihood: %.5f' % (epo+1, negLL / len(PSamples)) 
		embedding = ((embedding_word + embedding_context) / 2) 
		embedding = embedding_word / np.linalg.norm(embedding_word, axis=-1).reshape(-1, 1)
		
		# ME 
		# embedding = embedding_word.reshape(-1, 1)

		# if epo == 5: LR = LR/2

	# your prediction code
	with open('./vectors.txt', 'w') as out: 
		for element in allTargetWords: 
			vec = ['%.6f' % tempWeight for tempWeight in embedding[vocab[element][0]]]
			myLine = element + " " + " ".join(vec) + '\n'
			out.write(myLine)

if __name__ == '__main__':
	run(corpus_file, vocab_file, output_file)

























