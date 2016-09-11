from nltk.tokenize import sent_tokenize, word_tokenize
from os import path, listdir
import numpy as np
import random



illegalSymbols ={"@","|","_","-",">","<","=","+","$","#","%","^","*","/","\\","~","`","...","(",")","[","]","{","}","com","gov","org","edu"}


def cleanText(filePath):
	file = open(filePath)
	text = file.read()

	#Remove email headers - we don't want to generate a random string with an email header, 
	#nor are they relevant in the context of a sentence
	# scikit learn might be better for this: http://scikit-learn.org/stable/datasets/#the-20-newsgroups-text-dataset
	startBodyIdentifier = "writes :";
	bodyStart = max(text.rfind("writes :"), text.rfind("wrote :"), text.rfind("Subject :"), text.rfind("Re : "))
	startBody = text.rfind(startBodyIdentifier) + len(startBodyIdentifier)

	newText = text[startBody:]
	return newText

def isLegal(token, illegalSymbols):
	for symbol in illegalSymbols:
		if symbol in token:
			return False
	return True

def getTokens(corpus):
	tokens = word_tokenize(corpus)
	legalTokens = []
	for token in tokens:
		if isLegal(token, illegalSymbols):
			legalTokens.append(token)
	return legalTokens

def getBigrams(corpus):
	tokens = getTokens(corpus)
	bigrams = {}
	prev = "unk"
	for token in tokens: 
		if(prev not in bigrams): 
			bigrams[prev] = {}
		if(token not in bigrams[prev]):
			bigrams[prev][token] = 0
		bigrams[prev][token] = bigrams[prev][token] + 1
		prev = token
	return bigrams

def getUnigrams(corpus):
	tokens = getTokens(corpus)
	unigrams = {}
	for token in tokens: 
		if(token not in unigrams): 
			unigrams[token] = 1
		else:
			unigrams[token] += 1
	return unigrams

def getUniProbs(corpus):
	unigrams = getUnigrams(corpus)
	total = 0
	for unigram in unigrams.values():
		total += unigram

	total = float(total)
	probs = []
	for unigram in unigrams.values():
		probs.append(unigram/total)
	return probs

def randomSentenceUnigrams(corpus):
	unigrams = getUnigrams(corpus)
	probs = getUniProbs(corpus)
	output = ""
	current = unigrams.keys()[random.randint(0,len(unigrams) - 1)] #make this use a common word as the first
	while current != ".":
		current = np.random.choice(unigrams.keys(), 1, p=probs)[0]
		output += " " + current
	return output


def bigramProb(first, second):
	try:
		countFirst = 0
		for newSecond in bigrams[first].keys():
			countFirst = countFirst + bigrams[first][newSecond]
		return float(bigrams[first][second])/float(countFirst)
	except:
		return 0


def randomSentenceBigrams(bigrams):
	startTokens = bigrams['.'].keys()
	randomStartIndex = random.randint(0, len(startTokens) - 1)
	current = startTokens[randomStartIndex]
	output = ""
	while current != ".": 
		try:
			tokens = bigrams[current].keys()
			totalBigrams = reduce(lambda acc, k: acc + bigrams[current][k], tokens, 0)
			probs = map(lambda x: float(bigrams[current][x])/float(totalBigrams), tokens)
			token = np.random.choice(tokens, 1, p = probs)[0]
		except:
			token = "."
		output = output + " " + token
		current = token
	return output

def randomSentence(corpus, isUnigram):
	if isUnigram:
		return randomSentenceUnigrams(corpus)
	else:
		return randomSentenceBigrams(getBigrams(corpus))



def main():
	directories = ['atheism', 'autos', 'graphics', 'medicine', 'motorcycles', 'religion', 'space']
	# directories = ['atheism']
	for d in directories:
		directoryPath = 'data_corrected/classification task/' + d + '/train_docs'
		directory = path.relpath(directoryPath)
		print d
		corpus = ""
		for filename in listdir(directory):
			corpus = corpus + cleanText(directory + '/' + filename)
		#print randomSentence(corpus, False)
		print randomSentence(corpus, True)
main()
