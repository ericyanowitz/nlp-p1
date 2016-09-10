from nltk.tokenize import sent_tokenize, word_tokenize
from os import path, listdir
import numpy as np
import random


def cleanText(filePath):
	file = open(filePath)
	text = file.read()

	#Remove email headers - we don't want to generate a random string with an email header, 
	#nor are they relevant in the context of a sentence
	# scikit learn might be better for this: http://scikit-learn.org/stable/datasets/#the-20-newsgroups-text-dataset
	try:
		startSubjectIdentifier = "Subject : Re : "
		startSubject = text.index("Subject : Re : ") + len(startSubjectIdentifier)
		endSubject = text.index("In article")

		startBodyIdentifier = "writes :";
		startBody = text.index(startBodyIdentifier) + len(startBodyIdentifier)
	except:
		startSubject = 0
		endSubject = 0
		startBody = 0

	newText = text[startSubject:endSubject] + text[startBody:]
	return newText


def getBigrams(corpus):
	tokens = word_tokenize(corpus)
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


def bigramProb(first, second):
	try:
		countFirst = 0
		for newSecond in bigrams[first].keys():
			countFirst = countFirst + bigrams[first][newSecond]
		return float(bigrams[first][second])/float(countFirst)
	except:
		return 0


def randomSentence(bigrams):
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
		bigrams = getBigrams(corpus)
		print randomSentence(bigrams)
main()





