import numpy as np
import math

def loadTrainData():
	# load the training corpus

	# load vocabulary and assign index
	vocmap = dict()
	f = open('freqwords', 'r')
	c = 0
	for line in f:
		vocmap[line.rstrip()] = c
		c += 1
	vocmap['UNKA'] = c
	f.close()
	print 'Vocabulary size:', len(vocmap)

	# load sentence and label
	f = open('wsj.0-18', 'r')
	wordList = []
	labelList = []
	for line in f:
		data = line.strip().split()
		tmp = data[0::2]
		wordList.append([vocmap[s] if s in vocmap else vocmap['UNKA'] for s in tmp])
		tmp = data[1::2]
		labelList.append(tmp)
	f.close()

	# construct tagset and assign index
	tagmap = dict()
	for sent in labelList:
		for i in range(len(sent)):
			if not sent[i] in tagmap:
				tagmap[sent[i]] = len(tagmap)
			sent[i] = tagmap[sent[i]]
	print 'Tagset size:', len(tagmap)

	return wordList, labelList, vocmap, tagmap

def splitDataAndGetCounts(ratio, wordList, labelList, vocmap, tagmap):
	#split the fully labeled data by the ratio and return the count and all sentences

	# split data
	labelNum = int(ratio * len(wordList))
	unlabelWordList = wordList[labelNum:]
	unlabelLabelList = labelList[labelNum:]
	wordList = wordList[:labelNum]
	labelList = labelList[:labelNum]

	# construct parameter table
	W = len(vocmap)
	T = len(tagmap)

	t = np.zeros(T)
	tw = np.zeros((T, W))
	tpw = np.zeros((T, W))
	tnw = np.zeros((T, W))

	# calculate count to the table
	for i in range(len(wordList)):
		word = wordList[i]
		label = labelList[i]
		for j in range(len(word)):
			t[label[j]] += 1.0
			tw[label[j], word[j]] += 1.0
			if j > 0:
				tpw[label[j], word[j-1]] += 1.0
			if j < len(word) - 1:
				tnw[label[j], word[j+1]] += 1.0

	# smoothing
	smoothing(1.0, t, tw, tpw, tnw)

	return t, tw, tpw, tnw, unlabelWordList, unlabelLabelList

def smoothing(alpha, t, tw, tpw, tnw):
	# adding the smooth counts to the original ones
	T, W = tw.shape
	t += alpha / T
	tw += alpha / (T * W)
	tpw += alpha / (T * W)
	tnw += alpha / (T * W)

def loadTestData(vocmap, tagmap):
	# load and return the test data and gold label, converted into index

	# a list of sentences, each element contains a list of words
	wordList = []
	# a list of sentences, each element contains a list of labels (POS)
	labelList = []

	f = open('wsj.19-21', 'r')
	for line in f:
		data = line.strip().split()
		tmp = data[0::2]
		wordList.append([vocmap[s] if s in vocmap else vocmap['UNKA'] for s in tmp])
		tmp = data[1::2]
		labelList.append([tagmap[s] for s in tmp])
	f.close()

	return wordList, labelList

def Mstep(et, etw, etpw, etnw, t, tw, tpw, tnw):
	# ratio: the split ratio of labeled and unlabled data; used to compute weight of real counts
	
	# et: expected counts for \theta(t)
	# etw: expected counts for \theta_0(w|t)
	# etpw: expected counts for \theta_{-1}(w|t)
	# etnw: expected counts for \theta_{+1}(w|t)

	# t: real counts for \theta(t)
	# tw: real counts for \theta_0(w|t)
	# tpw: real counts for \theta_{-1}(w|t)
	# tnw: real counts for \theta_{+1}(w|t)

	# pt, ptw, ptpw, ptnw are parameters
	# pt: p(t)
	# ptw: p_0(w|t)
	# ptpw: p_{-1}(w|t)
	# ptnw: p_{+1}(w|t)

	T, W = etw.shape #tag set size and vocabulary size
	pt = np.zeros(et.shape)
	ptw = np.zeros(etw.shape)
	ptpw = np.zeros(etpw.shape)
	ptnw = np.zeros(etnw.shape)

	# c is the weight of real count
	c = 100.0

	# Estimate parameters pt, ptw, ptpw, ptnw based on the expected counts and real counts
	# Your code here:
	# T =  tag set size
	# W = Vocabulary size
	total_rt = np.sum(t)
	total_et = np.sum(et)
	total_tags = c * total_rt + total_et
	for i in range(T):
		pt[i] = (c * t[i] + et[i]) / float(total_tags)

	for tag in range(T):
		total_tw = np.sum(c * tw[tag] + etw[tag])
		for word in range(W):
			ptw[tag][word] = (c * tw[tag][word] + etw[tag][word]) / total_tw
	
	for tag in range(T):
		total_tpw = np.sum(c * tpw[tag] + etpw[tag])
		for word in range(W):
			ptpw[tag][word] = (c * tpw[tag][word] + etw[tag][word]) / total_tpw

	for tag in range(T):
		total_tnw = np.sum(c * tnw[tag] + etnw[tag])
		for word in range(W):
			ptnw[tag][word] = (c * tnw[tag][word] + etnw[tag][word]) / total_tnw

	return pt, ptw, ptpw, ptnw

def EstepA(pt, ptw, ptpw, ptnw, wordList):
	T, W = ptw.shape
	# Tables for expected counts
	# et: expected counts for \theta(t)
	# etw: expected counts for \theta_0(w|t)
	# etpw: expected counts for \theta_{-1}(w|t)
	# etnw: expected counts for \theta_{+1}(w|t)
	et = np.zeros(T)
	etw = np.zeros((T, W))
	etpw = np.zeros((T, W))
	etnw = np.zeros((T, W))

	for sent in wordList:
		for pos in range(len(sent)):
			w = sent[pos]
			for tag in range(T):
				posterior = (pt[tag] * ptw[tag][w]) / np.dot(pt, ptw.T[w])
				et[tag] += posterior
				etw[tag][w] += posterior

	return et, etw, etpw, etnw

def likelihoodA(pt, ptw, ptpw, ptnw, wordList, t, tw, tpw, tnw):
	# compute likelihood based on Model A
	l = sum([sum([np.log(sum(pt * ptw[:, word])) for word in sent]) for sent in wordList])
	# log-prior likelihood, resulting in smoothing
	c = 100.0
	l += c * (np.sum(t * np.log(pt)) + np.sum(tw * np.log(ptw)))
	return l

def EstepB(pt, ptw, ptpw, ptnw, wordList):
	T, W = ptw.shape
	# Tables for expected counts
	# et: expected counts for \theta(t)
	# etw: expected counts for \theta_0(w|t)
	# etpw: expected counts for \theta_{-1}(w|t)
	# etnw: expected counts for \theta_{+1}(w|t)
	et = np.zeros(T)
	etw = np.zeros((T, W))
	etpw = np.zeros((T, W))
	etnw = np.zeros((T, W))
	
	for sent in wordList:
		for pos in range(len(sent)):
			# Compute the posterior for the first word or other words
			# Hint: the posterior formula for the first word and others are different
			# Your code here:
			for tag in range(T):
				if pos == 0:
					w = sent[pos]
					posterior = (pt[tag] * ptw[tag][w]) / np.dot(pt, ptw.T[w])
					et[tag] += posterior
					etw[tag][w] += posterior
				else:
					w = sent[pos]
					w_p = sent[pos - 1]
					posterior = (pt[tag] * ptw[tag][w] * ptpw[tag][w_p]) / np.dot(pt, ptpw.T[w_p] * ptw.T[w])
					et[tag] += posterior
					etw[tag][w] += posterior
					etpw[tag][w_p] += posterior

	return et, etw, etpw, etnw

def likelihoodB(pt, ptw, ptpw, ptnw, wordList, t, tw, tpw, tnw):
	# compute likelihood based on Model B
	l = sum([sum([np.log(sum(pt * ptw[:, sent[i]] * ptpw[:, sent[i-1]])) if i > 0 else np.log(sum(pt * ptw[:, sent[i]])) for i in range(len(sent))]) for sent in wordList])
	# log-prior likelihood, resulting in smoothing
	c = 100.0
	l += c * (np.sum(t * np.log(pt)) + np.sum(tw * np.log(ptw)) + np.sum(tpw * np.log(ptpw)))	
	return l

def EstepC(pt, ptw, ptpw, ptnw, wordList):
	T, W = ptw.shape
	# Tables for expected counts
	# et: expected counts for \theta(t)
	# etw: expected counts for \theta_0(w|t)
	# etpw: expected counts for \theta_{-1}(w|t)
	# etnw: expected counts for \theta_{+1}(w|t)
	et = np.zeros(T)
	etw = np.zeros((T, W))
	etpw = np.zeros((T, W))
	etnw = np.zeros((T, W))
	
	for sent in wordList:
		for pos in range(len(sent)):
			# Compute the posterior for the first word, middle word or last owrd
			# Hint: the posterior formula for the first word, the last word and others are different
			# Your code here:
			if len(sent) == 1:
				w = sent[pos]
				for tag in range(T):
					posterior = (pt[tag] * ptw[tag][w]) / np.dot(pt, ptw.T[w] * ptnw.T[w_n])
					et[tag] += posterior
					etw[tag][w] += posterior
			else:
				for tag in range(T):
					if pos == 0:
						w = sent[pos]
						w_n = sent[pos + 1]
						posterior = (pt[tag] * ptw[tag][w] * ptnw[tag][w_n]) / np.dot(pt, ptw.T[w] * ptnw.T[w_n])
						et[tag] += posterior
						etw[tag][w] += posterior
						etpw[tag][w_n] += posterior
					elif pos == (len(sent) - 1):
						w_p = sent[pos - 1]
						w = sent[pos]
						posterior = (pt[tag] * ptw[tag][w] * ptpw[tag][w_p]) / np.dot(pt, ptw.T[w] * ptpw.T[w_p])
						et[tag] += posterior
						etw[tag][w] += posterior
						etpw[tag][w_p] += posterior
					else:
						w_p = sent[pos - 1]
						w = sent[pos]
						w_n = sent[pos + 1]
						posterior = (pt[tag] * ptpw[tag][w_p] * ptw[tag][w] * ptnw[tag][w_n]) / np.dot(pt, ptw.T[w] * ptpw.T[w_p] * ptnw.T[w_n])
						et[tag] += posterior
						etw[tag] += posterior
						etpw[tag][w_p] += posterior
						etnw[tag][w_n] += posterior

			# Accumulate expected counts based on posterior
			# Your code here:

	return et, etw, etpw, etnw

def likelihoodC(pt, ptw, ptpw, ptnw, wordList, t, tw, tpw, tnw):
	# compute likelihood based on Model C
	l = 0.0
	for sent in wordList:
		for pos in range(len(sent)):
			prob = pt * ptw[:, sent[pos]]
			if pos > 0:
				prob = prob* ptpw[:, sent[pos - 1]]
			if pos < len(sent) - 1:
				prob = prob * ptnw[:, sent[pos + 1]]
			l += np.log(sum(prob))
	# log-prior likelihood, resulting in smoothing
	c = 100.0
	l += c * (np.sum(t * np.log(pt)) + np.sum(tw * np.log(ptw)) + np.sum(tpw * np.log(ptpw)) + np.sum(tnw * np.log(ptnw)))
	
	return l

def predictA(wordList, pt, ptw, ptpw, ptnw):
	# wordList is the list for testing sentence; pt, ptw, ptpw, ptnw are parameters
	# pt: p(t)
	# ptw: p_0(w|t)
	# ptpw: p_{-1}(w|t)
	# ptnw: p_{+1}(w|t)
	
	# pred is the list of prediction, each element is a list of tag index predictions for each word in the sentence
	# e.g. pred = [[1,2], [2,3]]
	T, W = ptw.shape
	pred = []

	# Predict tag index in each sentence based on Model A
	for sent in wordList:
		cur_pred = []
		for pos in range(len(sent)):
			# pred_tag is the prediction of tag for the current word
			w = sent[pos]
			best_joint = -1
			best_tag = -1
			for tag in range(T):
				joint = (pt[tag] * ptw[tag][w])
				if joint > best_joint:
					best_tag = tag
					best_joint = joint
			pred_tag = best_tag
			cur_pred.append(pred_tag)
		pred.append(cur_pred)

	return pred

def predictB(wordList, pt, ptw, ptpw, ptnw):
	# wordList is the list for testing sentence; pt, ptw, ptpw, ptnw are parameters
	# pt: p(t)
	# ptw: p_0(w|t)
	# ptpw: p_{-1}(w|t)
	# ptnw: p_{+1}(w|t)


	# pred is the list of prediction, each element is a list of tag index predictions for each word in the sentence
	# e.g. pred = [[1,2], [2,3]]
	T, W = ptw.shape
	pred = []

	# Predict tag index in each sentence based on Model B
	for sent in wordList:
		cur_pred = []
		for pos in range(len(sent)):
			# pred_tag is the prediction of tag for the current word		
			# Your code here:
			# Hint: note that the probability definition is different for the first word and the rest
			if pos == 0:
				w = sent[pos]
				best_joint = -1
				best_tag = -1
				for tag in range(T):
					joint = (pt[tag] * ptw[tag][w])
					if joint > best_joint:
						best_tag = tag
						best_joint = joint
			else:
				w = sent[pos]
				w_p = sent[pos - 1]
				best_joint = -1
				best_tag = -1
				for tag in range(T):
					joint = (pt[tag] * ptw[tag][w] * ptpw[tag][w_p])
					if joint > best_joint:
						best_tag = tag
						best_joint = joint

			# append the prediction to the list
			pred_tag = best_tag
			cur_pred.append(pred_tag)
		pred.append(cur_pred)

	return pred

def predictC(wordList, pt, ptw, ptpw, ptnw):
	# wordList is the list for testing sentence; pt, ptw, ptpw, ptnw are parameters
	# pt: p(t)
	# ptw: p_0(w|t)
	# ptpw: p_{-1}(w|t)
	# ptnw: p_{+1}(w|t)

	# pred is the list of prediction, each element is a list of tag index predictions for each word in the sentence
	# e.g. pred = [[1,2], [2,3]]
	T, W = ptw.shape
	pred = []

	# Predict tag index in each sentence based on Model C
	for sent in wordList:
		cur_pred = []
		for pos in range(len(sent)):
			# pred_tag is the prediction of tag for the current word
			# Your code here:
			# Hint: note that the probability definition is different for the first word, the last word and the middle words
			if len(sent) == 1:
				w = sent[pos]
				best_joint = -1
				best_tag = -1
				for tag in range(T):
					joint = (pt[tag] * ptw[tag][w])
					if joint > best_joint:
						best_tag = tag
						best_joint = joint
			elif pos == 0:
				w = sent[pos]
				w_n = sent[pos + 1]
				best_joint = -1
				best_tag = -1
				for tag in range(T):
					joint = (pt[tag] * ptw[tag][w] * ptnw[tag][w_n])
					if joint > best_joint:
						best_tag = tag
						best_joint = joint
			elif pos == (len(sent) - 1):
				w_p = sent[pos - 1]
				w = sent[pos]
				best_joint = -1
				best_tag = -1
				for tag in range(T):
					joint = (pt[tag] * ptw[tag][w] * ptpw[tag][w_p])
					if joint > best_joint:
						best_tag = tag
						best_joint = joint
			else:
				w_p = sent[pos - 1]
				w = sent[pos]
				w_n = sent[pos + 1]
				best_joint = -1
				best_tag = -1
				for tag in range(T):
					joint = (pt[tag] * ptpw[tag][w_p] * ptw[tag][w] * ptnw[tag][w_n])
					if joint > best_joint:
						best_tag = tag
						best_joint = joint

			# append the prediction to the list
			pred_tag = best_tag
			cur_pred.append(pred_tag)
		pred.append(cur_pred)

	return pred

def evaluate(labelList, pred):
	# compute accuracy
	if len(labelList) != len(pred):
		print 'number of sentences mismatch!'
		return None
	
	acc = 0.0
	total = 0.0
	for i in range(len(labelList)):
		if len(labelList[i]) != len(pred[i]):
			print 'length mismatch on sentence', i
			return None
		total += len(labelList[i])
		acc += sum([1 if labelList[i][j] == pred[i][j] else 0 for j in range(len(labelList[i]))])
	return acc / total

def getHighestUnconditional(pt, ptw):
	T, W = ptw.shape
	highest_list = []
	highest_prob = -1
	for word in range(W):
		for tag in range(T):
			current_ptw = ptw[tag][word]
			current_posterior = (pt[tag] * ptw[tag][word]) / np.dot(pt, ptw.T[word])
			if current_posterior > highest_prob:
				highest_list = [word]
				highest_prob = current_posterior
			elif current_posterior == highest_prob:
				highest_list.append(word)
	return highest_list, highest_prob

def nums2word(word_nums, vocmap):
	words = []
	for i in range(len(word_nums)):
		for key in vocmap:
			if vocmap[key] == word_nums[i]:
				words.append(key)
	return words

def task1():
	# Hint: This function is fully implemented. Just call it and report your result

	# Test each model given labeled data
	# load the count from training corpus
	wordList, labelList, vocmap, tagmap = loadTrainData()
	print tagmap
	#print vocmap
	t, tw, tpw, tnw, unlabelWordList, unlabelLabelList = splitDataAndGetCounts(1.0, wordList, labelList, vocmap, tagmap)
	# estimate the parameters
	pt, ptw, ptpw, ptnw = Mstep(np.zeros(t.shape), np.zeros(tw.shape), np.zeros(tpw.shape), np.zeros(tnw.shape), t, tw, tpw, tnw)
	# load the testing data
	wordList, labelList = loadTestData(vocmap, tagmap)

	highest, highest_prob = getHighestUnconditional(pt , ptw)
	words = nums2word(highest, vocmap)
	print highest_prob, words

	# predict using each model and evaluate
	pred = predictA(wordList, pt, ptw, ptpw, ptnw)
	print "Model A accuracy:", evaluate(labelList, pred)

	pred = predictB(wordList, pt, ptw, ptpw, ptnw)
	print "Model B accuracy:", evaluate(labelList, pred)

	pred = predictC(wordList, pt, ptw, ptpw, ptnw)
	print "Model C accuracy:", evaluate(labelList, pred)

def task2():
	# Hint: This function is fully implemented. Just call it and report your result
	# You will get (1) the accuracy trained only on the labeled data, (2) the log-likelihood and model accuracy after each iteration

	taskem(0.5)

def task3():
	# Hint: This function is fully implemented. Just call it and report your result
	# You will get (1) the accuracy trained only on the labeled data, (2) the log-likelihood and model accuracy after each iteration

	taskem(0.01)

def taskem(ratio):
	wordList, labelList, vocmap, tagmap = loadTrainData()

	print ratio, 'labeled,', 1-ratio, 'unlabeled:'

	t, tw, tpw, tnw, unlabelWordList, unlabelLabelList = splitDataAndGetCounts(ratio, wordList, labelList, vocmap, tagmap)

	# try different models
	estepFunc = [EstepA, EstepB, EstepC]
	likelihoodFunc = [likelihoodA, likelihoodB, likelihoodC]
	predictFunc = [predictA, predictB, predictC]
	name = ['A', 'B', 'C']
	
	for m in range(len(name)):
		print 'Use model ' + name[m] + ':'
		# estimate on labeled data only
		pt, ptw, ptpw, ptnw = Mstep(np.zeros(t.shape), np.zeros(tw.shape), np.zeros(tpw.shape), np.zeros(tnw.shape), t, tw, tpw, tnw)
		pred = predictFunc[m](unlabelWordList, pt, ptw, ptpw, ptnw)
		print "Model accuracy on labeled data:", evaluate(unlabelLabelList, pred)

		# use the uniform distribution as initialization
		pt, ptw, ptpw, ptnw = Mstep(np.zeros(t.shape), np.zeros(tw.shape), np.zeros(tpw.shape), np.zeros(tnw.shape), np.ones(t.shape), np.ones(tw.shape), np.ones(tpw.shape), np.ones(tnw.shape))
		# run EM
		maxIter = 4
		Estep = estepFunc[m]
		likelihood = likelihoodFunc[m]
		for iter in range(maxIter):
			et, etw, etpw, etnw = Estep(pt, ptw, ptpw, ptnw, unlabelWordList)
			pt, ptw, ptpw, ptnw = Mstep(et, etw, etpw, etnw, t, tw, tpw, tnw)
			l = likelihood(pt, ptw, ptpw, ptnw, unlabelWordList, t, tw, tpw, tnw)
			pred = predictFunc[m](unlabelWordList, pt, ptw, ptpw, ptnw)
			print 'Iter', iter + 1, 'Log-likelihood =', l, "Model accuracy:", evaluate(unlabelLabelList, pred)

print "task 1: "
task1()
#print "\ntask 2: "
#task2()
# print "\ntask 3: "
# task3()

