

def accuracy_score(Y,predY,mode='binary'):
	acc = 0.0
	if (mode=='binary'):
		TP = ((predY == Y) & (predY == 1.)).sum()
		TN = ((predY == Y) & (predY == 0.)).sum()	
		acc = (TP + TN) / Y.shape[0]
	elif (mode=='multi'):
		TP = (predY == Y).sum()
		acc = TP / Y.shape[0]
	return acc

def precision_score(Y,predY,mode='binary'):
	precision = 0.0
	if (mode=='binary'):
		TP = ((predY == Y) & (predY == 1)).sum()
		FP = ((predY != Y) & (predY == 1)).sum()
		precision = TP / (TP + FP)
	elif (mode=='multi'):
		classes=np.unique(Y)
		for c in classes:
			TP = ((predY == Y) & (predY == c)).sum()
			FP = ((predY != Y) & (predY == c)).sum()
			precision += TP / (TP + FP)
		precision /= len(classes)
	return precision

def recall_score(Y,predY,mode='binary'):
	recall = 0.0
	if (mode=='binary'):
		TP = ((predY == Y) & (predY == 1)).sum()
		FN = ((predY != Y) & (predY == 0)).sum()
		recall = TP / (TP + FN)
	elif (mode=='multi'):
		classes=np.unique(Y)
		for c in classes:
			TP = ((predY == Y) & (predY == c)).sum()
			FN = ((predY != Y) & (Y == c)).sum()
			recall += TP / (TP + FN)
		recall /= len(classes)
	return recall

def fb_score(Y,predY,beta,mode='binary'):
	fbscore = 0.0
	if (mode=='binary'):
		precision = PrecisionScore(predY,Y)
		recall = RecallScore(predY,Y)
		fscore = (1 + beta*beta)*((precision*recall)/((beta*beta*precision)+recall))
	elif (mode=='multi'):
		precision = PrecisionScore(predY,Y,'multi')
		recall = RecallScore(predY,Y,'multi')
		fscore = (1 + beta*beta)*((precision*recall)/((beta*beta*precision)+recall))
	return fscore