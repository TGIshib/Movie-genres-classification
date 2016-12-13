import io
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer

all_genres = sorted(['Drama', 'Comedy', 'Action', 'Thriller', 'Romance', 'Family', 'Documentary', 'Crime', 'Adventure'])

len_all_genres = len(all_genres)

def fit_genres(line):
    labels = []
    genres = line.split(' ');
    p1 = 0
    p2 = 0
    len_genres = len(genres)
    while p1 < len_all_genres and p2 < len_genres:
        if all_genres[p1] == genres[p2]:
            labels.append(p1)
            p1 += 1
            p2 += 1
            continue
        if all_genres[p1] > genres[p2]:
            p2 += 1
        else:
            p1 += 1
    return labels

def teach_and_predict():
	X = []
	Y = []
	data = io.open('D:\data.txt', 'r', encoding='utf-8').readlines()
	counter = 2
	num_of_learn = 40000
	num_of_test = 8000
	test_reviews = []
    test_labels = []
	description = ""
	has_labels = True
	for line in data:
		if counter == 0:
			labels = fit_genres(line)
			if len(labels) != 0:
				if num_of_learn > 0:
					Y.append(labels)
				else:
					test_labels.append(labels)
				has_labels = True
			else:
				has_labels = False
		if line == "\n":
			if has_labels == True:
				if num_of_learn > 0:
					X.append(description)
				else:
					test_reviews.append([description])
			counter = 2
			description = ""
		if counter < 0:
			description += line
		counter -= 1
		num_of_learn -= 1
		if num_of_learn < 0:
			num_of_test -= 1
			if num_of_test < 0:
				break

	count_vect = CountVectorizer()
	mlb = MultiLabelBinarizer()
	X_transformed = count_vect.fit_transform(X).toarray()
	Y_transformed = mlb.fit_transform(Y)
    #classif = OneVsRestClassifier(SVC(kernel='linear'))
    classif = OneVsRestClassifier(LinearSVC())
	predicter = classif.fit(X_transformed, Y_transformed)

	f = open('D:\dans.txt', 'w')

    for review, labels in zip(test_reviews, test_labels):
        predicted_labels = mlb.inverse_transform(predicter.predict(count_vect.transform(review).toarray()))
        f.write('t: ' + ' '.join(str(i) for i in labels))
        f.write("\n")
        f.write('p: ' + ' '.join(str(i) for i in predicted_labels[0]))
        f.write("\n")
		
    f.close()

def num_of_intersection(test, predicted):
    p1 = p2 = res = 0
    while p1 < len(test) and p2 < len(predicted):
        if test[p1] == predicted[p2]:
            res += 1
            p1 += 1
            p2 += 1
        elif test[p1] < predicted[p2]:
            p1 += 1
        else:
            p2 += 1
    return res

def calc_stats(test, predicted):
    prec = recall = f1 = 0.0
    for tlabels, prlabels in zip(test, predicted):
        num_of_correct = num_of_intersection(tlabels, prlabels)
        prec += num_of_correct / len(prlabels)
        recall += num_of_correct / len(tlabels)
    prec = prec / len(test)
    recall = recall / len(test)
    f1 = 2 * prec * recall / (prec + recall)
    return prec, recall, f1

def read_labels():
    test = []
    predicted = []
    data = io.open('D:\dans.txt', 'r', encoding='utf-8').readlines()
    for i in range(len(data)):
        line = data[i]
        if line[0] == 'p':
            labels = [num for num in line[3:].replace('\n', '').split(' ')]
            if labels[0] != '':
                predicted.append([int(num) for num in labels])
                test.append([int(num) for num in data[i-1][3:].replace('\n', '').split(' ')])
    return test, predicted

def run():
	teach_and_predict()
	test, predicted = read_labels()
	return calc_stats(test, predicted)
	
print run()
