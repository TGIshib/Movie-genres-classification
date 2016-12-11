import io
from sklearn.svm import SVC
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

X = []
Y = []
data = io.open('D:\data.txt', 'r', encoding='utf-8').readlines()
counter = 2
description = ""
has_labels = True
for line in data:
    if counter == 0:
        labels = fit_genres(line)
        if len(labels) != 0:
            Y.append(labels)
            has_labels = True
        else:
            has_labels = False
    if line == "\n":
        if has_labels == True:
            X.append(description)
        counter = 2
        description = ""
    if counter < 0:
        description += line
    counter -= 1

count_vect = CountVectorizer()
mlb = MultiLabelBinarizer()
X_transformed = count_vect.fit_transform(X).toarray()
Y_transformed = mlb.fit_transform(Y)
classif = OneVsRestClassifier(SVC(kernel='linear'))
predicter = classif.fit(X_transformed, Y_transformed)

# Comedy Drama
test_review = ["Forrest Gump is a simple man with a low I.Q. but good intentions. He is\
running through childhood with his best and only friend Jenny. His 'mama'\
teaches him the ways of life and leaves him to choose his destiny. Forrest\
joins the army for service in Vietnam, finding new friends called Dan and\
Bubba, he wins medals, creates a famous shrimp fishing fleet, inspires\
people to jog, starts a ping-pong craze, creates the smiley, writes bumper\
stickers and songs, donates to people and meets the president several\
times. However, this is all irrelevant to Forrest who can only think of his\
childhood sweetheart Jenny Curran, who has messed up her life. Although in\
the end all he wants to prove is that anyone can love anyone."]

test = count_vect.transform(test_review).toarray()
predicted = predicter.predict(test)
predicted = mlb.inverse_transform(predicted)

print ', '.join(all_genres[ind] for ind in predicted[0])
