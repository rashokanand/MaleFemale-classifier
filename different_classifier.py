from sklearn import tree
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

clf = tree.DecisionTreeClassifier()

# CHALLENGE - create 3 more classifiers...
# 1
clf1 = svm.SVC()
# 2
clf2 = svm.NuSVC()
# 3
clf3 = svm.LinearSVC()

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=0)

# CHALLENGE - ...and train them on our data
clf = clf.fit(X_train, Y_train)
clf1 = clf1.fit(X_train, Y_train)
clf2 = clf2.fit(X_train, Y_train)
clf3 = clf3.fit(X_train, Y_train)

cvs = clf.score(X_test, Y_test)
cvs1 = clf1.score(X_test, Y_test)
cvs2 = clf2.score(X_test, Y_test)
cvs3 = clf3.score(X_test, Y_test)


prediction = clf.predict([[190, 70, 43]])

prediction1 = clf1.predict([[190, 70, 43]])
prediction2 = clf2.predict([[190, 70, 43]])
prediction3 = clf3.predict([[190, 70, 43]])

# CHALLENGE compare their reusults and print the best one!

print("Classifying accuracies: ","\n", "Decision tree: ",cvs,"\n SVC: ",cvs1,"\n NuSVC: ",cvs2,"\n LinearSVC: ",cvs3)

print("\n Predictions: ","\n", "Decision tree: ",prediction,"\n SVC: ",prediction1,"\n NuSVC: ",prediction2,"\n LinearSVC: ",prediction3)
