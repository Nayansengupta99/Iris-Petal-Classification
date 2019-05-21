from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np
iris=load_iris()
print (iris.feature_names)
print (iris.target_names)
print (iris.data[0])
print (iris.target[0])
for i in range(len(iris.target)):
    print("Example%d: label%s features %s" % (i,iris.target[i],iris.data[i]))
test_del=[0,50,100]
training_target=np.delete(iris.target,test_del)
training_data=np.delete(iris.data,test_del,axis=0)
testtarget=iris.target[test_del]
testdata=iris.data[test_del]
clf=tree.DecisionTreeClassifier()
clf.fit(training_data,training_target)
print(testtarget)
print(clf.predict(testdata))
