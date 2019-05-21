from sklearn import tree
f=[[140,1],[130,1],[150,0],[170,0]]
l=[0,0,1,1]
clf=tree.DecisionTreeClassifier()
clf=clf.fit(f,l)
print (clf.predict([[13,0]]))

