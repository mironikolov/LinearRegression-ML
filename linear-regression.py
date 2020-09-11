import pandas
import numpy
import sklearn
import sklearn.model_selection
import sklearn.linear_model
import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle

data = pandas.read_csv("student-mat.csv", sep=';')
data = data[["G1", "G2", "G3", "studytime", "failures", "absences", "age", "freetime", "Dalc"]]

predict = "G3"

X = numpy.array(data.drop([predict], 1))
Y = numpy.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)
'''
best = 0
for _ in range(30):
    # x_train is an arbitrarily selected section of 90% of the attributes, and x_test is the other 10%.
    # y_train is an arbitrarily selected section of 90% of the labels, and y_test is the other 10%.
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)

    linearModel = sklearn.linear_model.LinearRegression()
    linearModel.fit(x_train, y_train)

    accuracy = linearModel.score(x_test, y_test)

    if accuracy > best:
        print('Accuracy: ', accuracy)
        best = accuracy
        with open("studentmodel.pickle", "wb") as file:
            pickle.dump(linearModel, file)
'''
pickle_in = open("studentmodel.pickle", "rb")
linearModel = pickle.load(pickle_in)


print('Coefficient: ', linearModel.coef_)
print('Intercept: ', linearModel.intercept_)

# predictions = linearModel.predict(x_test)
# for i in range(len(predictions)):
#     print(predictions[i], x_test[i], y_test[i])

#Visualize attribute effect
effect = 'Dalc'
style.use("ggplot")
pyplot.scatter(data[effect], data["G3"])
pyplot.xlabel(effect)
pyplot.ylabel("Final Grade")
pyplot.show()