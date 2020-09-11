import pandas
import numpy
import sklearn
import sklearn.model_selection
import sklearn.linear_model

data = pandas.read_csv("student-mat.csv", sep=';')
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

X = numpy.array(data.drop([predict], 1))
Y = numpy.array(data[predict])

# x_train is an arbitrarily selected section of 90% of the attributes, and x_test is the other 10%.
# y_train is an arbitrarily selected section of 90% of the labels, and y_test is the other 10%.
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)

linear = sklearn.linear_model.LinearRegression()
linear.fit(x_train, y_train)
accuracy = linear.score(x_test, y_test)
print(accuracy)

print('Coefficient: ', linear.coef_)
print('Intercept: ', linear.intercept_)

predictions = linear.predict(x_test)
for i in range(len(predictions)):
    print(predictions[i], x_test[i], y_test[i])