# Building a supervised learning model and making predictions using it

# Importing model class from scikit-learn
from sklearn import Model  

# Create object of this class
model = Model()

# Fit the model by passing the matrix of features and the vector with how the result should look like
model.fit(X, y)

# Predict with new matrix of features the result, based on what the model learned in the fit
vector_predictions = model.predict(X_new)

# print the vector which predict function returns
print(vector_predictions)