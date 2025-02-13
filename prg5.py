#Naïve Bayesian Classifier 

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

# Load the training dataset
train_data = pd.read_csv('/Users/Joanna letica Pinto/OneDrive/Desktop/ML/PlayTennis.csv')
# Load the testing dataset
test_data = pd.read_csv('/Users/Joanna letica Pinto/OneDrive/Desktop/ML/Test.csv')

# Mapping categorical variables to numerical values for both datasets
train_data['Outlook'] = train_data['Outlook'].map({'Sunny': 0, 'Overcast': 1, 'Rain': 2})
train_data['Temperature'] = train_data['Temperature'].map({'Hot': 0, 'Mild': 1, 'Cool': 2})
train_data['Humidity'] = train_data['Humidity'].map({'High': 0, 'Normal': 1})
train_data['Wind'] = train_data['Wind'].map({'Weak': 0, 'Strong': 1})
train_data['PlayTennis'] = train_data['PlayTennis'].map({'No': 0, 'Yes': 1})

test_data['Outlook'] = test_data['Outlook'].map({'Sunny': 0, 'Overcast': 1, 'Rain': 2})
test_data['Temperature'] = test_data['Temperature'].map({'Hot': 0, 'Mild': 1, 'Cool': 2})
test_data['Humidity'] = test_data['Humidity'].map({'High': 0, 'Normal': 1})
test_data['Wind'] = test_data['Wind'].map({'Weak': 0, 'Strong': 1})
test_data['PlayTennis'] = test_data['PlayTennis'].map({'No': 0, 'Yes': 1})

# Separate features and target variable for both datasets
X_train = train_data.drop('PlayTennis', axis=1)
y_train = train_data['PlayTennis']
X_test = test_data.drop('PlayTennis', axis=1)
y_test = test_data['PlayTennis']

# Uncomment the following line to print out the actual labels of the test data
# print(y_test)

# Initialize Gaussian Naive Bayes classifier
nb_classifier = GaussianNB()

# Train the classifier
nb_classifier.fit(X_train, y_train)

# Predict the target variable
y_pred = nb_classifier.predict(X_test)

# Print predicted output
print("Predicted Output:")
for i, prediction in enumerate(y_pred):
    print(f"Sample {i + 1}: {'Yes' if prediction == 1 else 'No'}")

# Calculate accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

