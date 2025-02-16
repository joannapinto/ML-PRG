#ID3 (Desicion Tree)
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder

class ID3DecisionTree:
    def __init__(self):
        self.model = DecisionTreeClassifier(criterion='entropy')
        self.label_encoders = {}

    def fit(self, data, target_attr):
        X, y = data.drop(columns=[target_attr]), data[target_attr]
        for col in X.select_dtypes(include='object').columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.label_encoders[col] = le
        y = LabelEncoder().fit_transform(y)
        self.model.fit(X, y)

    def predict(self, query):
        query_df = pd.DataFrame([{
            col: self.label_encoders[col].transform([val])[0] if col in self.label_encoders else val
            for col, val in query.items()
        }])
        return self.model.classes_[self.model.predict(query_df)[0]]

    def visualize(self, feature_names, class_names):
        plot_tree(self.model, feature_names=feature_names, class_names=class_names, filled=True)

# Load data from Excel
file_path = 'data.xlsx'  # Replace with your Excel file path
data = pd.read_excel(file_path)

target_attr = 'Play'  # Replace with the target column in your dataset
id3 = ID3DecisionTree()
id3.fit(data, target_attr)

# Example query
query = {'Outlook': 'Sunny', 'Temperature': 'Cool', 'Humidity': 'High', 'Windy': 'False'}
print("Prediction:", id3.predict(query))

# Visualization
id3.visualize(feature_names=data.columns[:-1], class_names=['No', 'Yes'])
