# FIND-S ALGORITHM
import pandas as pd
# Load data
file_name = '/Users/Joanna letica Pinto/OneDrive/Desktop/ML/EnjoySport.csv'  # Corrected the variable name and added missing quotes
data = pd.read_csv(file_name)

# Initialize the hypothesis
hypothesis = ['%' for _ in range(len(data.columns) - 1)]  # Corrected the syntax

# Filter positive examples (where EnjoySport is 'Yes')
positive_examples = data[data['EnjoySport'] == 'Yes'].iloc[:, :-1].values.tolist()  # Corrected the syntax and logic

# Apply the FIND-S algorithm
for example in positive_examples: 
    for i in range(len(example)):
        if hypothesis[i] != '%' and hypothesis[i] != example[i]:
            hypothesis[i] = '?'
        else:
            hypothesis[i] = example[i]  # Corrected the syntax

# Print the maximally specific hypothesis
print("The maximally specific Find-S hypothesis for the given training examples is:")
print(hypothesis)
