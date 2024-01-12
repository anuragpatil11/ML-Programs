
import pandas as df
# import 
# Assuming the data is stored in a pandas DataFrame called 'df'
shape_of_data = df.shape
print("Shape of Data:", shape_of_data)

missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)

data_types = df.dtypes
print("Data Types of each Column:\n", data_types)

zero_counts = (df == 0).sum()
print("Zero Counts for each Column:\n", zero_counts)

mean_age = df["Age"].mean()
print("Mean Age of Patients:", mean_age)

from sklearn.model_selection import train_test_split

# Assuming 'X' contains the features and 'y' contains the labels (COVID positive or not)
X = df[["Age", "Sex", "ChestPain", "RestBP", "Chol"]]
y = df["COVID_Status"]  # Assuming there is a column called "COVID_Status" indicating COVID positive or not

# Split the dataset into training (75%) and testing (25%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Assuming 'y_test_pred' contains the predicted labels for the testing set
# and 'y_test' contains the true labels for the testing set
y_test_pred = ["COVID_Positive"] * 100 + ["COVID_Negative"] * (len(y_test) - 100)

# Create the confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)

# Extract values from the confusion matrix
true_negatives, false_positives, false_negatives, true_positives = conf_matrix.ravel()

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, pos_label="COVID_Positive")
recall = recall_score(y_test, y_test_pred, pos_label="COVID_Positive")
f1 = f1_score(y_test, y_test_pred, pos_label="COVID_Positive")

print("Confusion Matrix:\n", conf_matrix)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F-1 Score:", f1)
