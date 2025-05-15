import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Import AI algorithms
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Setup directories
data_folder = 'Data'
original_data_folder = os.path.join(data_folder, 'original_data')
preprocessed_data_folder = os.path.join(data_folder, 'preprocessed_data')
results_folder = os.path.join(data_folder, 'Results')
os.makedirs(results_folder, exist_ok=True)

# Load original dataset
dataset_path = os.path.join(original_data_folder, 'glaucoma_dataset.csv')
df = pd.read_csv(dataset_path)

# Clean column names
df.columns = df.columns.str.strip()

# Define features and target
features = ["Age", "Intraocular Pressure (IOP)", "Cup-to-Disc Ratio (CDR)", "Family History", "Medical History", "Medication Usage"]
target = "Diagnosis"

# Check if required features exist
existing_features = [col for col in features if col in df.columns]
missing_features = set(features) - set(existing_features)

if missing_features:
    print(f"⚠ Warning: The following columns were not found in the dataset: {missing_features}")

# Extract features and target
X = df[existing_features]
Y = df[target]

# Convert categorical variables to numerical using One-Hot Encoding
X = pd.get_dummies(X, columns=["Family History", "Medical History", "Medication Usage"], drop_first=True)

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Handle missing values
X_train.fillna(X_train.mean(), inplace=True)
X_test.fillna(X_test.mean(), inplace=True)

# Save preprocessed data
X_train.to_csv(os.path.join(preprocessed_data_folder, 'X.csv'), index=False)
X_test.to_csv(os.path.join(preprocessed_data_folder, 'X_test.csv'), index=False)
Y_train.to_csv(os.path.join(preprocessed_data_folder, 'Y.csv'), index=False)
Y_test.to_csv(os.path.join(preprocessed_data_folder, 'Y_test.csv'), index=False)

print("✅ Training and test data successfully updated with real dataset!")

# Normalize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    'SVM': SVC(),
    'NaiveBayes': GaussianNB(),
    'KNN': KNeighborsClassifier(),
    'RandomForest': RandomForestClassifier(),
    'DecisionTree': DecisionTreeClassifier(),
    'LogisticRegression': LogisticRegression(),
    'ANN': MLPClassifier(max_iter=500)
}

# Train models and calculate accuracy
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, Y_train.values.ravel())
    predictions = model.predict(X_test_scaled)
    acc = accuracy_score(Y_test, predictions)
    results[name] = acc
    
    # Save results for each model
    pd.DataFrame(predictions, columns=[f'{name}_prediction']).to_csv(os.path.join(results_folder, f'predictions_{name}_model.csv'), index=False)
    
    print(f'{name} Accuracy: {acc:.4f}')

# Visualize model performance using a bar plot
plt.figure(figsize=(8, 6))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.title('Model Accuracies')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.xticks(rotation=30)
plt.savefig(os.path.join(results_folder, 'model_accuracies.png'))
plt.show()
