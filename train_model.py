import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import json

print("Memuat data...")
df = pd.read_csv('WineQT.csv')

X = df.drop(['quality', 'Id'], axis=1)
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_pred)
nb_f1 = f1_score(y_test, nb_pred, average='weighted')

id3_model = DecisionTreeClassifier(criterion='entropy', random_state=42)
id3_model.fit(X_train, y_train)
id3_pred = id3_model.predict(X_test)
id3_accuracy = accuracy_score(y_test, id3_pred)
id3_f1 = f1_score(y_test, id3_pred, average='weighted')

if nb_f1 > id3_f1:
    best_model = nb_model
    best_model_name = "Naive Bayes"
else:
    best_model = id3_model
    best_model_name = "ID3 (Decision Tree)"

print(f"\nModel terbaik berdasarkan F1-Score adalah: {best_model_name}")

joblib.dump(best_model, 'best_model.pkl')
feature_names = X.columns.tolist()
joblib.dump(feature_names, 'feature_names.pkl')

performance_data = {
    "naive_bayes": {
        "accuracy": round(nb_accuracy, 4),
        "f1_score": round(nb_f1, 4)
    },
    "id3": {
        "accuracy": round(id3_accuracy, 4),
        "f1_score": round(id3_f1, 4)
    },
    "best_model_name": best_model_name
}

with open('model_performance.json', 'w') as f:
    json.dump(performance_data, f, indent=4)

print("\nModel terbaik, nama fitur, dan data kinerja telah disimpan.")