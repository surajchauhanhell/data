import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Dataset
data = {
"Outlook": ["Sunny","Sunny","Overcast","Rain","Rain","Rain","Overcast","Sunny","Sunny","Rain",
            "Sunny","Overcast","Overcast","Rain","Sunny","Overcast","Overcast","Rain","Sunny","Rain"],
"Wind": ["Weak","Strong","Weak","Weak","Weak","Strong","Strong","Weak","Weak","Weak",
         "Strong","Strong","Weak","Strong","Weak","Strong","Weak","Weak","Weak","Strong"],
"PlayTennis": ["No","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes",
               "Yes","Yes","Yes","No","Yes","No","Yes","No","No","Yes"]
}

df = pd.DataFrame(data)

print("Dataset:\n", df)

# Encode categorical variables
le_outlook = LabelEncoder()
le_wind = LabelEncoder()
le_play = LabelEncoder()

df["Outlook"] = le_outlook.fit_transform(df["Outlook"])
df["Wind"] = le_wind.fit_transform(df["Wind"])
df["PlayTennis"] = le_play.fit_transform(df["PlayTennis"])

# Features and target
X = df[["Outlook","Wind"]]
y = df["PlayTennis"]

# Train Decision Tree
model = DecisionTreeClassifier()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

print("\nPredictions:", y_pred)

# Accuracy
accuracy = accuracy_score(y, y_pred)
print("Accuracy:", accuracy)

# Decision Tree Rules
rules = export_text(model, feature_names=["Outlook","Wind"])
print("\nDecision Tree Rules:\n")
print(rules)
