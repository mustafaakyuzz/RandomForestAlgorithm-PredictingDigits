import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

# Load dataset
from sklearn.datasets import load_digits

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

digits = load_digits()

# To show images
plt.gray()
for i in range(10):
    plt.matshow(digits.images[i])
    plt.axis("off")
    
# Showing each numbers features
df = pd.DataFrame(digits.data)
df['target'] = digits.target
print(df.head())

X = df.drop(['target'], axis = 'columns')
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

rf_clf = RandomForestClassifier(n_estimators = 20)

# Train the model
rf_clf.fit(X_train, y_train)

# Predicting
y_pred = rf_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: ', accuracy)

cm = confusion_matrix(y_test, y_pred)

# Showing heatmap
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot = True)
plt.xlabel('Predicted')
plt.ylabel('Truth')         

