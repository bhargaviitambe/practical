#1. Matplotlib
import pandas as pd

data = {"Student": ["A", "B", "C", "D", "E"],
        "Marks": [65, 72, 81, 56, 90]}

df = pd.DataFrame(data)
print("Student Data:\n", df)
print("Average Marks:", df["Marks"].mean())
print("Highest Marks:", df["Marks"].max())

import matplotlib.pyplot as plt

months = ["Jan", "Feb", "Mar", "Apr", "May"]
attendance = [85, 90, 78, 92, 88]

plt.plot(months, attendance, marker="o")
plt.title("Monthly Student Attendance")
plt.xlabel("Month")
plt.ylabel("Attendance (%)")
plt.show()

#2 Linear Regression- dataset

import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import datasets, linear_model 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score 

diabetes = datasets.load_diabetes() 

X = diabetes.data 
y = diabetes.target 

X_train, X_test, y_train, y_test = train_test_split(    X, y, test_size=0.4, random_state=1 )

lin_reg = linear_model.LinearRegression() 

lin_reg.fit(X_train, y_train) 

predicted = lin_reg.predict(X_test) 

print('\nCoefficients:\n', lin_reg.coef_) 
print('\nIntercept:', lin_reg.intercept_) 

print('Variance score:', lin_reg.score(X_test, y_test)) 

print("Mean squared error: %.2f\n" % mean_squared_error(y_test, predicted)) 

plt.title('Linear Regression (Diabetes Dataset)') 
plt.scatter(y_test, predicted, c='b', marker='.', s=36) 
plt.plot(np.linspace(0, 330, 100), np.linspace(0, 330, 100), '--r', linewidth=2) 
plt.xlabel("Expected values") 
plt.ylabel("Predicted values") 
plt.show()

#2 Linear Regression- values
# w/o dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([[1],[2],[3],[4],[5],[6],[7],[8]])
Y = np.array([35,42,50,55,63,70,78,85])

model = LinearRegression()
model.fit(X, Y)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

hours = np.array([[6.5]])
print("Predicted Score:", model.predict(hours)[0])
print("R2 Score:", model.score(X, Y))

plt.scatter(X, Y, label="Actual Data")
plt.plot(X, model.predict(X), label="Regression Line")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Linear Regression")
plt.legend()
plt.grid()
plt.show()

#3 Naive bayes

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

msg = pd.read_csv('document.csv', names=['message', 'label'])

print("Total Instances of Dataset:", msg.shape[0])

msg = msg.dropna()

msg['labelnum'] = msg['label'].map({'pos': 1, 'neg': 0})

msg = msg.dropna(subset=['labelnum'])

X = msg['message']
y = msg['labelnum']

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.25, random_state=42
)

count_v = CountVectorizer()
Xtrain_dm = count_v.fit_transform(Xtrain)
Xtest_dm = count_v.transform(Xtest)

df = pd.DataFrame(
    Xtrain_dm.toarray(),
    columns=count_v.get_feature_names_out()
)

print("\nWord Frequency Matrix:")
print(df)

clf = MultinomialNB()
clf.fit(Xtrain_dm, ytrain)

pred = clf.predict(Xtest_dm)

print('\nAccuracy Metrics:')
print('Accuracy:', accuracy_score(ytest, pred))
print('Recall:', recall_score(ytest, pred, zero_division=0))
print('Precision:', precision_score(ytest, pred, zero_division=0))
print('Confusion Matrix:\n', confusion_matrix(ytest, pred))

#4 KNN

from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_iris 
irisData = load_iris() 

X = irisData.data 
y = irisData.target 

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, random_state=42) 
knn = KNeighborsClassifier(n_neighbors=7) 
knn.fit(X_train, y_train) 
print(knn.predict(X_test) )

#4 SVM

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

data = pd.read_csv('emails.csv')

print("Total Instances of Dataset:", data.shape[0])

data = data.drop('Email No.', axis=1)

X = data.drop('Prediction', axis=1)
y = data['Prediction']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = LinearSVC(random_state=42, max_iter=10000)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("\nSVM Classification Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#5 K- means

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist 

data = np.array([ [1.713, 1.586], [0.180, 1.786], [0.353, 1.240], [0.940, 1.566], [1.486, 0.759], [1.266, 1.106], [1.540, 0.419], [0.459, 1.799], [0.773, 0.186] ])  
labels = np.array([0, 1, 1, 0, 1, 0, 1, 1, 1]) 

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10) 
kmeans.fit(data) 

new_point = np.array([[0.906, 0.606]])
predicted_cluster = kmeans.predict(new_point) 

# Assign class based on the majority class in the predicted cluster 
cluster_labels = kmeans.labels_ 
cluster_classes = {i: np.bincount(labels[cluster_labels == i]).argmax() for i in range(3)} 
predicted_class = cluster_classes[predicted_cluster[0]] 

plt.scatter(data[:, 0], data[:, 1], c=cluster_labels, cmap='viridis', marker='o', label='Data Points') 
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200, label='Centroids') 
plt.scatter(new_point[:, 0], new_point[:, 1], c='blue', marker='s', s=100, label='New Point') 
plt.xlabel('VAR1' ) 
plt.ylabel('VAR2' ) 
plt.legend() 
plt.title('K-Means Clustering with New Data Point') 
plt.show() 
print(f"Predicted cluster: {predicted_cluster[0]}") 
print(f"Predicted class for (0.906, 0.606): {predicted_class}")

#5 random forest

import pandas as pd 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
import matplotlib.pyplot as plt 
# Load dataset 
df = pd.read_csv('car_evaluation.csv', names=["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]) 
# Encode categorical variables 
le = LabelEncoder() 
for column in df.columns: 
    df[column] = le.fit_transform(df[column])
X = df.drop('class', axis=1) 
y = df['class'] 
# Split data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
# Train model 
model = RandomForestClassifier(n_estimators=100, random_state=42) 
model.fit(X_train, y_train)
# Predict and Evaluate 
y_pred = model.predict(X_test) 
print("Accuracy:", accuracy_score(y_test, y_pred)) 
print("Classification Report:\n", classification_report(y_test, y_pred)) 
# Feature Importance 
importances = model.feature_importances_ 
plt.figure(figsize=(8, 6)) 
plt.barh(X.columns, importances) 
plt.xlabel("Feature Importance") 
plt.title("Random Forest Feature Importance")
plt.grid() 
plt.show() 

#6 Ada and gradient

from sklearn.datasets import load_iris 
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier 
from xgboost import XGBClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, classification_report 
iris = load_iris() 
X, y = iris.data, iris.target 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
ada = AdaBoostClassifier(n_estimators=50) 
ada.fit(X_train, y_train) 
y_pred_ada = ada.predict(X_test) 
gb = GradientBoostingClassifier(n_estimators=100) 
gb.fit(X_train, y_train) 
y_pred_gb = gb.predict(X_test) 
xgb = XGBClassifier( eval_metric='mlogloss') 
xgb.fit(X_train, y_train) 
y_pred_xgb = xgb.predict(X_test) 
print("AdaBoost Accuracy:", accuracy_score(y_test, y_pred_ada)) 
print("Gradient Boosting Accuracy:", accuracy_score(y_test, y_pred_gb)) 
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb)) 

#7 Q learn taxi

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("Taxi-v4")

actions = env.action_space.n
states = env.observation_space.n
Q = np.zeros((states, actions))

episodes = 1000
learning_rate = 0.7
discount = 0.618
epsilon = 1.0
epsilon_decay = 0.01

rewards = []

for ep in range(episodes):

    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:

        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        
        next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        old_value = Q[state, action]
        next_max = np.max(Q[next_state])

        Q[state, action] = old_value + learning_rate * (
            reward + discount * next_max - old_value
        )

        state = next_state
        total_reward += reward
    epsilon = max(0.01, epsilon - epsilon_decay)

    rewards.append(total_reward)

plt.plot(rewards)
plt.title("Episode Rewards Over Time")
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.grid()
plt.show()

env.close()

#8 tictactoe 

#assgn 8
import random

# a. Setting up environment
def new_board():
    return [' '] * 9

# b. Defining Tic-Tac-Toe game
def winner(b):
    wins = [(0,1,2),(3,4,5),(6,7,8),
            (0,3,6),(1,4,7),(2,5,8),
            (0,4,8),(2,4,6)]
    for a,c,d in wins:
        if b[a] == b[c] == b[d] != ' ':
            return b[a]
    return None

def display(b):
    for i in range(0,9,3):
        print('|'.join(b[i:i+3]))
        if i < 6:
            print('-----')

# c. Q-Learning model
Q = {}
lr = 0.1
gamma = 0.9

def choose_action(b, epsilon):
    moves = [i for i in range(9) if b[i] == ' ']
    state = tuple(b)

    if random.random() < epsilon:
        return random.choice(moves)

    return max(moves, key=lambda a: Q.get((state,a), 0))

def update(state, action, reward, next_state, done):
    old = Q.get((state,action), 0)
    future = 0 if done else max(
        [Q.get((next_state,a),0)
         for a in range(9) if next_state[a] == ' '] or [0]
    )
    Q[(state,action)] = old + lr * (
        reward + gamma * future - old
    )

# d. Training the model
for episode in range(5000):
    b = new_board()

    while True:
        # Random opponent plays X
        x_moves = [i for i in range(9) if b[i] == ' ']
        b[random.choice(x_moves)] = 'X'

        if winner(b) == 'X':
            break

        if ' ' not in b:
            break

        state = tuple(b)
        action = choose_action(b, 0.2)
        b[action] = 'O'

        if winner(b) == 'O':
            update(state, action, 1, tuple(b), True)
            break

        if ' ' not in b:
            update(state, action, 0, tuple(b), True)
            break

        # Opponent's next move
        x_moves = [i for i in range(9) if b[i] == ' ']
        b[random.choice(x_moves)] = 'X'

        if winner(b) == 'X':
            update(state, action, -1, tuple(b), True)
            break

        update(state, action, 0, tuple(b), False)

print("Training completed!")

# e. Testing the trained model
b = new_board()

while True:
    display(b)

    move = int(input("Enter position (0-8): "))
    if move not in range(9) or b[move] != ' ':
        print("Invalid move!")
        continue

    b[move] = 'X'

    if winner(b) == 'X':
        display(b)
        print("You win!")
        break

    if ' ' not in b:
        display(b)
        print("Draw!")
        break

    action = choose_action(b, 0)
    b[action] = 'O'
    print("Agent chose:", action)

    if winner(b) == 'O':
        display(b)
        print("Agent wins!")
        break

    if ' ' not in b:
        display(b)
        print("Draw!")
        break
