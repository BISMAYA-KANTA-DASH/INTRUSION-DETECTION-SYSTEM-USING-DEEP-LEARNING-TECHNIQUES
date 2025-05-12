import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.utils import plot_model
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import joblib

pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')

# Load the dataset
data_train = pd.read_csv("KDDTrain.txt")

# Rename columns
columns = (['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
            'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 
            'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 
            'dst_host_serror_rate', 'dst_host_srv_serror_rate', 
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'outcome', 'level'])
data_train.columns = columns

# Data exploration
data_train.head()
data_train.info()
data_train.describe().style.background_gradient(cmap='Blues').set_properties(**{'font-family': 'Segoe UI'})

# Reformat outcome labels
data_train.loc[data_train['outcome'] == "normal", "outcome"] = 0
data_train.loc[data_train['outcome'] != 0, "outcome"] = 1

# Pie chart function
def pie_plot(df, cols_list, rows, cols):
    fig, axes = plt.subplots(rows, cols)
    for ax, col in zip(axes.ravel(), cols_list):
        df[col].value_counts().plot(ax=ax, kind='pie', figsize=(15, 15), fontsize=10, autopct='%1.0f%%')
        ax.set_title(str(col), fontsize=12)
    plt.show()

# Plot pie charts
pie_plot(data_train, ['protocol_type', 'outcome'], 1, 2)

# Scaling function
def Scaling(df_num, cols):
    std_scaler = RobustScaler()
    std_scaler_temp = std_scaler.fit_transform(df_num)
    std_df = pd.DataFrame(std_scaler_temp, columns=cols)
    return std_df

# Categorical columns
cat_cols = ['is_host_login', 'protocol_type', 'service', 'flag', 'land', 
            'logged_in', 'is_guest_login', 'level', 'outcome']

# Preprocessing function
def preprocess(dataframe):
    df_num = dataframe.drop(cat_cols, axis=1)
    num_cols = df_num.columns
    scaled_df = Scaling(df_num, num_cols)
    
    dataframe.drop(labels=num_cols, axis="columns", inplace=True)
    dataframe[num_cols] = scaled_df[num_cols]
    
    dataframe.loc[dataframe['outcome'] == "normal", "outcome"] = 0
    dataframe.loc[dataframe['outcome'] != 0, "outcome"] = 1
    
    dataframe = pd.get_dummies(dataframe, columns=['protocol_type', 'service', 'flag'])
    return dataframe

# Preprocess the data
scaled_train = preprocess(data_train)

train_columns = scaled_train.columns

# Save the column names using joblib
joblib.dump(train_columns, 'train_columns.pkl')

# Display unique values
for column in data_train.columns:
    print(column.upper(), ':', data_train[column].nunique())

# Prepare features and labels
x = scaled_train.drop(['outcome', 'level'], axis=1).values
y = scaled_train['outcome'].values

# PCA
pca = PCA(n_components=20)
x_reduced = pca.fit_transform(x)
print("Number of original features is {} and of reduced features is {}".format(x.shape[1], x_reduced.shape[1]))

# Split the data
y = y.astype('int')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train_reduced, x_test_reduced, y_train_reduced, y_test_reduced = train_test_split(x_reduced, y, test_size=0.2, random_state=42)

# Classification evaluation dictionary
kernal_evals = dict()

# Evaluation function for classification models
def evaluate_classification(model, name, X_train, X_test, y_train, y_test):
    # Evaluate model performance
    train_accuracy = metrics.accuracy_score(y_train, model.predict(X_train))
    test_accuracy = metrics.accuracy_score(y_test, model.predict(X_test))
    
    train_precision = metrics.precision_score(y_train, model.predict(X_train))
    test_precision = metrics.precision_score(y_test, model.predict(X_test))
    
    train_recall = metrics.recall_score(y_train, model.predict(X_train))
    test_recall = metrics.recall_score(y_test, model.predict(X_test))
    
    kernal_evals[str(name)] = [train_accuracy, test_accuracy, train_precision, test_precision, train_recall, test_recall]
    
    print(f"Training Accuracy {name}: {train_accuracy * 100:.2f}%, Test Accuracy {name}: {test_accuracy * 100:.2f}%")
    print(f"Training Precision {name}: {train_precision * 100:.2f}%, Test Precision {name}: {test_precision * 100:.2f}%")
    print(f"Training Recall {name}: {train_recall * 100:.2f}%, Test Recall {name}: {test_recall * 100:.2f}%")
    
    actual = y_test
    predicted = model.predict(X_test)
    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=['normal', 'attack'])

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.grid(False)
    cm_display.plot(ax=ax)
    plt.title(f'Confusion Matrix for {name}')
    plt.show()

# Logistic Regression model
lr = LogisticRegression(class_weight='balanced').fit(x_train, y_train)
evaluate_classification(lr, "Logistic Regression", x_train, x_test, y_train, y_test)
joblib.dump(lr, 'model_lr.pkl')

# KNeighborsClassifier model
knn = KNeighborsClassifier(n_neighbors=20).fit(x_train, y_train)
evaluate_classification(knn, "KNeighborsClassifier", x_train, x_test, y_train, y_test)
joblib.dump(knn, 'model_knn.pkl')

# GaussianNB model
gnb = GaussianNB().fit(x_train, y_train)
evaluate_classification(gnb, "GaussianNB", x_train, x_test, y_train, y_test)
joblib.dump(gnb, 'model_gnb.pkl')

# Linear SVC model
lin_svc = svm.LinearSVC(class_weight='balanced').fit(x_train, y_train)
evaluate_classification(lin_svc, "Linear SVC(LBasedImpl)", x_train, x_test, y_train, y_test)
joblib.dump(lin_svc, 'model_linear_svc.pkl')

# Decision Tree models
dt = DecisionTreeClassifier(max_depth=3).fit(x_train, y_train)
tdt = DecisionTreeClassifier(class_weight='balanced').fit(x_train, y_train)
evaluate_classification(tdt, "DecisionTreeClassifier", x_train, x_test, y_train, y_test)
joblib.dump(tdt, 'model_tdt.pkl')

# Feature importance visualization
def f_importances(coef, names, top=-1):
    imp = coef
    imp, names = zip(*sorted(list(zip(imp, names))))
    
    if top == -1:
        top = len(names)
    
    plt.figure(figsize=(10, 10))
    plt.barh(range(top), imp[::-1][0:top], align='center')
    plt.yticks(range(top), names[::-1][0:top])
    plt.title('Feature Importances for Decision Tree')
    plt.show()

features_names = data_train.drop(['outcome', 'level'], axis=1)
f_importances(abs(tdt.feature_importances_), features_names, top=18)

# Plot decision tree
fig = plt.figure(figsize=(15, 12))
tree.plot_tree(dt, filled=True)

# Random Forest model
rf = RandomForestClassifier(class_weight='balanced').fit(x_train, y_train)
evaluate_classification(rf, "Random Forest", x_train, x_test, y_train, y_test)
joblib.dump(rf, 'model_rf.pkl')

# XGBoost model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss').fit(x_train, y_train)
evaluate_classification(xgb_model, "XGBoost", x_train, x_test, y_train, y_test)
joblib.dump(xgb_model, 'model_xgb.pkl')

# Neural Network model
model_nn = tf.keras.Sequential()
model_nn.add(tf.keras.layers.InputLayer(input_shape=(x_train.shape[1],)))
model_nn.add(tf.keras.layers.Dense(units=512, activation='relu'))
model_nn.add(tf.keras.layers.Dropout(0.3))
model_nn.add(tf.keras.layers.Dense(units=128, activation='relu'))
model_nn.add(tf.keras.layers.Dropout(0.3))
model_nn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compile the model
model_nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model_nn.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# Evaluate Neural Network model
evaluate_classification(model_nn, "Neural Network", x_train, x_test, y_train, y_test)
joblib.dump(model_nn, 'model_nn.pkl')

# Save the training history
history_df = pd.DataFrame(history.history)
history_df.to_csv('nn_training_history.csv', index=False)

# Plot training history
plt.figure(figsize=(14, 7))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Neural Network Training History')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

# Show model evaluations
results_df = pd.DataFrame.from_dict(kernal_evals, orient='index', columns=['Train Acc.', 'Test Acc.', 'Train Prec.', 'Test Prec.', 'Train Rec.', 'Test Rec.'])
results_df.to_csv('model_evaluations.csv', index=True)
