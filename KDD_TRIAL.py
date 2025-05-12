import pandas as pd
import numpy as np
import joblib
import pickle
from tensorflow import keras
from sklearn import metrics
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from fpdf import FPDF
# Load the dataset
df_test = pd.read_csv("KDDTest.txt")
columns = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot',
           'num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations',
           'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count',
           'serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
           'srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
           'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
           'dst_host_rerror_rate','dst_host_srv_rerror_rate','outcome','level']
df_test.columns = columns

# Map protocol types
protocol_map = {'tcp': 1, 'udp': 2, 'icmp': 3}
df_test['protocol_type'] = df_test['protocol_type'].map(protocol_map)

# Convert 'outcome' to binary values
df_test.loc[df_test['outcome'] == "normal", "outcome"] = 0
df_test.loc[df_test['outcome'] != 0, "outcome"] = 1

# Pie plot function
def pie_plot(df, cols_list, rows, cols):
    fig, axes = plt.subplots(rows, cols)
    for ax, col in zip(axes.ravel(), cols_list):
        df[col].value_counts().plot(ax=ax, kind='pie', figsize=(15, 15), fontsize=10, autopct='%1.0f%%')
        ax.set_title(str(col), fontsize=12)
    plt.show()

# Plot pie charts for 'protocol_type' and 'outcome'
pie_plot(df_test, ['protocol_type', 'outcome'], 1, 2)

# Data Preprocessing
def preprocess(dataframe, train_columns=None):
    cat_cols = ['is_host_login', 'protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_guest_login', 'level', 'outcome']
    df_num = dataframe.drop(cat_cols, axis=1)
    num_cols = df_num.columns

    # Scale numeric columns
    scaler = RobustScaler()
    scaled_df = scaler.fit_transform(df_num)
    scaled_df = pd.DataFrame(scaled_df, columns=num_cols)

    dataframe.drop(labels=num_cols, axis="columns", inplace=True)
    dataframe[num_cols] = scaled_df

    dataframe = pd.get_dummies(dataframe, columns=['protocol_type', 'service', 'flag'])

    if train_columns is not None:
        missing_cols = set(train_columns) - set(dataframe.columns)
        for col in missing_cols:
            dataframe[col] = 0
        dataframe = dataframe[train_columns]

    return dataframe

# Load models and columns
model_lr = joblib.load('model_lr.pkl')
train_columns = joblib.load('train_columns.pkl')

# Preprocess the test data
scaled_test = preprocess(df_test, train_columns)

# Ensure test columns are aligned with training columns
missing_in_test = set(train_columns) - set(scaled_test.columns)
extra_in_test = set(scaled_test.columns) - set(train_columns)

for col in missing_in_test:
    scaled_test[col] = 0

scaled_test = scaled_test.drop(columns=extra_in_test, errors='ignore')
scaled_test = scaled_test[train_columns]

# Prepare train and test data
x = scaled_test.drop(['outcome', 'level'], axis=1).values
y = scaled_test['outcome'].astype('int').values
y_reg = scaled_test['level'].values

pca = PCA(n_components=20).fit(x)
x_reduced = pca.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Define a function to evaluate classification models
kernal_evals = dict()

def evaluate_classification(model, name, X_train, X_test, y_train, y_test):
    train_predictions = (model.predict(X_train) > 0.5).astype(int)
    test_predictions = (model.predict(X_test) > 0.5).astype(int)

    train_accuracy = metrics.accuracy_score(y_train, train_predictions)
    test_accuracy = metrics.accuracy_score(y_test, test_predictions)

    train_precision = metrics.precision_score(y_train, train_predictions)
    test_precision = metrics.precision_score(y_test, test_predictions)

    train_recall = metrics.recall_score(y_train, train_predictions)
    test_recall = metrics.recall_score(y_test, test_predictions)

    kernal_evals[str(name)] = [train_accuracy, test_accuracy, train_precision, test_precision, train_recall, test_recall]

    print(f"Training Accuracy {name}: {train_accuracy * 100:.2f}%  Test Accuracy {name}: {test_accuracy * 100:.2f}%")
    print(f"Training Precision {name}: {train_precision * 100:.2f}%  Test Precision {name}: {test_precision * 100:.2f}%")
    print(f"Training Recall {name}: {train_recall * 100:.2f}%  Test Recall {name}: {test_recall * 100:.2f}%")

    confusion_matrix = metrics.confusion_matrix(y_test, test_predictions)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=['normal', 'attack'])

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.grid(False)
    cm_display.plot(ax=ax)
    plt.title(f'Confusion Matrix for {name}')
    plt.show()

# Evaluate models
model_knn = joblib.load('model_knn.pkl')
model_gnb = joblib.load('model_gnb.pkl')
model_tdt = joblib.load('model_tdt.pkl')
model_rf = joblib.load('model_rf.pkl')
model_nn = joblib.load('model_history.pkl')
model_xg_r = joblib.load('model_xg_r.pkl')

evaluate_classification(model_lr, "Logistic Regression", x_train, x_test, y_train, y_test)
#evaluate_classification(model_nn, "Neural Network", x_train, x_test, y_train, y_test)
evaluate_classification(model_knn, "KNN", x_train, x_test, y_train, y_test)
evaluate_classification(model_gnb, "GaussianNB", x_train, x_test, y_train, y_test)
evaluate_classification(model_tdt, "Decision Tree", x_train, x_test, y_train, y_test)
evaluate_classification(model_rf, "Random Forest", x_train, x_test, y_train, y_test)
evaluate_classification(model_xg_r, "XGBoost", x_train, x_test, y_train, y_test)

# Plot bar charts for evaluation metrics
def plot_metrics(kernal_evals, metric_index, metric_name):
    keys = [key for key in kernal_evals.keys()]
    values = [value for value in kernal_evals.values()]
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.bar(np.arange(len(keys)) - 0.2, [value[metric_index] for value in values], color='darkred', width=0.25, align='center')
    ax.bar(np.arange(len(keys)) + 0.2, [value[metric_index+1] for value in values], color='y', width=0.25, align='center')
    ax.legend([f"Training {metric_name}", f"Test {metric_name}"])
    ax.set_xticklabels(keys, rotation=45)
    ax.set_xticks(np.arange(len(keys)))
    plt.ylabel(metric_name)
    plt.title(f'Training vs Test {metric_name}')
    plt.show()

plot_metrics(kernal_evals, 0, "Accuracy")
plot_metrics(kernal_evals, 2, "Precision")
plot_metrics(kernal_evals, 4, "Recall")

# Create PDF with FPDF
pdf = FPDF()

pdf.add_page()
pdf.set_font('Arial', 'B', 16)
pdf.cell(200, 10, txt="Model Evaluation Results", ln=True, align='C')

pdf.set_font('Arial', '', 12)
for model_name, metrics in kernal_evals.items():
    pdf.cell(200, 10, txt=f"{model_name}:", ln=True)
    pdf.cell(200, 10, txt=f"  Training Accuracy: {metrics[0]*100:.2f}% - Test Accuracy: {metrics[1]*100:.2f}%", ln=True)
    pdf.cell(200, 10, txt=f"  Training Precision: {metrics[2]*100:.2f}% - Test Precision: {metrics[3]*100:.2f}%", ln=True)
    pdf.cell(200, 10, txt=f"  Training Recall: {metrics[4]*100:.2f}% - Test Recall: {metrics[5]*100:.2f}%", ln=True)

pdf.output("evaluation_results.pdf")

print("Evaluation results PDF generated successfully.")
