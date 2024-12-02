# Name: Dylan Lazar
# Hopkins ID: AB1041
# Final Capstone Project Assignment Submission
# EN.585.771 - Biomedical Datascience

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns

data = pd.read_csv("heart.csv")

# Features and Target
X = data.drop('target', axis=1) # ALL Columns except 'Target'
y = data['target'] # target column

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_scaled, y, test_size=0.2, random_state=42) 

# Train Logistic Regression Model
model = LogisticRegression(class_weight="balanced", max_iter=500)
model.fit(Xtrain, Ytrain)

# Predictions
y_pred = model.predict(Xtest)
y_prob = model.predict_proba(Xtest)[:,1]

# Evaluate model
accuracy = accuracy_score(Ytest, y_pred)
roc_auc = roc_auc_score(Ytest, y_prob)

# Calculate ROC Curve
fpr, tpr, thresholds = roc_curve(Ytest, y_prob)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(Xtrain, dtype=torch.float32)
X_test_tensor = torch.tensor(Xtest, dtype=torch.float32)
y_train_tensor = torch.tensor(Ytrain.values, dtype=torch.float32)
y_test_tensor = torch.tensor(Ytest.values, dtype=torch.float32)

# Create PyTorch datasets and data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Define the neural network with dropout layers
class NeuralNetworkWithDropout(nn.Module):
    def __init__(self):
        super(NeuralNetworkWithDropout, self).__init__()
        self.fc1 = nn.Linear(X.shape[1], 16)  # First hidden layer
        self.dropout1 = nn.Dropout(0.3)       # Dropout after first layer
        self.fc2 = nn.Linear(16, 8)           # Second hidden layer
        self.dropout2 = nn.Dropout(0.3)       # Dropout after second layer
        self.output = nn.Linear(8, 1)         # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.sigmoid(self.output(x))     # Sigmoid for binary classification
        return x

# Instantiate the model
model = NeuralNetworkWithDropout()

# Define optimizer, loss function, and learning rate scheduler
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
loss_fn = nn.BCELoss()

# Train the model
epochs = 50
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for batch in train_loader:
        X_batch, y_batch = batch
        optimizer.zero_grad()
        predictions = model(X_batch).squeeze()
        loss = loss_fn(predictions, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    scheduler.step()  # Update the learning rate
    #print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

# Evaluate the model
model.eval()
y_pred_nn = []
y_true_nn = []
with torch.no_grad():
    for batch in test_loader:
        X_batch, y_batch = batch
        y_pred_batch = model(X_batch).squeeze()
        y_pred_nn.extend(y_pred_batch.numpy())
        y_true_nn.extend(y_batch.numpy())

# Calculate AUC and ROC curve for the neural network
threshold = 0.5
y_pred_labels_nn = [1 if pred >= threshold else 0 for pred in y_pred_nn]
correct_predictions_nn = sum([1 for true, pred in zip(y_true_nn, y_pred_labels_nn) if true == pred])
accuracy_nn = correct_predictions_nn / len(y_true_nn)
roc_auc_nn = roc_auc_score(y_true_nn, y_pred_nn)
fpr_nn, tpr_nn, _ = roc_curve(y_true_nn, y_pred_nn)

# Streamlit setup
st.set_page_config(layout="wide")
# Inject custom CSS for sidebar font size
st.markdown(
    """
    <style>
    /* Increase font size in the sidebar */
    .css-1d391kg, .css-1v3fvcr {
        font-size: 18px !important;
    }

    /* Optional: Increase font size of headers in the sidebar */
    .css-145kmo2 {
        font-size: 20px !important;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Heart Disease Prediction")

# Page navigation
# Sidebar with radio buttons
page = st.sidebar.radio(
    "Navigation",
    ["Data Exploration", "Logistic Regression", "Neural Network"],
)

if page == "Data Exploration":
    st.header("Heart Disease Dataset Overview")
    
    # Feature Explanations
    with st.expander("Click to view feature explanations"):
        st.write("""
        - ```age```: Age of the individual.  
        - ```sex```: Sex of the individual (1 = male; 0 = female).  
        - ```cp```: Type of chest pain experienced by the individual (4 values: 0-3).  
        - ```trestbps```: Resting blood pressure in mm Hg.  
        - ```chol```: Serum cholesterol in mg/dl.  
        - ```fbs```: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false).  
        - ```restecg```: Resting ECG results (0, 1, 2).  
        - ```thalach```: Maximum heart rate achieved during exercise.  
        - ```exang```: Presence of exercise-induced angina (1 = yes; 0 = no).  
        - ```oldpeak```: ST depression induced by exercise relative to rest.  
        - ```slope```: The slope of the peak exercise ST segment.  
        - ```ca```: Number of major vessels (0-3) colored by fluoroscopy.  
        - ```thal```: Type of thalassemia (0 = normal; 1 = fixed defect; 2 = reversible defect).  
        """)

    # Display Table
    st.subheader("Dataset Table")
    st.dataframe(data, width=800, height=400, use_container_width=True)
    
    st.subheader("Feature Distributions")
    selected_feature = st.selectbox("Select feature to visualize", data.columns[:-1])
    sns.histplot(data[selected_feature], kde=True)
    # Constrain figure to a column
    col1, col2, col3 = st.columns([1, 2, 1])  # Adjust proportions
    with col2:
        st.pyplot(plt)

elif page == "Logistic Regression":
    st.header("Logistic Regression Performance")
    st.subheader("Model Evaluation")

    # Description of Model
    st.write("The logistic regression model implemented is a binary classification algorithm used to predict the likelihood of heart disease based on features from the dataset. It uses the sigmoid function to estimate probabilities, mapping input features such as age, cholesterol levels, and maximum heart rate to a value between 0 and 1, where the output represents the probability of heart disease presence. The model is trained using the Scikit-learn library, optimizing the log-loss function with gradient descent to minimize classification errors. Key metrics, including accuracy and the area under the ROC curve (AUC), are used to evaluate the model's predictive performance, with an observed AUC of approximately 0.88 indicating good but improvable discrimination ability. The app visualizes the ROC curve to help interpret the model's trade-offs between sensitivity and specificity in diagnosing heart disease.")
    
    # Display metrics
    st.write(f"**Accuracy**: {accuracy:.2f}")
    st.write(f"**ROC AUC**: {roc_auc:.2f}")
    st.write("Classification Report:")
    classificationReport = pd.DataFrame(classification_report(Ytest, y_pred, output_dict=True)).transpose()
    st.dataframe(classificationReport, width=800, height=400, use_container_width=True)
    
    # Display ROC Curve
    st.subheader("ROC Curve")
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'r--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    # Constrain figure to a column
    col4, col5, col6 = st.columns([1, 2, 1])  # Adjust proportions
    with col5:
        st.pyplot(fig)

    # Prepare Comparison Data
    st.subheader("Logistic Regression vs Neural Network:")
    comparison_data = {
        "Metric": ["Accuracy", "ROC AUC"],
        "Logistic Regression": [accuracy, roc_auc],
        "Neural Network": [accuracy_nn, roc_auc_nn]
    }
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, width=800, height=200, use_container_width=True)

    # Create a bar plot with separate bars for each model and metric
    fig_bar, ax_bar = plt.subplots(figsize=(10, 6))

    # Bar positions
    x = np.arange(len(comparison_df["Metric"]))  # Number of metrics
    width = 0.35  # Bar width
    bar1 = ax_bar.bar(x - width / 2, comparison_df["Logistic Regression"], width, label="Logistic Regression", color="blue")
    bar2 = ax_bar.bar(x + width / 2, comparison_df["Neural Network"], width, label="Neural Network", color="green", alpha=0.7)
    ax_bar.set_xlabel("Metric")
    ax_bar.set_ylabel("Score")
    ax_bar.set_title("Comparison of Model Metrics")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(comparison_df["Metric"])
    ax_bar.legend()
    plt.tight_layout()
    col7, col8, col9 = st.columns([1, 2, 1])  # Adjust proportions
    with col8:
        st.pyplot(fig_bar)

elif page == "Neural Network":
    st.header("Neural Network Performance")

    st.subheader("Model Evaluation")

    # Description of Neural Network
    st.write("The neural network implemented is a binary classification model built using PyTorch to predict heart disease based on dataset features. The network architecture consists of two fully connected hidden layers with ReLU activation, a dropout layer (with a rate of 0.3) to reduce overfitting, and a sigmoid output layer for binary probability estimation. Features like age, cholesterol levels, and exercise-induced angina are used as inputs, with the network trained using the binary cross-entropy loss function and the Adam optimizer. The model achieves an AUC of approximately 0.94, better than the logistic regression, with enhanced generalization due to the inclusion of dropout. ROC curve visualization and metric comparisons in the app provide insights into the neural network's predictive performance relative to logistic regression.")
    
    # Display metrics
    st.write(f"**Accuracy**: {accuracy_nn:.2f}")
    st.write(f"**ROC AUC**: {roc_auc_nn:.2f}")

    # ROC Curve visualization
    st.subheader("ROC Curve")
    fpr_nn, tpr_nn, _ = roc_curve(y_true_nn, y_pred_nn)
    fpr_lr, tpr_lr, _ = roc_curve(Ytest, y_pred)

    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    ax_roc.plot(fpr_nn, tpr_nn, label=f"Neural Network (AUC = {roc_auc_nn:.2f})", color="green")
    ax_roc.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc:.2f})", color="blue")
    ax_roc.plot([0, 1], [0, 1], 'r--')
    ax_roc.set_title("ROC Curve")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend()
    ax_roc.grid(alpha=0.3)
    col10, col11, col12 = st.columns([1, 2, 1])  # Adjust proportions
    with col11:
        st.pyplot(fig_roc)

    # Prepare Comparison Data
    st.subheader("Neural Network vs Logistic Regression:")
    comparison_data = {
        "Metric": ["Accuracy", "ROC AUC"],
        "Logistic Regression": [accuracy, roc_auc],
        "Neural Network": [accuracy_nn, roc_auc_nn]
    }
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, width=800, height=200, use_container_width=True)

    # Create a bar plot with separate bars for each model and metric
    fig_bar, ax_bar = plt.subplots(figsize=(10, 6))

    # Bar positions
    x = np.arange(len(comparison_df["Metric"]))  # Number of metrics
    width = 0.35  # Bar width
    bar1 = ax_bar.bar(x - width / 2, comparison_df["Logistic Regression"], width, label="Logistic Regression", color="blue")
    bar2 = ax_bar.bar(x + width / 2, comparison_df["Neural Network"], width, label="Neural Network", color="green", alpha=0.7)
    ax_bar.set_xlabel("Metric")
    ax_bar.set_ylabel("Score")
    ax_bar.set_title("Comparison of Model Metrics")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(comparison_df["Metric"])
    ax_bar.legend()
    plt.tight_layout()
    col13, col14, col15 = st.columns([1, 2, 1])  # Adjust proportions
    with col14:
        st.pyplot(fig_bar)