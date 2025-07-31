import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Load dataset
@st.cache
def load_data():
    # Set the correct path to your dataset
    file_path = '/content/drive/MyDrive/kddcup.data_10_percent_corrected'

    # Define column names from KDD documentation
    columns = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent",
               "hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root",
               "num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login",
               "is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
               "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
               "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
               "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
               "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]
    
    df = pd.read_csv(file_path, names=columns)
    df['label'] = df['label'].apply(lambda x: 'normal' if x == 'normal.' else 'attack')

    # Preprocess Data
    cat_cols = ['protocol_type', 'service', 'flag']
    df[cat_cols] = df[cat_cols].apply(LabelEncoder().fit_transform)
    df.drop(['num_outbound_cmds'], axis=1, inplace=True)

    X = df.drop('label', axis=1)
    y = df['label']
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

# Build Autoencoder Model
def build_autoencoder(X_train):
    input_dim = X_train.shape[1]
    encoding_dim = 20

    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation="relu")(input_layer)
    decoded = Dense(input_dim, activation="sigmoid")(encoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder

# Train Autoencoder and predict anomalies
def train_autoencoder(X_train, X_test):
    autoencoder = build_autoencoder(X_train)
    autoencoder.fit(X_train, X_train, epochs=10, batch_size=256, shuffle=True, validation_split=0.2, verbose=1)
    reconstructions = autoencoder.predict(X_test)
    mse = np.mean(np.square(X_test - reconstructions), axis=1)
    threshold = np.percentile(mse, 95)
    y_pred_ae = (mse > threshold).astype(int)  # 1 = anomaly
    return mse, y_pred_ae, threshold

# Train Isolation Forest and predict anomalies
def train_isolation_forest(X_scaled):
    iso = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    y_pred_if = iso.fit_predict(X_scaled)
    scores_if = iso.decision_function(X_scaled)
    mse_if = -scores_if
    threshold_if = np.percentile(mse_if, 95)
    return mse_if, y_pred_if, threshold_if

# Streamlit App UI
st.title("Network Traffic Anomaly Detection")
st.write("Detect unusual patterns or anomalies in network traffic data that could indicate security breaches or system malfunctions.")

# Load data
X_scaled, y = load_data()

# Display data summary
if st.checkbox("Show data summary"):
    st.write(pd.DataFrame(X_scaled).describe())

# Train Autoencoder and Isolation Forest models
if st.button("Train and Detect Anomalies"):
    # Autoencoder Anomaly Detection
    mse_ae, y_pred_ae, threshold_ae = train_autoencoder(X_scaled[y == 'normal'], X_scaled)
    st.write("### Autoencoder Anomaly Detection")
    st.write("Anomalies detected by Autoencoder:")
    st.write(confusion_matrix(y, y_pred_ae))
    st.write(classification_report(y, y_pred_ae, target_names=["Normal", "Attack"]))
    
    # Plot Autoencoder Reconstruction Error
    st.subheader("Autoencoder Reconstruction Error")
    fig_ae, ax_ae = plt.subplots(figsize=(10, 6))
    sns.histplot(mse_ae[y == 'normal'], label='Normal', color='blue', bins=100, ax=ax_ae)
    sns.histplot(mse_ae[y == 'attack'], label='Attack', color='red', bins=100, ax=ax_ae)
    ax_ae.axvline(threshold_ae, color='black', linestyle='--', label='Threshold')
    ax_ae.set_title("Reconstruction Error (Autoencoder)")
    ax_ae.set_xlabel("Reconstruction Error")
    ax_ae.set_ylabel("Count")
    ax_ae.legend()
    st.pyplot(fig_ae)
    
    # Isolation Forest Anomaly Detection
    mse_if, y_pred_if, threshold_if = train_isolation_forest(X_scaled)
    st.write("### Isolation Forest Anomaly Detection")
    st.write("Anomalies detected by Isolation Forest:")
    st.write(confusion_matrix(y, y_pred_if))
    st.write(classification_report(y, y_pred_if, target_names=["Normal", "Attack"]))
    
    # Plot Isolation Forest Anomaly Scores
    st.subheader("Isolation Forest Anomaly Scores")
    fig_if, ax_if = plt.subplots(figsize=(10, 6))
    sns.histplot(mse_if[y == 'normal'], label='Normal', color='blue', bins=100, ax=ax_if)
    sns.histplot(mse_if[y == 'attack'], label='Attack', color='red', bins=100, ax=ax_if)
    ax_if.axvline(threshold_if, color='black', linestyle='--', label='Threshold')
    ax_if.set_title("Anomaly Scores (Isolation Forest)")
    ax_if.set_xlabel("Anomaly Score")
    ax_if.set_ylabel("Count")
    ax_if.legend()
    st.pyplot(fig_if)

# Save the model for later use
if st.button("Save Model"):
    joblib.dump(autoencoder, 'autoencoder_model.pkl')
    joblib.dump(iso, 'isolation_forest_model.pkl')
    st.write("Models have been saved.")
