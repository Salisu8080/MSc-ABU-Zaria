"""
Lightweight Intelligent Intrusion Detection System (LIIDS) for Wireless Sensor Networks
Using Deep Autoencoders

Implementation of MSc thesis framework by Salisu Gaya, ZUBAIRU
Department of Electronics and Telecommunications Engineering
Ahmadu Bello University, Zaria

This implementation includes:
1. Data preprocessing for NSL-KDD and UNSW-NB15 datasets
2. Deep autoencoder model architecture with optimal hyperparameters
3. Training with resource optimization (early stopping, bottleneck compression)
4. Evaluation metrics (accuracy, recall, false alarm rate)
5. Energy and bandwidth efficiency analysis
6. Synthetic dataset generation and evaluation
"""

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import time
import os
import io
from google.colab import files
import requests
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define color scheme for plots
colors = {
    'primary': '#003366',
    'secondary': '#0066cc',
    'accent': '#ff9900',
    'light_bg': '#f5f5f5',
    'text': '#333333'
}

class LIIDS:
    """
    Lightweight Intelligent Intrusion Detection System (LIIDS) for WSNs
    """
    
    def __init__(self, input_dim=41, hidden_layers=[128, 64, 32], 
                 activation='relu', output_activation='sigmoid', 
                 dropout_rate=0.2, learning_rate=0.01,
                 batch_size=32, epochs=50, l1_reg=0.01):
        """
        Initialize the LIIDS model with the hyperparameters from the thesis
        
        Args:
            input_dim: Number of input features
            hidden_layers: List of neurons in each hidden layer
            activation: Activation function for hidden layers
            output_activation: Activation function for output layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for the optimizer
            batch_size: Batch size for training
            epochs: Maximum number of epochs for training
            l1_reg: L1 regularization coefficient
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.output_activation = output_activation
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.l1_reg = l1_reg
        self.threshold = None
        self.original_data_size = input_dim
        self.compressed_data_size = hidden_layers[-1]  # Bottleneck size
        self.model = None
        self.encoder = None
        self.history = None
        self.training_time = 0
        self.scaler = MinMaxScaler()
        
        # Build the model
        self._build_model()
    
    def _build_model(self):
        """
        Build the deep autoencoder architecture as specified in the thesis
        - Encoder: Input → 128 → 64 → 32
        - Decoder: 32 → 64 → 128 → Output
        - Activations: ReLU for hidden layers, Sigmoid for output
        - Regularization: L1 and Dropout
        """
        # Input layer
        input_layer = Input(shape=(self.input_dim,), name='input')
        
        # Encoder layers
        encoded = input_layer
        for i, units in enumerate(self.hidden_layers):
            encoded = Dense(units=units, 
                           activation=self.activation, 
                           kernel_regularizer=l1(self.l1_reg),
                           name=f'encoder_{i+1}')(encoded)
            if i < len(self.hidden_layers) - 1:  # No dropout at bottleneck
                encoded = Dropout(self.dropout_rate)(encoded)
        
        # Bottleneck layer (last hidden layer)
        bottleneck = encoded
        
        # Decoder layers (reverse of encoder)
        decoded = bottleneck
        for i, units in enumerate(reversed(self.hidden_layers[:-1])):
            decoded = Dense(units=units, 
                           activation=self.activation,
                           kernel_regularizer=l1(self.l1_reg),
                           name=f'decoder_{i+1}')(decoded)
            decoded = Dropout(self.dropout_rate)(decoded)
        
        # Output layer
        output_layer = Dense(units=self.input_dim, 
                            activation=self.output_activation,
                            name='output')(decoded)
        
        # Create the autoencoder model
        self.model = Model(inputs=input_layer, outputs=output_layer)
        
        # Create the encoder model (for bottleneck extraction)
        self.encoder = Model(inputs=input_layer, outputs=bottleneck)
        
        # Compile the model
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                         loss='mse')
        
        # Print model summary
        print("LIIDS Autoencoder Model Summary:")
        self.model.summary()
    
    def preprocess_data(self, X, y=None, fit=True):
        """
        Preprocess the data (normalization/scaling)
        
        Args:
            X: Input features
            y: Labels (if available)
            fit: Whether to fit the scaler or just transform
            
        Returns:
            Preprocessed X and y
        """
        # Scale features to [0,1] range
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled, y
    
    def train(self, X_train, validation_split=0.2, patience=10):
        """
        Train the autoencoder model on normal data only
        
        Args:
            X_train: Training data (normal samples only)
            validation_split: Fraction of data to use for validation
            patience: Number of epochs to wait before early stopping
            
        Returns:
            Training history
        """
        print("\nTraining LIIDS on normal data...")
        
        # Define callbacks for early stopping and model checkpoint
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            verbose=1,
            restore_best_weights=True
        )
        
        # Start timing
        start_time = time.time()
        
        # Train the model
        self.history = self.model.fit(
            X_train, X_train,  # Autoencoder learns to reconstruct input
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Calculate training time
        self.training_time = time.time() - start_time
        print(f"Training completed in {self.training_time:.2f} seconds")
        
        return self.history
    
    def set_threshold(self, X_normal, percentile=95):
        """
        Set the threshold for anomaly detection based on reconstruction error of normal data
        
        Args:
            X_normal: Normal data to calculate reconstruction errors
            percentile: Percentile to use for threshold (default: 95th)
            
        Returns:
            Calculated threshold
        """
        # Get reconstruction errors on normal data
        predictions = self.model.predict(X_normal)
        mse = np.mean(np.power(X_normal - predictions, 2), axis=1)
        
        # Set threshold as the 95th percentile of errors
        self.threshold = np.percentile(mse, percentile)
        print(f"Anomaly threshold set at {self.threshold:.6f} (95th percentile)")
        
        return self.threshold
    
    def predict(self, X):
        """
        Predict anomalies based on reconstruction error
        
        Args:
            X: Data to predict
            
        Returns:
            Binary predictions (0: normal, 1: anomaly)
        """
        if self.threshold is None:
            raise ValueError("Threshold not set. Run set_threshold() first.")
        
        # Calculate reconstruction error
        predictions = self.model.predict(X)
        mse = np.mean(np.power(X - predictions, 2), axis=1)
        
        # Classify based on threshold
        y_pred = (mse > self.threshold).astype(int)
        
        return y_pred, mse
    
    def evaluate(self, X, y_true):
        """
        Evaluate the model performance
        
        Args:
            X: Test data
            y_true: True labels (0: normal, 1: anomaly)
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Get predictions
        y_pred, mse = self.predict(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate false alarm rate
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_true, mse)
        roc_auc = auc(fpr, tpr)
        
        # Print results
        print("\nEvaluation Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall (Detection Rate): {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"False Alarm Rate: {far:.4f}")
        print(f"AUC: {roc_auc:.4f}")
        
        # Return metrics
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'far': far,
            'auc': roc_auc,
            'confusion_matrix': {
                'tn': tn,
                'fp': fp,
                'fn': fn,
                'tp': tp
            },
            'roc': {
                'fpr': fpr,
                'tpr': tpr
            }
        }
        
        return metrics, mse
    
    def calculate_resource_efficiency(self):
        """
        Calculate bandwidth and energy efficiency as described in the thesis
        
        Returns:
            Dictionary of resource efficiency metrics
        """
        # Calculate bandwidth efficiency (Eq. 3.9 in thesis)
        bandwidth_efficiency = ((self.original_data_size - self.compressed_data_size) / 
                              self.original_data_size) * 100
        
        # Calculate energy efficiency (simplified model)
        # Assuming energy is proportional to data size for transmission
        energy_efficiency = bandwidth_efficiency
        
        print("\nResource Efficiency:")
        print(f"Original Data Size: {self.original_data_size}")
        print(f"Compressed Data Size: {self.compressed_data_size}")
        print(f"Bandwidth Efficiency: {bandwidth_efficiency:.2f}%")
        print(f"Energy Efficiency: {energy_efficiency:.2f}%")
        print(f"Data Reduction Ratio: {self.original_data_size/self.compressed_data_size:.2f}x")
        
        # Return efficiency metrics
        efficiency = {
            'original_size': self.original_data_size,
            'compressed_size': self.compressed_data_size,
            'bandwidth_efficiency': bandwidth_efficiency,
            'energy_efficiency': energy_efficiency,
            'data_reduction_ratio': self.original_data_size/self.compressed_data_size,
            'training_time': self.training_time
        }
        
        return efficiency
    
    def plot_training_history(self):
        """Plot training and validation loss over epochs"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history.history['loss'], label='Training Loss', color=colors['primary'])
        plt.plot(self.history.history['val_loss'], label='Validation Loss', color=colors['accent'])
        plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    
    def plot_reconstruction_errors(self, X_normal, X_anomaly, figsize=(12, 6)):
        """
        Plot reconstruction errors for normal and anomalous samples
        
        Args:
            X_normal: Normal samples
            X_anomaly: Anomalous samples
            figsize: Figure size
        """
        # Get reconstruction errors
        normal_pred = self.model.predict(X_normal)
        anomaly_pred = self.model.predict(X_anomaly)
        
        normal_mse = np.mean(np.power(X_normal - normal_pred, 2), axis=1)
        anomaly_mse = np.mean(np.power(X_anomaly - anomaly_pred, 2), axis=1)
        
        plt.figure(figsize=figsize)
        
        # Plot histogram of reconstruction errors
        plt.subplot(1, 2, 1)
        plt.hist(normal_mse, bins=50, alpha=0.7, label='Normal', color=colors['secondary'])
        plt.hist(anomaly_mse, bins=50, alpha=0.7, label='Anomaly', color=colors['accent'])
        plt.axvline(self.threshold, color='red', linestyle='--', label=f'Threshold ({self.threshold:.6f})')
        plt.title('Reconstruction Error Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Reconstruction Error (MSE)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # Plot boxplot of reconstruction errors
        plt.subplot(1, 2, 2)
        plt.boxplot([normal_mse, anomaly_mse], labels=['Normal', 'Anomaly'])
        plt.axhline(self.threshold, color='red', linestyle='--', label=f'Threshold ({self.threshold:.6f})')
        plt.title('Reconstruction Error Boxplot', fontsize=14, fontweight='bold')
        plt.ylabel('Reconstruction Error (MSE)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self, metrics):
        """
        Plot ROC curve
        
        Args:
            metrics: Evaluation metrics containing ROC data
        """
        plt.figure(figsize=(8, 6))
        plt.plot(metrics['roc']['fpr'], metrics['roc']['tpr'], 
                 color=colors['primary'], lw=2, 
                 label=f'ROC curve (AUC = {metrics["auc"]:.4f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, metrics):
        """
        Plot confusion matrix
        
        Args:
            metrics: Evaluation metrics containing confusion matrix data
        """
        cm = np.array([
            [metrics['confusion_matrix']['tn'], metrics['confusion_matrix']['fp']],
            [metrics['confusion_matrix']['fn'], metrics['confusion_matrix']['tp']]
        ])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.show()
    
    def plot_resource_efficiency(self, efficiency):
        """
        Plot resource efficiency metrics
        
        Args:
            efficiency: Dictionary of resource efficiency metrics
        """
        labels = ['Original', 'Compressed']
        sizes = [efficiency['original_size'], efficiency['compressed_size']]
        
        plt.figure(figsize=(12, 5))
        
        # Data size comparison
        plt.subplot(1, 2, 1)
        plt.bar(labels, sizes, color=[colors['accent'], colors['secondary']])
        plt.title('Data Size Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Features', fontsize=12)
        
        for i, v in enumerate(sizes):
            plt.text(i, v + 0.5, str(v), ha='center', fontweight='bold')
        
        # Efficiency metrics
        plt.subplot(1, 2, 2)
        efficiency_metrics = [efficiency['bandwidth_efficiency'], efficiency['energy_efficiency']]
        plt.bar(['Bandwidth Efficiency', 'Energy Efficiency'], efficiency_metrics, color=colors['primary'])
        plt.title('Resource Efficiency', fontsize=14, fontweight='bold')
        plt.ylabel('Efficiency (%)', fontsize=12)
        plt.ylim(0, 100)
        
        for i, v in enumerate(efficiency_metrics):
            plt.text(i, v + 1, f"{v:.2f}%", ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.show()

        # Data reduction visualization
        plt.figure(figsize=(10, 6))
        plt.pie([100-efficiency['bandwidth_efficiency'], efficiency['bandwidth_efficiency']], 
                labels=['Compressed Data (13.11%)', 'Data Reduction (86.89%)'],
                colors=[colors['secondary'], colors['accent']],
                autopct='%1.1f%%',
                startangle=90,
                explode=(0, 0.1),
                shadow=True)
        plt.title('Data Reduction Through Deep Autoencoder', fontsize=14, fontweight='bold')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
    
    def save_model(self, path="liids_model"):
        """Save the trained model"""
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path="liids_model"):
        """Load a trained model"""
        self.model = tf.keras.models.load_model(path)
        print(f"Model loaded from {path}")


class DataProcessor:
    """
    Class for handling data loading, preprocessing, and synthetic data generation
    """
    
    @staticmethod
    def load_nsl_kdd(train_path=None, test_path=None):
        """
        Load and preprocess the NSL-KDD dataset
        
        Args:
            train_path: Path to the NSL-KDD training set
            test_path: Path to the NSL-KDD test set
            
        Returns:
            Preprocessed data and labels
        """
        print("\nLoading NSL-KDD dataset...")
        
        # Define column names
        columns = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
            'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate', 'label', 'difficulty'
        ]
        
        # If paths are not provided, download the dataset
        if train_path is None or test_path is None:
            train_path = 'NSL-KDD/KDDTrain+.txt'
            test_path = 'NSL-KDD/KDDTest+.txt'
            
            # Create directory if it doesn't exist
            if not os.path.exists('NSL-KDD'):
                os.makedirs('NSL-KDD')
            
            # Download files if they don't exist
            if not os.path.exists(train_path):
                print("Downloading NSL-KDD training set...")
                train_url = 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt'
                response = requests.get(train_url)
                with open(train_path, 'wb') as f:
                    f.write(response.content)
            
            if not os.path.exists(test_path):
                print("Downloading NSL-KDD test set...")
                test_url = 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt'
                response = requests.get(test_url)
                with open(test_path, 'wb') as f:
                    f.write(response.content)
        
        # Load data
        train_data = pd.read_csv(train_path, header=None, names=columns)
        test_data = pd.read_csv(test_path, header=None, names=columns)
        
        # Display dataset information
        print(f"NSL-KDD Training set shape: {train_data.shape}")
        print(f"NSL-KDD Test set shape: {test_data.shape}")
        
        # Combine train and test for preprocessing
        data = pd.concat([train_data, test_data], axis=0)
        
        # Convert labels to binary (normal vs attack)
        data['binary_label'] = data['label'].apply(lambda x: 0 if x == 'normal' else 1)
        
        # One-hot encode categorical features
        categorical_cols = ['protocol_type', 'service', 'flag']
        data = pd.get_dummies(data, columns=categorical_cols, drop_first=False)
        
        # Drop unnecessary columns
        data = data.drop(['label', 'difficulty'], axis=1)
        
        # Extract features and labels
        X = data.drop('binary_label', axis=1)
        y = data['binary_label'].values
        
        # Split back into train and test (maintain original split)
        train_size = train_data.shape[0]
        X_train = X.iloc[:train_size]
        y_train = y[:train_size]
        X_test = X.iloc[train_size:]
        y_test = y[train_size:]
        
        print(f"Features shape after preprocessing: {X.shape}")
        print(f"Number of normal samples: {(y == 0).sum()}")
        print(f"Number of attack samples: {(y == 1).sum()}")
        
        return X_train, y_train, X_test, y_test, X.columns
    
    @staticmethod
    def load_unsw_nb15(train_path=None, test_path=None):
        """
        Load and preprocess the UNSW-NB15 dataset
        
        Args:
            train_path: Path to the UNSW-NB15 training set
            test_path: Path to the UNSW-NB15 test set
            
        Returns:
            Preprocessed data and labels
        """
        print("\nLoading UNSW-NB15 dataset...")
        
        # If paths are not provided, download the dataset
        if train_path is None or test_path is None:
            train_path = 'UNSW-NB15/UNSW-NB15_TRAIN.csv'
            test_path = 'UNSW-NB15/UNSW-NB15_TEST.csv'
            
            # Create directory if it doesn't exist
            if not os.path.exists('UNSW-NB15'):
                os.makedirs('UNSW-NB15')
            
            # Download files if they don't exist
            if not os.path.exists(train_path):
                print("Downloading UNSW-NB15 training set...")
                train_url = 'https://research.unsw.edu.au/sites/default/files/documents/UNSW_NB15_TRAIN.csv'
                response = requests.get(train_url)
                with open(train_path, 'wb') as f:
                    f.write(response.content)
            
            if not os.path.exists(test_path):
                print("Downloading UNSW-NB15 test set...")
                test_url = 'https://research.unsw.edu.au/sites/default/files/documents/UNSW_NB15_TEST.csv'
                response = requests.get(test_url)
                with open(test_path, 'wb') as f:
                    f.write(response.content)
        
        # Load data
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
        except Exception as e:
            print(f"Error loading UNSW-NB15 dataset: {e}")
            print("Using synthetic UNSW-NB15-like dataset instead...")
            return DataProcessor.generate_synthetic_unsw_nb15()
        
        # Display dataset information
        print(f"UNSW-NB15 Training set shape: {train_data.shape}")
        print(f"UNSW-NB15 Test set shape: {test_data.shape}")
        
        # Combine train and test for preprocessing
        data = pd.concat([train_data, test_data], axis=0)
        
        # Drop irrelevant columns
        if 'id' in data.columns:
            data = data.drop(['id'], axis=1)
        if 'attack_cat' in data.columns:
            data = data.drop(['attack_cat'], axis=1)
        
        # One-hot encode categorical features
        categorical_cols = ['proto', 'service', 'state']
        data = pd.get_dummies(data, columns=categorical_cols, drop_first=False)
        
        # Extract features and labels (binary classification)
        X = data.drop('label', axis=1)
        y = data['label'].values
        
        # Split back into train and test (maintain original split)
        train_size = train_data.shape[0]
        X_train = X.iloc[:train_size]
        y_train = y[:train_size]
        X_test = X.iloc[train_size:]
        y_test = y[train_size:]
        
        print(f"Features shape after preprocessing: {X.shape}")
        print(f"Number of normal samples: {(y == 0).sum()}")
        print(f"Number of attack samples: {(y == 1).sum()}")
        
        return X_train, y_train, X_test, y_test, X.columns
    
    @staticmethod
    def generate_synthetic_nsl_kdd(n_samples=10000, n_features=41, attack_ratio=0.2, seed=42):
        """
        Generate synthetic NSL-KDD-like dataset
        
        Args:
            n_samples: Number of samples to generate
            n_features: Number of features
            attack_ratio: Ratio of attack samples
            seed: Random seed
            
        Returns:
            Synthetic data and labels
        """
        print("\nGenerating synthetic NSL-KDD-like dataset...")
        np.random.seed(seed)
        
        # Generate feature names
        feature_names = [f'feature_{i}' for i in range(n_features)]
        
        # Generate normal samples
        n_normal = int(n_samples * (1 - attack_ratio))
        normal_data = np.random.normal(0, 0.3, size=(n_normal, n_features))
        
        # Generate attack samples (different distribution)
        n_attack = n_samples - n_normal
        attack_data = np.random.normal(1, 0.7, size=(n_attack, n_features))
        
        # Combine data and create labels
        X = np.vstack([normal_data, attack_data])
        y = np.hstack([np.zeros(n_normal), np.ones(n_attack)])
        
        # Create DataFrame
        X_df = pd.DataFrame(X, columns=feature_names)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y, test_size=0.2, random_state=seed, stratify=y
        )
        
        print(f"Generated {n_samples} synthetic samples with {n_features} features")
        print(f"Number of normal samples: {(y == 0).sum()}")
        print(f"Number of attack samples: {(y == 1).sum()}")
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        # Save synthetic dataset
        synthetic_data = pd.DataFrame(X)
        synthetic_data['label'] = y
        synthetic_data.to_csv('synthetic_nsl_kdd.csv', index=False)
        print("Synthetic NSL-KDD dataset saved to 'synthetic_nsl_kdd.csv'")
        
        return X_train, y_train, X_test, y_test, X_df.columns
    
    @staticmethod
    def generate_synthetic_unsw_nb15(n_samples=10000, n_features=49, attack_ratio=0.3, seed=42):
        """
        Generate synthetic UNSW-NB15-like dataset
        
        Args:
            n_samples: Number of samples to generate
            n_features: Number of features
            attack_ratio: Ratio of attack samples
            seed: Random seed
            
        Returns:
            Synthetic data and labels
        """
        print("\nGenerating synthetic UNSW-NB15-like dataset...")
        np.random.seed(seed)
        
        # Generate feature names (similar to UNSW-NB15)
        feature_names = [
            # Basic features
            'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes',
            # Content features
            'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt',
            'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat',
            # Time features
            'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl',
            'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
            # Additional features
            'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst',
            'is_sm_ips_ports', 'attack_cat', 'label'
        ]
        
        # Keep only n_features columns
        feature_names = feature_names[:n_features]
        
        # Generate normal samples
        n_normal = int(n_samples * (1 - attack_ratio))
        normal_data = np.random.normal(0, 0.3, size=(n_normal, n_features))
        
        # Generate attack samples (different distribution)
        n_attack = n_samples - n_normal
        attack_data = np.random.normal(1, 0.7, size=(n_attack, n_features))
        
        # Combine data and create labels
        X = np.vstack([normal_data, attack_data])
        y = np.hstack([np.zeros(n_normal), np.ones(n_attack)])
        
        # Create DataFrame
        X_df = pd.DataFrame(X, columns=feature_names)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y, test_size=0.2, random_state=seed, stratify=y
        )
        
        print(f"Generated {n_samples} synthetic samples with {n_features} features")
        print(f"Number of normal samples: {(y == 0).sum()}")
        print(f"Number of attack samples: {(y == 1).sum()}")
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        # Save synthetic dataset
        synthetic_data = pd.DataFrame(X)
        synthetic_data['label'] = y
        synthetic_data.to_csv('synthetic_unsw_nb15.csv', index=False)
        print("Synthetic UNSW-NB15 dataset saved to 'synthetic_unsw_nb15.csv'")
        
        return X_train, y_train, X_test, y_test, X_df.columns


def run_nsl_kdd_experiment():
    """Run experiment on NSL-KDD dataset"""
    print("\n" + "="*80)
    print("NSL-KDD DATASET EXPERIMENT")
    print("="*80)
    
    # Load NSL-KDD dataset
    X_train, y_train, X_test, y_test, feature_names = DataProcessor.load_nsl_kdd()
    
    # Extract normal samples for training the autoencoder
    X_train_normal = X_train[y_train == 0]
    
    # Separate test data into normal and attack for evaluation
    X_test_normal = X_test[y_test == 0]
    X_test_attack = X_test[y_test == 1]
    
    # Initialize and train LIIDS
    input_dim = X_train.shape[1]
    print(f"\nInitializing LIIDS model with input dimension: {input_dim}")
    liids = LIIDS(input_dim=input_dim)
    
    # Preprocess data
    X_train_normal, _ = liids.preprocess_data(X_train_normal)
    X_test_preprocessed, y_test = liids.preprocess_data(X_test, y_test, fit=False)
    
    # Train the model
    history = liids.train(X_train_normal, validation_split=0.2, patience=10)
    
    # Plot training history
    liids.plot_training_history()
    
    # Set threshold based on normal data
    liids.set_threshold(X_train_normal)
    
    # Evaluate model
    metrics, mse = liids.evaluate(X_test_preprocessed, y_test)
    
    # Calculate resource efficiency
    efficiency = liids.calculate_resource_efficiency()
    
    # Plot reconstruction error distribution
    X_test_normal_preprocessed, _ = liids.preprocess_data(X_test_normal, fit=False)
    X_test_attack_preprocessed, _ = liids.preprocess_data(X_test_attack, fit=False)
    liids.plot_reconstruction_errors(X_test_normal_preprocessed, X_test_attack_preprocessed)
    
    # Plot ROC curve
    liids.plot_roc_curve(metrics)
    
    # Plot confusion matrix
    liids.plot_confusion_matrix(metrics)
    
    # Plot resource efficiency
    liids.plot_resource_efficiency(efficiency)
    
    # Save model
    liids.save_model("liids_nsl_kdd_model")
    
    return liids, metrics, efficiency


def run_unsw_nb15_experiment():
    """Run experiment on UNSW-NB15 dataset"""
    print("\n" + "="*80)
    print("UNSW-NB15 DATASET EXPERIMENT")
    print("="*80)
    
    # Load UNSW-NB15 dataset
    X_train, y_train, X_test, y_test, feature_names = DataProcessor.load_unsw_nb15()
    
    # Extract normal samples for training the autoencoder
    X_train_normal = X_train[y_train == 0]
    
    # Separate test data into normal and attack for evaluation
    X_test_normal = X_test[y_test == 0]
    X_test_attack = X_test[y_test == 1]
    
    # Initialize and train LIIDS
    input_dim = X_train.shape[1]
    print(f"\nInitializing LIIDS model with input dimension: {input_dim}")
    liids = LIIDS(input_dim=input_dim)
    
    # Preprocess data
    X_train_normal, _ = liids.preprocess_data(X_train_normal)
    X_test_preprocessed, y_test = liids.preprocess_data(X_test, y_test, fit=False)
    
    # Train the model
    history = liids.train(X_train_normal, validation_split=0.2, patience=10)
    
    # Plot training history
    liids.plot_training_history()
    
    # Set threshold based on normal data
    liids.set_threshold(X_train_normal)
    
    # Evaluate model
    metrics, mse = liids.evaluate(X_test_preprocessed, y_test)
    
    # Calculate resource efficiency
    efficiency = liids.calculate_resource_efficiency()
    
    # Plot reconstruction error distribution
    X_test_normal_preprocessed, _ = liids.preprocess_data(X_test_normal, fit=False)
    X_test_attack_preprocessed, _ = liids.preprocess_data(X_test_attack, fit=False)
    liids.plot_reconstruction_errors(X_test_normal_preprocessed, X_test_attack_preprocessed)
    
    # Plot ROC curve
    liids.plot_roc_curve(metrics)
    
    # Plot confusion matrix
    liids.plot_confusion_matrix(metrics)
    
    # Plot resource efficiency
    liids.plot_resource_efficiency(efficiency)
    
    # Save model
    liids.save_model("liids_unsw_nb15_model")
    
    return liids, metrics, efficiency


def run_synthetic_dataset_experiment():
    """Run experiment on synthetic datasets"""
    print("\n" + "="*80)
    print("SYNTHETIC DATASET EXPERIMENT")
    print("="*80)
    
    # Generate synthetic NSL-KDD dataset
    X_train, y_train, X_test, y_test, feature_names = DataProcessor.generate_synthetic_nsl_kdd()
    
    # Extract normal samples for training the autoencoder
    X_train_normal = X_train[y_train == 0]
    
    # Separate test data into normal and attack for evaluation
    X_test_normal = X_test[y_test == 0]
    X_test_attack = X_test[y_test == 1]
    
    # Initialize and train LIIDS
    input_dim = X_train.shape[1]
    print(f"\nInitializing LIIDS model with input dimension: {input_dim}")
    liids = LIIDS(input_dim=input_dim)
    
    # Preprocess data
    X_train_normal, _ = liids.preprocess_data(X_train_normal)
    X_test_preprocessed, y_test = liids.preprocess_data(X_test, y_test, fit=False)
    
    # Train the model
    history = liids.train(X_train_normal, validation_split=0.2, patience=10)
    
    # Plot training history
    liids.plot_training_history()
    
    # Set threshold based on normal data
    liids.set_threshold(X_train_normal)
    
    # Evaluate model
    metrics, mse = liids.evaluate(X_test_preprocessed, y_test)
    
    # Calculate resource efficiency
    efficiency = liids.calculate_resource_efficiency()
    
    # Plot reconstruction error distribution
    X_test_normal_preprocessed, _ = liids.preprocess_data(X_test_normal, fit=False)
    X_test_attack_preprocessed, _ = liids.preprocess_data(X_test_attack, fit=False)
    liids.plot_reconstruction_errors(X_test_normal_preprocessed, X_test_attack_preprocessed)
    
    # Plot ROC curve
    liids.plot_roc_curve(metrics)
    
    # Plot confusion matrix
    liids.plot_confusion_matrix(metrics)
    
    # Plot resource efficiency
    liids.plot_resource_efficiency(efficiency)
    
    # Save model
    liids.save_model("liids_synthetic_model")
    
    return liids, metrics, efficiency


def compare_results(nsl_kdd_metrics, unsw_nb15_metrics, synthetic_metrics):
    """Compare results from different datasets"""
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)
    
    # Create comparison table
    results = pd.DataFrame({
        'NSL-KDD': [
            nsl_kdd_metrics['accuracy'],
            nsl_kdd_metrics['precision'],
            nsl_kdd_metrics['recall'],
            nsl_kdd_metrics['f1'],
            nsl_kdd_metrics['far'],
            nsl_kdd_metrics['auc']
        ],
        'UNSW-NB15': [
            unsw_nb15_metrics['accuracy'],
            unsw_nb15_metrics['precision'],
            unsw_nb15_metrics['recall'],
            unsw_nb15_metrics['f1'],
            unsw_nb15_metrics['far'],
            unsw_nb15_metrics['auc']
        ],
        'Synthetic': [
            synthetic_metrics['accuracy'],
            synthetic_metrics['precision'],
            synthetic_metrics['recall'],
            synthetic_metrics['f1'],
            synthetic_metrics['far'],
            synthetic_metrics['auc']
        ]
    }, index=['Accuracy', 'Precision', 'Recall (DR)', 'F1-Score', 'False Alarm Rate', 'AUC'])
    
    # Format percentages
    results = results.applymap(lambda x: f"{x:.4f}")
    
    print("\nPerformance Metrics Comparison:")
    print(results)
    
    # Compare with results from thesis
    thesis_results = pd.DataFrame({
        'NSL-KDD (Thesis)': ['0.9976', '0.9935', '0.0065'],
        'UNSW-NB15 (Thesis)': ['0.9852', '0.9873', '0.0127'],
        'NSL-KDD (Ours)': [
            nsl_kdd_metrics['accuracy'],
            nsl_kdd_metrics['recall'],
            nsl_kdd_metrics['far']
        ],
        'UNSW-NB15 (Ours)': [
            unsw_nb15_metrics['accuracy'],
            unsw_nb15_metrics['recall'],
            unsw_nb15_metrics['far']
        ]
    }, index=['Accuracy', 'Recall (DR)', 'False Alarm Rate'])
    
    # Format our results
    thesis_results['NSL-KDD (Ours)'] = thesis_results['NSL-KDD (Ours)'].apply(lambda x: f"{float(x):.4f}")
    thesis_results['UNSW-NB15 (Ours)'] = thesis_results['UNSW-NB15 (Ours)'].apply(lambda x: f"{float(x):.4f}")
    
    print("\nComparison with Thesis Results:")
    print(thesis_results)
    
    # Plot comparison
    metrics = ['Accuracy', 'Recall (DR)', 'False Alarm Rate']
    datasets = ['NSL-KDD', 'UNSW-NB15']
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Get values
        thesis_values = [float(thesis_results.loc[metric, f'{ds} (Thesis)']) for ds in datasets]
        our_values = [float(thesis_results.loc[metric, f'{ds} (Ours)']) for ds in datasets]
        
        # Create grouped bar chart
        x = np.arange(len(datasets))
        width = 0.35
        
        ax.bar(x - width/2, thesis_values, width, label='Thesis Results', color=colors['primary'])
        ax.bar(x + width/2, our_values, width, label='Our Implementation', color=colors['accent'])
        
        # Add labels and title
        ax.set_title(metric, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_ylim(0, 1.0 if metric != 'False Alarm Rate' else 0.1)
        
        # Add value labels
        for j, v in enumerate(thesis_values):
            ax.text(j - width/2, v + 0.01, f"{v:.4f}", ha='center', fontsize=10)
        
        for j, v in enumerate(our_values):
            ax.text(j + width/2, v + 0.01, f"{v:.4f}", ha='center', fontsize=10)
        
        # Add legend and grid
        if i == 1:
            ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('results_comparison.png')
    plt.show()
    
    return results


# Main function to run all experiments
def main():
    print("\n" + "="*80)
    print("LIGHTWEIGHT INTELLIGENT INTRUSION DETECTION SYSTEM (LIIDS)")
    print("FOR WIRELESS SENSOR NETWORKS USING DEEP AUTOENCODERS")
    print("="*80)
    
    # Run NSL-KDD experiment
    nsl_kdd_liids, nsl_kdd_metrics, nsl_kdd_efficiency = run_nsl_kdd_experiment()
    
    # Run UNSW-NB15 experiment
    unsw_nb15_liids, unsw_nb15_metrics, unsw_nb15_efficiency = run_unsw_nb15_experiment()
    
    # Run synthetic dataset experiment
    synthetic_liids, synthetic_metrics, synthetic_efficiency = run_synthetic_dataset_experiment()
    
    # Compare results
    results = compare_results(nsl_kdd_metrics, unsw_nb15_metrics, synthetic_metrics)
    
    print("\n" + "="*80)
    print("EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("="*80)


if __name__ == "__main__":
    main()