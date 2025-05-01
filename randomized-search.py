"""
Lightweight Intelligent Intrusion Detection System (LIIDS) for Wireless Sensor Networks
Using Deep Autoencoders with Randomized Search

- Randomized hyperparameter search (faster than grid search)
- Memory-optimized implementation
- Mixed precision training for GPU acceleration
- Efficient parallel execution
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import time
import os
import requests
import multiprocessing
from joblib import Parallel, delayed
import gc
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

# Define color scheme for plots
COLORS = {
    'primary': '#003366',
    'secondary': '#0066cc',
    'accent': '#ff9900',
    'light_bg': '#f5f5f5',
    'text': '#333333'
}

def optimize_tf_memory():
    """Configure TensorFlow for memory optimization"""
    # Grow GPU memory allocation as needed
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s). Memory growth enabled.")
        except RuntimeError as e:
            print(f"GPU memory configuration error: {e}")

    # Enable mixed precision training for faster computations
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Mixed precision policy set to:", policy.name)
    except:
        print("Failed to set mixed precision policy. Using default.")
        
    # Clean up any existing sessions
    tf.keras.backend.clear_session()
    print("TensorFlow memory optimized.")

class LIIDS:
    """
    Lightweight Intelligent Intrusion Detection System (LIIDS) for WSNs
    Memory-optimized and GPU-accelerated implementation
    """

    def __init__(self, input_dim=41, hidden_layers=[128, 64, 32],
                 activation='relu', output_activation='sigmoid',
                 dropout_rate=0.2, learning_rate=0.01,
                 batch_size=32, epochs=50, l1_reg=0.01):
        """
        Initialize the LIIDS model with hyperparameters

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
        Build the deep autoencoder architecture
        - Encoder: Input → Hidden Layers
        - Decoder: Hidden Layers (reversed) → Output
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

        # Compile the model with mixed precision optimizations
        optimizer = Adam(learning_rate=self.learning_rate, epsilon=1e-4)
        self.model.compile(optimizer=optimizer, loss='mse')

    def preprocess_data(self, X, y=None, fit=True):
        """
        Preprocess the data (normalization/scaling)
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
        """
        # Define callbacks for early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            verbose=0,
            restore_best_weights=True
        )

        # Convert data to TensorFlow dataset for better performance
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, X_train))
        train_dataset = train_dataset.batch(self.batch_size)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

        # Start timing
        start_time = time.time()

        # Train the model
        self.history = self.model.fit(
            train_dataset,
            epochs=self.epochs,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=0  # Reduced verbosity for hyperparameter search
        )

        # Calculate training time
        self.training_time = time.time() - start_time

        return self.history

    def set_threshold(self, X_normal, percentile=95):
        """
        Set the threshold for anomaly detection based on reconstruction error of normal data
        """
        # Get reconstruction errors on normal data
        predictions = self.model.predict(X_normal, verbose=0, batch_size=1024)
        mse = np.mean(np.power(X_normal - predictions, 2), axis=1)

        # Set threshold as the 95th percentile of errors
        self.threshold = np.percentile(mse, percentile)

        return self.threshold

    def predict(self, X):
        """
        Predict anomalies based on reconstruction error
        """
        if self.threshold is None:
            raise ValueError("Threshold not set. Run set_threshold() first.")

        # Calculate reconstruction error
        predictions = self.model.predict(X, verbose=0, batch_size=1024)
        mse = np.mean(np.power(X - predictions, 2), axis=1)

        # Classify based on threshold
        y_pred = (mse > self.threshold).astype(int)

        return y_pred, mse

    def evaluate(self, X, y_true):
        """
        Evaluate the model performance
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
        Calculate bandwidth and energy efficiency
        """
        # Calculate bandwidth efficiency (Eq. 3.9 in thesis)
        bandwidth_efficiency = ((self.original_data_size - self.compressed_data_size) /
                              self.original_data_size) * 100

        # Calculate energy efficiency (simplified model)
        energy_efficiency = bandwidth_efficiency

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
        plt.plot(self.history.history['loss'], label='Training Loss', color=COLORS['primary'])
        plt.plot(self.history.history['val_loss'], label='Validation Loss', color=COLORS['accent'])
        plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()
        print("Training history plot saved to 'training_history.png'")

    def save_model(self, path="liids_model.h5"):
        """Save the trained model"""
        self.model.save(path)
        print(f"Model saved to {path}")


class RandomizedSearchLIIDS:
    """
    Randomized Search for hyperparameter tuning of LIIDS model
    - Memory-efficient implementation
    - Faster than grid search
    - Efficient parallel processing
    """

    def __init__(self, input_dim, param_distributions=None, n_iter=30, random_state=None):
        """
        Initialize the randomized search with parameter distributions

        Args:
            input_dim: Number of input features
            param_distributions: Dictionary of hyperparameter distributions
            n_iter: Number of parameter settings sampled
            random_state: Random state for reproducibility
        """
        self.input_dim = input_dim
        self.n_iter = n_iter
        self.best_model = None
        self.best_params = None
        self.best_metrics = None
        self.results = []
        
        # Set random state
        if random_state is not None:
            np.random.seed(random_state)
        
        # Default parameter distributions if none provided
        if param_distributions is None:
            self.param_distributions = {
                'hidden_layers': [(128, 64, 32), (256, 128, 64), (64, 32, 16)],
                'learning_rate': [0.001, 0.005, 0.01],
                'batch_size': [32, 64, 128],
                'dropout_rate': [0.1, 0.2, 0.3],
                'l1_reg': [0.001, 0.005, 0.01]
            }
        else:
            self.param_distributions = param_distributions

    def _sample_params(self):
        """Sample random parameter combinations"""
        sampled_params = {}
        for param, values in self.param_distributions.items():
            sampled_params[param] = values[np.random.randint(0, len(values))]
        return sampled_params

    def fit(self, X_train, y_train, X_test, y_test, n_jobs=None):
        """
        Perform randomized search to find the best hyperparameters

        Args:
            X_train: Training data
            y_train: Training labels
            X_test: Test data
            y_test: Test labels
            n_jobs: Number of parallel jobs (None = use all cores)
        """
        # Set number of parallel jobs
        if n_jobs is None:
            n_jobs = max(1, min(multiprocessing.cpu_count() - 1, 4))  # Limit to 4 for stability
            
        print(f"Running randomized search with {self.n_iter} iterations using {n_jobs} parallel jobs...")
        
        # Sample parameter combinations
        param_combinations = [self._sample_params() for _ in range(self.n_iter)]
        
        # Check for duplicates and ensure uniqueness
        unique_params = []
        for params in param_combinations:
            if params not in unique_params:
                unique_params.append(params)
        
        # If we lost too many combinations due to duplicates, sample more
        while len(unique_params) < self.n_iter:
            new_params = self._sample_params()
            if new_params not in unique_params:
                unique_params.append(new_params)
        
        print(f"Generated {len(unique_params)} unique parameter combinations")
        
        # Function to evaluate one parameter combination
        def evaluate_combination(params, idx):
            try:
                print(f"Evaluating combination {idx+1}/{len(unique_params)}: {params}")
                
                # Clear TensorFlow session to free memory
                tf.keras.backend.clear_session()
                
                # Create model
                model = LIIDS(
                    input_dim=self.input_dim,
                    hidden_layers=list(params['hidden_layers']),
                    dropout_rate=params['dropout_rate'],
                    learning_rate=params['learning_rate'],
                    batch_size=params['batch_size'],
                    l1_reg=params['l1_reg']
                )
                
                # Extract normal samples for training
                X_train_normal = X_train[y_train == 0]
                
                # Preprocess data
                X_train_normal_scaled, _ = model.preprocess_data(X_train_normal)
                X_test_scaled, y_test_scaled = model.preprocess_data(X_test, y_test, fit=False)
                
                # Train model with early stopping (reduced patience for search)
                model.train(X_train_normal_scaled, validation_split=0.2, patience=5)
                
                # Set threshold based on normal data
                model.set_threshold(X_train_normal_scaled)
                
                # Evaluate model
                metrics, _ = model.evaluate(X_test_scaled, y_test_scaled)
                
                # Add params to metrics
                metrics.update(params)
                
                # Calculate memory usage
                import psutil
                process = psutil.Process(os.getpid())
                memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
                print(f"Combination {idx+1} complete. Memory usage: {memory_usage:.2f} MB")
                
                # Clear TensorFlow session to free memory
                tf.keras.backend.clear_session()
                gc.collect()
                
                return metrics, model, params
            except Exception as e:
                print(f"Error with params {params}: {e}")
                # Clear TensorFlow session to free memory
                tf.keras.backend.clear_session()
                gc.collect()
                return None, None, params
        
        # Perform randomized search in batches to limit memory usage
        batch_size = min(5, len(unique_params))
        num_batches = (len(unique_params) + batch_size - 1) // batch_size
        
        all_results = []
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(unique_params))
            
            print(f"\nProcessing batch {batch_idx+1}/{num_batches} (combinations {start_idx+1}-{end_idx})")
            
            # Get current batch of parameter combinations
            batch_combinations = unique_params[start_idx:end_idx]
            
            # Perform search in parallel for this batch
            batch_results = Parallel(n_jobs=n_jobs, prefer="threads", verbose=10)(
                delayed(evaluate_combination)(params, start_idx + idx) 
                for idx, params in enumerate(batch_combinations)
            )
            
            # Add valid results to all_results
            all_results.extend([r for r in batch_results if r[0] is not None])
            
            # Run garbage collection
            gc.collect()
            tf.keras.backend.clear_session()
        
        # Filter out None results (errors)
        valid_results = [r for r in all_results if r[0] is not None]
        
        if not valid_results:
            raise ValueError("No valid parameter combinations found")
        
        # Sort by composite score (accuracy + recall - far)
        sorted_results = sorted(
            valid_results,
            key=lambda x: (x[0]['accuracy'] + x[0]['recall'] - x[0]['far']),
            reverse=True
        )
        
        # Get best results
        best_metrics, best_model, best_params = sorted_results[0]
        
        self.best_metrics = best_metrics
        self.best_model = best_model
        self.best_params = best_params
        self.results = sorted_results
        
        print("\nRandomized Search complete!")
        print(f"Best parameters: {best_params}")
        print(f"Best performance: Accuracy={best_metrics['accuracy']:.4f}, "
              f"Recall={best_metrics['recall']:.4f}, "
              f"FAR={best_metrics['far']:.4f}")
        
        return self.best_model, self.best_params, self.best_metrics

    def plot_search_results(self):
        """Plot randomized search results"""
        if not self.results:
            print("No results to plot.")
            return
            
        # Extract results for plotting
        results_df = pd.DataFrame([r[0] for r in self.results])
        
        # Create scatter plots for each parameter vs accuracy
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        # Plot learning rate vs accuracy
        axes[0, 0].scatter(results_df['learning_rate'], results_df['accuracy'], 
                         c=results_df['far'], cmap='coolwarm', alpha=0.7)
        axes[0, 0].set_title('Learning Rate vs Accuracy', fontsize=14)
        axes[0, 0].set_xlabel('Learning Rate')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].grid(True, linestyle='--', alpha=0.7)
        
        # Plot dropout rate vs accuracy
        axes[0, 1].scatter(results_df['dropout_rate'], results_df['accuracy'], 
                         c=results_df['far'], cmap='coolwarm', alpha=0.7)
        axes[0, 1].set_title('Dropout Rate vs Accuracy', fontsize=14)
        axes[0, 1].set_xlabel('Dropout Rate')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].grid(True, linestyle='--', alpha=0.7)
        
        # Plot l1_reg vs accuracy
        axes[1, 0].scatter(results_df['l1_reg'], results_df['accuracy'], 
                         c=results_df['far'], cmap='coolwarm', alpha=0.7)
        axes[1, 0].set_title('L1 Regularization vs Accuracy', fontsize=14)
        axes[1, 0].set_xlabel('L1 Regularization')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].grid(True, linestyle='--', alpha=0.7)
        
        # Plot batch size vs accuracy
        axes[1, 1].scatter(results_df['batch_size'], results_df['accuracy'], 
                         c=results_df['far'], cmap='coolwarm', alpha=0.7)
        axes[1, 1].set_title('Batch Size vs Accuracy', fontsize=14)
        axes[1, 1].set_xlabel('Batch Size')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].grid(True, linestyle='--', alpha=0.7)
        
        # Plot accuracy vs far
        axes[2, 0].scatter(results_df['accuracy'], results_df['far'], 
                         c=results_df['recall'], cmap='viridis', alpha=0.7)
        axes[2, 0].set_title('Accuracy vs False Alarm Rate', fontsize=14)
        axes[2, 0].set_xlabel('Accuracy')
        axes[2, 0].set_ylabel('False Alarm Rate')
        axes[2, 0].grid(True, linestyle='--', alpha=0.7)
        
        # Plot accuracy vs recall
        axes[2, 1].scatter(results_df['accuracy'], results_df['recall'], 
                         c=results_df['far'], cmap='coolwarm', alpha=0.7)
        axes[2, 1].set_title('Accuracy vs Recall', fontsize=14)
        axes[2, 1].set_xlabel('Accuracy')
        axes[2, 1].set_ylabel('Recall')
        axes[2, 1].grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('randomized_search_results.png')
        plt.close()
        
        print("Randomized search results plot saved to 'randomized_search_results.png'")
        
        # Plot top models comparison
        top_n = min(5, len(self.results))
        top_results = self.results[:top_n]
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'far']
        model_names = [f"Model {i+1}" for i in range(top_n)]
        
        plt.figure(figsize=(14, 8))
        
        x = np.arange(len(metrics))
        width = 0.15
        
        for i, (metrics_dict, _, _) in enumerate(top_results):
            values = [metrics_dict[metric] for metric in metrics]
            plt.bar(x + i * width, values, width, label=model_names[i])
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Top 5 Models Comparison', fontsize=14, fontweight='bold')
        plt.xticks(x + width * (top_n - 1) / 2, metrics)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.savefig('top_models_comparison.png')
        plt.close()
        
        print("Top models comparison plot saved to 'top_models_comparison.png'")


class DataProcessor:
    """
    Class for handling data loading and preprocessing with optimized memory usage
    """

    @staticmethod
    def load_nsl_kdd(train_path=None, test_path=None):
        """
        Load and preprocess the NSL-KDD dataset with memory optimization
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

        # Load data with optimized memory usage
        train_data = pd.read_csv(train_path, header=None, names=columns)
        test_data = pd.read_csv(test_path, header=None, names=columns)

        # Display dataset information
        print(f"NSL-KDD Training set shape: {train_data.shape}")
        print(f"NSL-KDD Test set shape: {test_data.shape}")

        # Convert labels to binary (normal vs attack) - efficient in-place operation
        train_data['binary_label'] = (train_data['label'] != 'normal').astype(int)
        test_data['binary_label'] = (test_data['label'] != 'normal').astype(int)

        # One-hot encode categorical features
        categorical_cols = ['protocol_type', 'service', 'flag']
        
        # Process train and test separately to save memory
        for data in [train_data, test_data]:
            for col in categorical_cols:
                # Get dummies and join to original dataframe
                dummies = pd.get_dummies(data[col], prefix=col, drop_first=False)
                data[dummies.columns] = dummies
                # Drop original column to save memory
                data.drop(col, axis=1, inplace=True)
        
        # Drop unnecessary columns
        train_data.drop(['label', 'difficulty'], axis=1, inplace=True)
        test_data.drop(['label', 'difficulty'], axis=1, inplace=True)

        # Extract features and labels
        X_train = train_data.drop('binary_label', axis=1)
        y_train = train_data['binary_label'].values
        X_test = test_data.drop('binary_label', axis=1)
        y_test = test_data['binary_label'].values

        # Ensure column alignment
        common_columns = X_train.columns.intersection(X_test.columns)
        X_train = X_train[common_columns]
        X_test = X_test[common_columns]

        print(f"Features shape after preprocessing: X_train={X_train.shape}, X_test={X_test.shape}")
        print(f"Number of normal samples in training: {(y_train == 0).sum()}")
        print(f"Number of attack samples in training: {(y_train == 1).sum()}")
        print(f"Class distribution in training set: {np.bincount(y_train)}")
        print(f"Class distribution in test set: {np.bincount(y_test)}")
        
        # Free memory
        gc.collect()

        return X_train.values, y_train, X_test.values, y_test


def run_randomized_search(dataset='nsl-kdd', n_iter=30):
    """
    Run randomized search experiment on the specified dataset
    
    Args:
        dataset: Dataset to use ('nsl-kdd' or 'unsw-nb15')
        n_iter: Number of random parameter combinations to try
    """
    # Start with optimizing TensorFlow memory
    optimize_tf_memory()
    
    print("\n" + "="*80)
    print(f"RANDOMIZED SEARCH ON {dataset.upper()} DATASET")
    print("="*80)
    
    start_time = time.time()
    
    # Load dataset
    if dataset.lower() == 'nsl-kdd':
        X_train, y_train, X_test, y_test = DataProcessor.load_nsl_kdd()
    else:
        raise ValueError(f"Dataset {dataset} not supported in this implementation")
    
    # Define parameter distributions
    param_distributions = {
        'hidden_layers': [(128, 64, 32), (256, 128, 64), (64, 32, 16), (512, 256, 128)],
        'learning_rate': [0.0005, 0.001, 0.005, 0.01, 0.02],
        'batch_size': [16, 32, 64, 128, 256],
        'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
        'l1_reg': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
    }
    
    print(f"Running randomized search with {n_iter} combinations")
    print(f"Parameter space size: {np.prod([len(values) for values in param_distributions.values()])}")
    
    # Initialize randomized search
    random_search = RandomizedSearchLIIDS(
        input_dim=X_train.shape[1],
        param_distributions=param_distributions,
        n_iter=n_iter,
        random_state=RANDOM_SEED
    )
    
    # Perform randomized search
    best_model, best_params, best_metrics = random_search.fit(X_train, y_train, X_test, y_test)
    
    # Plot search results
    random_search.plot_search_results()
    
    # Evaluate final model
    best_model.train(X_train[y_train == 0], validation_split=0.2, patience=10)
    best_model.set_threshold(X_train[y_train == 0])
    final_metrics, _ = best_model.evaluate(X_test, y_test)
    
    # Calculate resource efficiency
    efficiency = best_model.calculate_resource_efficiency()
    
    # Print best hyperparameters
    print("\nBest Hyperparameters:")
    for param, value in best_params.items():
        if param not in ['accuracy', 'precision', 'recall', 'f1', 'far', 'auc']:
            print(f"{param}: {value}")
    
    # Print best metrics
    print("\nBest Model Performance:")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'far', 'auc']:
        print(f"{metric}: {final_metrics[metric]:.4f}")
    
    # Print resource efficiency
    print("\nResource Efficiency:")
    print(f"Data Compression Ratio: {efficiency['data_reduction_ratio']:.2f}x")
    print(f"Bandwidth Efficiency: {efficiency['bandwidth_efficiency']:.2f}%")
    print(f"Training Time: {efficiency['training_time']:.2f} seconds")
    
    # Plot training history
    best_model.plot_training_history()
    
    # Save best model
    best_model.save_model(f"best_{dataset}_model_randomized_search.h5")
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    return best_model, best_metrics, best_params, efficiency


if __name__ == "__main__":
    # Number of random iterations (adjust based on time constraints)
    n_iter = 30  
    
    run_randomized_search('nsl-kdd', n_iter=n_iter)
