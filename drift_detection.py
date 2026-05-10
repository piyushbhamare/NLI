import numpy as np
from collections import deque
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# ENHANCED EWMA DETECTOR
# ============================================================================


class EWMADriftDetector:
    """Exponentially Weighted Moving Average for error tracking"""
    
    def __init__(self, alpha=0.3, threshold=1.8):
        self.alpha = alpha
        self.threshold = threshold
        self.ewma = None
        self.baseline = None
        self.baseline_std = None
        self.error_history = deque(maxlen=2000)
        
    def update(self, new_error):
        """Update EWMA with new error measurement"""
        self.error_history.append(new_error)
        
        if self.ewma is None:
            self.ewma = new_error
            self.baseline = new_error
        else:
            self.ewma = self.alpha * new_error + (1 - self.alpha) * self.ewma
        
        # Update baseline statistics
        if len(self.error_history) >= 200:
            recent_errors = list(self.error_history)[-200:]
            self.baseline = np.mean(recent_errors)
            self.baseline_std = np.std(recent_errors) + 1e-10
    
    def is_drifting(self):
        """Check if current EWMA indicates drift"""
        if self.baseline_std is None or self.baseline_std == 0:
            return False
        
        z_score = (self.ewma - self.baseline) / self.baseline_std
        return z_score > self.threshold
    
    def get_confidence(self):
        """Get confidence score [0, 1] for drift detection"""
        if self.baseline_std is None or self.baseline_std == 0:
            return 0.0
        
        z_score = (self.ewma - self.baseline) / self.baseline_std
        confidence = min(1.0, max(0.0, z_score / self.threshold))
        return confidence


# ============================================================================
# IMPROVED PSI DETECTOR - PRIMARY DETECTOR
# ============================================================================


class PSIDriftDetector:
    """Population Stability Index - detects distribution shifts"""
    
    def __init__(self, n_bins=25, threshold=0.06):
        self.n_bins = n_bins
        self.threshold = threshold
        self.baseline_distribution = None
        self.baseline_edges = None
        self.baseline_counts = None
        
    def fit_baseline(self, keys):
        """Fit detector to baseline key distribution"""
        if keys is None or len(keys) == 0:
            return
        
        keys_array = np.array(keys, dtype=np.float64)
        
        # Use adaptive binning based on data distribution
        if len(keys_array) >= self.n_bins * 10:
            # Quantile-based binning for better sensitivity
            quantiles = np.linspace(0, 100, self.n_bins + 1)
            self.baseline_edges = np.percentile(keys_array, quantiles)
            
            # Ensure unique edges
            self.baseline_edges = np.unique(self.baseline_edges)
            if len(self.baseline_edges) < 3:
                # Fallback to linear binning
                self.baseline_edges = np.linspace(
                    keys_array.min(), 
                    keys_array.max(), 
                    self.n_bins + 1
                )
        else:
            # Standard linear binning for small datasets
            self.baseline_edges = np.linspace(
                keys_array.min(),
                keys_array.max(),
                self.n_bins + 1
            )
        
        # Compute baseline distribution with Laplace smoothing
        self.baseline_counts, _ = np.histogram(keys_array, bins=self.baseline_edges)
        self.baseline_distribution = (self.baseline_counts + 1) / (
            self.baseline_counts.sum() + len(self.baseline_counts)
        )
        
    def compute_psi(self, current_keys):
        """Compute Population Stability Index"""
        if self.baseline_edges is None or current_keys is None:
            return 0.0
        
        if len(current_keys) == 0:
            return 0.0
        
        keys_array = np.array(current_keys, dtype=np.float64)
        
        # Clip keys to baseline range
        keys_array = np.clip(
            keys_array,
            self.baseline_edges[0],
            self.baseline_edges[-1]
        )
        
        # Compute current distribution
        current_counts, _ = np.histogram(keys_array, bins=self.baseline_edges)
        current_distribution = (current_counts + 1) / (
            current_counts.sum() + len(current_counts)
        )
        
        # PSI formula with numerical stability
        eps = 1e-10
        psi = np.sum(
            (current_distribution - self.baseline_distribution) * 
            np.log((current_distribution + eps) / (self.baseline_distribution + eps))
        )
        
        return max(0.0, psi)
    
    def is_drifting(self, current_keys):
        """Check if PSI exceeds threshold"""
        psi = self.compute_psi(current_keys)
        return psi > self.threshold
    
    def get_normalized_score(self, current_keys):
        """Get normalized PSI score [0, 1]"""
        psi = self.compute_psi(current_keys)
        # ✅ FIX: Normalize consistently with threshold
        normalize_by = self.threshold * 3.0  # = 0.18 when threshold=0.06
        return min(1.0, psi / normalize_by)


# ============================================================================
# LIGHTWEIGHT AUTOENCODER (OPTIONAL)
# ============================================================================


class SimpleAutoencoder:
    """Lightweight autoencoder for anomaly detection"""
    
    def __init__(self, input_dim, encoding_dim=6):
        self.available = False
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.baseline_error = None
        self.autoencoder = None
        
        try:
            import tensorflow as tf
            from tensorflow import keras
            tf.get_logger().setLevel('ERROR')
            
            # Simple autoencoder architecture
            encoder_input = keras.Input(shape=(input_dim,))
            encoded = keras.layers.Dense(encoding_dim, activation='relu')(encoder_input)
            encoder = keras.Model(encoder_input, encoded, name='encoder')
            
            decoder_input = keras.Input(shape=(encoding_dim,))
            decoded = keras.layers.Dense(input_dim, activation='linear')(decoder_input)
            decoder = keras.Model(decoder_input, decoded, name='decoder')
            
            autoencoder_input = keras.Input(shape=(input_dim,))
            encoded_data = encoder(autoencoder_input)
            decoded_data = decoder(encoded_data)
            self.autoencoder = keras.Model(autoencoder_input, decoded_data, name='autoencoder')
            
            self.autoencoder.compile(optimizer='adam', loss='mse')
            self.available = True
            
        except Exception as e:
            # Graceful fallback - autoencoder is optional
            self.available = False
    
    def fit(self, X, epochs=8, batch_size=32):
        """Train autoencoder on baseline data"""
        if not self.available or X is None or len(X) == 0:
            return
        
        try:
            self.autoencoder.fit(
                X, X, 
                epochs=epochs, 
                batch_size=batch_size, 
                verbose=0,
                shuffle=True
            )
            
            # Compute baseline reconstruction error
            X_pred = self.autoencoder.predict(X, verbose=0)
            self.baseline_error = np.mean((X - X_pred) ** 2)
            
        except Exception as e:
            self.available = False
    
    def get_reconstruction_error(self, X):
        """Get reconstruction error for new data"""
        if not self.available or X is None or len(X) == 0:
            return np.zeros(len(X)) if X is not None else np.array([])
        
        try:
            X_pred = self.autoencoder.predict(X, verbose=0)
            mse = np.mean((X - X_pred) ** 2, axis=1)
            return mse
        except:
            return np.zeros(len(X))
    
    def get_normalized_score(self, X):
        """Get normalized anomaly score [0, 1]"""
        if not self.available or self.baseline_error is None:
            return 0.0
        
        errors = self.get_reconstruction_error(X)
        if len(errors) == 0:
            return 0.0
        
        mean_error = np.mean(errors)
        # Normalize: 2x baseline error = score of 1.0
        score = mean_error / (2.0 * self.baseline_error)
        return min(1.0, max(0.0, score))


# ============================================================================
# COMBINED DRIFT DETECTOR - ENHANCED
# ============================================================================


class CombinedDriftDetector:
    """
    Multi-method drift detector combining:
    - EWMA (error tracking)
    - PSI (distribution shift detection) - PRIMARY
    - Autoencoder (anomaly detection) - OPTIONAL
    """
    
    def __init__(self):
        self.ewma_detector = EWMADriftDetector(alpha=0.3, threshold=1.8)
        self.psi_detector = PSIDriftDetector(n_bins=25, threshold=0.06)
        self.autoencoder = None
        
        # Statistics tracking
        self.drift_events = []
        self.repair_actions = []
        self.detection_scores = {
            'ewma': [],
            'psi': [],
            'autoencoder': [],
            'combined': []
        }
        
    def fit_baseline(self, baseline_keys):
        """
        Fit all detectors to baseline data
        
        Args:
            baseline_keys: List of keys from baseline data
        """
        if baseline_keys is None or len(baseline_keys) == 0:
            return
        
        # 1. Train EWMA on simulated errors
        n_samples = min(len(baseline_keys), 2000)
        simulated_errors = np.random.uniform(10, 50, n_samples)
        for error in simulated_errors:
            self.ewma_detector.update(error)
        
        # 2. Train PSI on key distribution
        self.psi_detector.fit_baseline(baseline_keys)
        
        # 3. Train autoencoder on features (optional)
        features = self._keys_to_features(baseline_keys)
        if features is not None and len(features) >= 100:
            self.autoencoder = SimpleAutoencoder(features.shape[1], encoding_dim=6)
            if self.autoencoder.available:
                try:
                    self.autoencoder.fit(features, epochs=8, batch_size=32)
                except:
                    self.autoencoder = None
    
    def _keys_to_features(self, keys, window_size=None):
        """
        Convert key sequence to statistical features
        
        Args:
            keys: List of keys
            window_size: Size of sliding window (adaptive if None)
            
        Returns:
            numpy array of shape (n_windows, n_features)
        """
        if keys is None:
            return None
        
        # ✅ FIX: Adaptive window size
        if window_size is None:
            window_size = max(8, min(64, len(keys) // 100))
        
        if len(keys) < window_size:
            return None
        
        features = []
        keys_array = np.array(keys, dtype=np.float64)
        
        # Sliding window with stride
        stride = max(1, len(keys_array) // 500)
        
        for i in range(0, len(keys_array) - window_size, stride):
            window = keys_array[i:i+window_size]
            if len(window) < window_size:
                continue
            
            # Statistical features
            diffs = np.diff(window)
            feature_vec = np.array([
                np.mean(window),           # Mean
                np.std(window) + 1e-10,    # Std dev
                np.min(window),            # Min
                np.max(window),            # Max
                np.median(window),         # Median
                np.percentile(window, 25), # Q1
                np.percentile(window, 75), # Q3
                np.max(diffs) if len(diffs) > 0 else 0,   # Max gap
                np.mean(diffs) if len(diffs) > 0 else 0,  # Mean gap
                len(np.unique(window)),    # Cardinality
                np.sum(diffs > 0) if len(diffs) > 0 else 0, # Monotonicity
                window[-1] - window[0]     # Range
            ])
            features.append(feature_vec)
        
        return np.array(features) if features else None
    
    def detect_drift(self, current_keys, current_errors):
        """
        Detect drift using ensemble of methods
        
        Args:
            current_keys: List of current keys
            current_errors: List of current prediction errors
            
        Returns:
            Tuple of (is_drifting, combined_score, method_scores)
        """
        
        # Method 1: EWMA on errors (20% weight)
        ewma_score = 0.0
        if current_errors is not None and len(current_errors) > 0:
            for error in current_errors[:200]:
                self.ewma_detector.update(error)
            ewma_score = self.ewma_detector.get_confidence()
        self.detection_scores['ewma'].append(ewma_score)
        
        # Method 2: PSI on key distribution (60% weight) - PRIMARY
        psi_score = self.psi_detector.get_normalized_score(current_keys)
        self.detection_scores['psi'].append(psi_score)
        
        # Method 3: Autoencoder anomaly score (20% weight) - OPTIONAL
        ae_score = 0.0
        if self.autoencoder is not None and self.autoencoder.available:
            try:
                features = self._keys_to_features(current_keys)
                if features is not None and len(features) > 0:
                    ae_score = self.autoencoder.get_normalized_score(features)
            except:
                ae_score = 0.0
        self.detection_scores['autoencoder'].append(ae_score)
        
        # Weighted combination (PSI is primary detector)
        weights = {'ewma': 0.2, 'psi': 0.6, 'autoencoder': 0.2}
        combined_score = (
            weights['ewma'] * ewma_score +
            weights['psi'] * psi_score +
            weights['autoencoder'] * ae_score
        )
        self.detection_scores['combined'].append(combined_score)
        
        # Drift threshold (lowered for better sensitivity)
        drift_threshold = 0.40
        is_drifting = combined_score > drift_threshold
        
        # Record drift event
        if is_drifting:
            self.drift_events.append({
                'timestamp': datetime.now(),
                'ewma_score': ewma_score,
                'psi_score': psi_score,
                'ae_score': ae_score,
                'combined_score': combined_score
            })
        
        method_scores = {
            'ewma': ewma_score,
            'psi': psi_score,
            'autoencoder': ae_score
        }
        
        return is_drifting, combined_score, method_scores
    
    def trigger_repair(self, repair_type='REFIT'):
        """
        Record a repair action
        
        Args:
            repair_type: Type of repair (REFIT, SPLIT, MERGE, etc.)
        """
        self.repair_actions.append({
            'timestamp': datetime.now(),
            'repair_type': repair_type,
            'drift_event_idx': len(self.drift_events) - 1 if self.drift_events else -1
        })
    
    def get_statistics(self):
        """Get summary statistics"""
        stats = {
            'drift_events': len(self.drift_events),
            'repair_actions': len(self.repair_actions),
            'avg_ewma_score': (np.mean(self.detection_scores['ewma']) 
                              if self.detection_scores['ewma'] else 0.0),
            'avg_psi_score': (np.mean(self.detection_scores['psi']) 
                             if self.detection_scores['psi'] else 0.0),
            'avg_ae_score': (np.mean(self.detection_scores['autoencoder']) 
                            if self.detection_scores['autoencoder'] else 0.0),
            'avg_combined_score': (np.mean(self.detection_scores['combined'])
                                  if self.detection_scores['combined'] else 0.0)
        }
        return stats
    
    def reset(self):
        """Reset detector state"""
        self.drift_events = []
        self.repair_actions = []
        self.detection_scores = {
            'ewma': [],
            'psi': [],
            'autoencoder': [],
            'combined': []
        }
