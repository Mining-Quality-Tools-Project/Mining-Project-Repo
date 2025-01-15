import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel

class EnsembleModel:
    def __init__(self):
        # Initialize base models with enhanced hyperparameters
        self.models = {
            'rf': RandomForestClassifier(
                n_estimators=2000,
                max_depth=25,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                class_weight='balanced',
                bootstrap=True,
                random_state=42,
                n_jobs=-1
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=1000,
                learning_rate=0.005,
                max_depth=10,
                min_samples_split=2,
                min_samples_leaf=1,
                subsample=0.8,
                max_features='sqrt',
                random_state=42
            ),
            'xgb': xgb.XGBClassifier(
                n_estimators=1000,
                learning_rate=0.005,
                max_depth=10,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=2.33,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1,
                random_state=42,
                n_jobs=-1,
                tree_method='hist'
            ),
            'lgb': lgb.LGBMClassifier(
                n_estimators=1000,
                learning_rate=0.005,
                max_depth=10,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=2.33,
                min_child_samples=20,
                reg_alpha=0.1,
                reg_lambda=1,
                random_state=42,
                n_jobs=-1
            )
        }
        self.neural_net = None
        self.weights = None
        self.feature_selector = None
        self.scaler = StandardScaler()

    def _create_neural_net(self, input_dim):
        model = Sequential([
            Dense(1024, activation='relu', input_dim=input_dim),
            BatchNormalization(),
            Dropout(0.5),

            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),

            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),

            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),

            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.1),

            Dense(32, activation='relu'),
            BatchNormalization(),

            Dense(1, activation='sigmoid')
        ])

        optimizer = Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC']
        )
        return model

    def _perform_feature_selection(self, X, y):
        selector = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.05,
            random_state=42
        )
        self.feature_selector = SelectFromModel(selector, prefit=False, threshold='median')
        self.feature_selector.fit(X, y)
        return self.feature_selector.transform(X)

    def train(self, X, y):
        print("Starting enhanced model training...")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Perform feature selection
        print("Performing feature selection...")
        X_selected = self._perform_feature_selection(X_scaled, y)
        print(f"Selected {X_selected.shape[1]} features out of {X_scaled.shape[1]}")

        # Reset indices to ensure alignment
        X_selected = np.array(X_selected)  # Ensure numpy array format for compatibility
        if isinstance(y, pd.Series):
            y = y.reset_index(drop=True)

        # Train neural network with enhanced configuration
        print("Training neural network...")
        self.neural_net = self._create_neural_net(X_selected.shape[1])

        # Enhanced callbacks
        early_stopping = EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=30,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_auc',
            mode='max',
            factor=0.2,
            patience=10,
            min_lr=1e-6,
            verbose=1
        )

        # Train with k-fold cross-validation
        n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_selected, y), 1):
            print(f"\nTraining fold {fold}/{n_splits}")
            X_train, X_val = X_selected[train_idx], X_selected[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            self.neural_net.fit(
                X_train, y_train,
                epochs=300,
                batch_size=64,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )

        # Train tree-based models with cross-validation
        print("\nTraining tree-based models...")
        predictions = {}
        scores = {}

        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            cv_scores = cross_val_score(
                model,
                X_selected,
                y,
                cv=skf,
                scoring='roc_auc',
                n_jobs=-1
            )
            print(f"{name} CV ROC-AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

            model.fit(X_selected, y)
            pred_proba = model.predict_proba(X_selected)[:, 1]
            predictions[name] = pred_proba
            scores[name] = roc_auc_score(y, pred_proba)

        nn_pred = self.neural_net.predict(X_selected, verbose=0).flatten()
        predictions['nn'] = nn_pred
        scores['nn'] = roc_auc_score(y, nn_pred)

        total_score = sum(scores.values())
        self.weights = {name: score / total_score for name, score in scores.items()}

        print("\nFinal Model Weights based on AUC scores:")
        for name, weight in self.weights.items():
            print(f"{name}: {weight:.3f} (AUC: {scores[name]:.3f})")

        ensemble_pred = np.zeros(X_selected.shape[0])
        for name, pred in predictions.items():
            ensemble_pred += pred * self.weights[name]

        ensemble_accuracy = accuracy_score(y, (ensemble_pred > 0.5).astype(int))
        ensemble_auc = roc_auc_score(y, ensemble_pred)
        ensemble_precision = precision_score(y, (ensemble_pred > 0.5).astype(int))
        ensemble_recall = recall_score(y, (ensemble_pred > 0.5).astype(int))
        ensemble_f1 = f1_score(y, (ensemble_pred > 0.5).astype(int))

        print("\nEnsemble Performance:")
        print(f"Accuracy: {ensemble_accuracy:.3f}")
        print(f"ROC-AUC: {ensemble_auc:.3f}")
        print(f"Precision: {ensemble_precision:.3f}")
        print(f"Recall: {ensemble_recall:.3f}")
        print(f"F1-Score: {ensemble_f1:.3f}")

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        X_selected = self.feature_selector.transform(X_scaled)

        predictions = np.zeros(X_selected.shape[0])
        for name, model in self.models.items():
            predictions += model.predict_proba(X_selected)[:, 1] * self.weights[name]

        nn_pred = self.neural_net.predict(X_selected, verbose=0).flatten()
        predictions += nn_pred * self.weights['nn']

        return predictions

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) > threshold).astype(int)

    def save(self, path):
        model_data = {
            'tree_models': self.models,
            'neural_net_weights': self.neural_net.get_weights(),
            'weights': self.weights,
            'feature_selector': self.feature_selector,
            'scaler': self.scaler
        }
        joblib.dump(model_data, path)

    @staticmethod
    def load(path):
        model = EnsembleModel()
        model_data = joblib.load(path)
        model.models = model_data['tree_models']
        model.weights = model_data['weights']
        model.feature_selector = model_data['feature_selector']
        model.scaler = model_data['scaler']
        return model