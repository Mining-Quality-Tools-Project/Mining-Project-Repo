from data_load import load_data
from preprocessor import DropoutPreprocessor
from models import EnsembleModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from config import MODEL_CONFIG, MODEL_PATH

def train_model():
    # Load data
    print("Loading data...")
    df = load_data()
    
    # Preprocess data
    print("Preprocessing data...")
    preprocessor = DropoutPreprocessor()
    X, y = preprocessor.fit_transform(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=MODEL_CONFIG['test_size'],
        random_state=MODEL_CONFIG['random_state']
    )
    
    # Train ensemble model
    print("Training ensemble model...")
    model = EnsembleModel()
    model.train(X_train, y_train)
    
    # Evaluate model
    print("\nModel Evaluation:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Save model and preprocessor
    print("\nSaving model...")
    joblib.dump({
        'model': model,
        'preprocessor': preprocessor
    }, MODEL_PATH)
    
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()