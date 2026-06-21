import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

def train_random_forest(df):
    """
    Train Random Forest Classifier to predict customer segments.
    Returns accuracy, confusion matrix, and classification report.
    """
    # Aggregate per customer: total amount and frequency
    customer_df = df.groupby('customer_name').agg(
        total_amount=('amount', 'sum'),
        frequency=('amount', 'count')
    ).reset_index()

    # Create labels based on total_amount (supervised learning needs labels)
    # High Value: top 33%, Medium: middle 33%, Low: bottom 33%
    customer_df = customer_df.sort_values('total_amount')
    n = len(customer_df)
    
    labels = []
    for i in range(n):
        if i < n // 3:
            labels.append('Low Value')
        elif i < 2 * n // 3:
            labels.append('Medium Value')
        else:
            labels.append('High Value')
    
    customer_df['segment'] = labels

    # Features and target
    X = customer_df[['total_amount', 'frequency']]
    y = customer_df['segment']

    # Split data: 70% train, 30% test
    class_counts = y.value_counts()
    use_stratify = all(class_counts >= 2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y if use_stratify else None
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,      # 100 decision trees
        max_depth=5,           # Maximum tree depth
        random_state=42,
        min_samples_split=2,
        min_samples_leaf=1
    )
    rf_model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = rf_model.predict(X_test_scaled)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred, labels=['High Value', 'Medium Value', 'Low Value'])
    class_report = classification_report(y_test, y_pred, target_names=['High Value', 'Medium Value', 'Low Value'])

    # Print results
    print("\n" + "="*60)
    print("RANDOM FOREST CLASSIFIER - ACCURACY METRICS")
    print("="*60)
    print(f"Model Accuracy: {accuracy*100:.2f}%")
    print(f"Training Samples: {len(X_train)}")
    print(f"Testing Samples: {len(X_test)}")
    print("="*60)
    print("\nCLASSIFICATION REPORT:")
    print(class_report)
    print("="*60)
    print("\nCONFUSION MATRIX:")
    print(conf_matrix)
    print("="*60 + "\n")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': ['Total Amount', 'Frequency'],
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("FEATURE IMPORTANCE:")
    for idx, row in feature_importance.iterrows():
        print(f"  {row['feature']}: {row['importance']*100:.2f}%")
    print("="*60 + "\n")

    # Save confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['High Value', 'Medium Value', 'Low Value'],
                yticklabels=['High Value', 'Medium Value', 'Low Value'])
    plt.title(f'Confusion Matrix - Accuracy: {accuracy*100:.2f}%')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    
    plot_path = os.path.join('static', 'confusion_matrix.png')
    plt.savefig(plot_path)
    plt.close()

    # Save feature importance plot
    plt.figure(figsize=(8, 5))
    plt.barh(feature_importance['feature'], feature_importance['importance'], color='#4f46e5')
    plt.xlabel('Importance')
    plt.title('Feature Importance in Random Forest Model')
    plt.tight_layout()
    
    importance_path = os.path.join('static', 'feature_importance.png')
    plt.savefig(importance_path)
    plt.close()

    return {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': class_report,
        'feature_importance': feature_importance.to_dict(orient='records'),
        'predictions': customer_df[['customer_name', 'total_amount', 'frequency', 'segment']].to_dict(orient='records')
    }
