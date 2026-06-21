import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

def train_logistic_regression(df):
    """
    Train Logistic Regression to predict sales trends (High/Medium/Low sales months).
    Returns accuracy, predictions, and visualizations.
    """
    # Aggregate by date to get daily sales
    daily_sales = df.groupby('date').agg(
        total_sales=('amount', 'sum'),
        num_orders=('amount', 'count')
    ).reset_index()
    
    # Create features from date
    daily_sales['date'] = pd.to_datetime(daily_sales['date'])
    daily_sales['day_of_week'] = daily_sales['date'].dt.dayofweek  # 0=Monday, 6=Sunday
    daily_sales['day_of_month'] = daily_sales['date'].dt.day
    daily_sales['month'] = daily_sales['date'].dt.month
    
    # Create target: Classify sales as High/Medium/Low
    sales_sorted = daily_sales['total_sales'].sort_values()
    n = len(sales_sorted)
    
    # Define thresholds
    low_threshold = sales_sorted.quantile(0.33)
    high_threshold = sales_sorted.quantile(0.67)
    
    def classify_sales(amount):
        if amount <= low_threshold:
            return 'Low Sales'
        elif amount <= high_threshold:
            return 'Medium Sales'
        else:
            return 'High Sales'
    
    daily_sales['sales_category'] = daily_sales['total_sales'].apply(classify_sales)
    
    # Features and target
    X = daily_sales[['num_orders', 'day_of_week', 'day_of_month', 'month']]
    y = daily_sales['sales_category']
    
    # Split data: 70% train, 30% test
    # Use stratify only if all classes have enough samples
    class_counts = y.value_counts()
    use_stratify = all(class_counts >= 2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y if use_stratify else None
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Logistic Regression
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    )
    lr_model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = lr_model.predict(X_test_scaled)
    y_pred_proba = lr_model.predict_proba(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred, labels=['High Sales', 'Medium Sales', 'Low Sales'])
    class_report = classification_report(y_test, y_pred, target_names=['High Sales', 'Medium Sales', 'Low Sales'])
    
    # Print results
    print("\n" + "="*60)
    print("LOGISTIC REGRESSION - SALES PREDICTION")
    print("="*60)
    print(f"Model Accuracy: {accuracy*100:.2f}%")
    print(f"Training Samples: {len(X_train)}")
    print(f"Testing Samples: {len(X_test)}")
    print(f"Sales Thresholds:")
    print(f"  Low Sales: ≤ ₹{low_threshold:,.0f}")
    print(f"  Medium Sales: ₹{low_threshold:,.0f} - ₹{high_threshold:,.0f}")
    print(f"  High Sales: > ₹{high_threshold:,.0f}")
    print("="*60)
    print("\nCLASSIFICATION REPORT:")
    print(class_report)
    print("="*60)
    print("\nCONFUSION MATRIX:")
    print(conf_matrix)
    print("="*60 + "\n")
    
    # Feature coefficients
    feature_names = ['Number of Orders', 'Day of Week', 'Day of Month', 'Month']
    coefficients = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': lr_model.coef_[0]  # For first class
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    print("FEATURE COEFFICIENTS (Impact on Prediction):")
    for idx, row in coefficients.iterrows():
        print(f"  {row['Feature']}: {row['Coefficient']:.4f}")
    print("="*60 + "\n")
    
    # Save confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens',
                xticklabels=['High Sales', 'Medium Sales', 'Low Sales'],
                yticklabels=['High Sales', 'Medium Sales', 'Low Sales'])
    plt.title(f'Logistic Regression - Accuracy: {accuracy*100:.2f}%')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    
    plot_path = os.path.join('static', 'lr_confusion_matrix.png')
    plt.savefig(plot_path)
    plt.close()
    
    # Save feature coefficients plot
    plt.figure(figsize=(8, 5))
    colors = ['#10b981' if x > 0 else '#ef4444' for x in coefficients['Coefficient']]
    plt.barh(coefficients['Feature'], coefficients['Coefficient'], color=colors)
    plt.xlabel('Coefficient Value')
    plt.title('Feature Impact on Sales Prediction')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    
    coef_path = os.path.join('static', 'lr_coefficients.png')
    plt.savefig(coef_path)
    plt.close()
    
    # Predict next 7 days (example)
    future_predictions = []
    last_date = daily_sales['date'].max()
    
    for i in range(1, 8):
        next_date = last_date + pd.Timedelta(days=i)
        future_features = np.array([[
            daily_sales['num_orders'].mean(),  # Average orders
            next_date.dayofweek,
            next_date.day,
            next_date.month
        ]])
        future_scaled = scaler.transform(future_features)
        prediction = lr_model.predict(future_scaled)[0]
        probability = lr_model.predict_proba(future_scaled)[0]
        
        future_predictions.append({
            'date': next_date.strftime('%Y-%m-%d'),
            'predicted_category': prediction,
            'confidence': max(probability) * 100
        })
    
    print("FUTURE SALES PREDICTIONS (Next 7 Days):")
    for pred in future_predictions:
        print(f"  {pred['date']}: {pred['predicted_category']} (Confidence: {pred['confidence']:.1f}%)")
    print("="*60 + "\n")
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': class_report,
        'coefficients': coefficients.to_dict(orient='records'),
        'future_predictions': future_predictions,
        'thresholds': {
            'low': float(low_threshold),
            'high': float(high_threshold)
        }
    }
