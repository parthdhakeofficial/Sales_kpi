pytop# Sales KPI Analytics Website

## Project Structure
```
sales_kpi/
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── sample_sales.csv        # Sample data for testing
├── templates/
│   ├── index.html          # Home page
│   ├── upload.html         # File upload page
│   ├── dashboard.html      # KPI dashboard
│   └── ml.html             # ML segmentation results
├── static/
│   └── css/style.css       # Stylesheet
├── model/
│   └── kmeans_model.py     # K-Means clustering logic
└── uploads/                # Uploaded CSV files stored here
```

---

## Step 1 – Install Dependencies

Open Command Prompt, navigate to the project folder, then run:

```
cd "C:\Users\Tushar\OneDrive\เอกสาร\Python program\sales_kpi"
pip install -r requirements.txt
```

---

## Step 2 – Setup MySQL

1. Open MySQL Workbench or MySQL command line.
2. Make sure MySQL server is running.
3. Open `app.py` and update the DB_CONFIG section:

```python
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',        # your MySQL username
    'password': '',        # your MySQL password
    'database': 'sales_kpi'
}
```

The app will automatically create the database and table on first run.

---

## Step 3 – Run the Flask App

```
cd "C:\Users\Tushar\OneDrive\เอกสาร\Python program\sales_kpi"
python app.py
```

Open your browser and go to: **http://127.0.0.1:5000**

---

## Step 4 – Using the App

1. Go to **Upload** page → upload `sample_sales.csv` (or your own CSV)
2. Go to **Dashboard** to see KPIs and charts
3. Go to **ML Insights** to see K-Means customer segmentation

---

## CSV Format Required

Your CSV must have these exact column names:

| customer_name | product | amount | date       | region |
|---------------|---------|--------|------------|--------|
| Alice Johnson | Laptop  | 1200   | 2024-01-05 | North  |

---

## Features

- Upload CSV sales data
- KPI cards: Total Sales, Orders, Customers, Avg Order Value
- Charts: Line (sales over time), Bar (by product), Pie (by region)
- Date range filtering
- K-Means clustering: High / Medium / Low value customer segments
- Cluster scatter plot visualization
- MySQL database storage
