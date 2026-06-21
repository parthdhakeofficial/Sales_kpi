import sqlite3, pandas as pd, sys, traceback
sys.path.insert(0, 'model')
from kmeans_model import run_kmeans
from logistic_regression_model import train_logistic_regression

conn = sqlite3.connect('sales_kpi.db')
rows = conn.execute('SELECT customer_name, amount FROM sales_data').fetchall()
rows_full = conn.execute('SELECT customer_name, amount, date FROM sales_data').fetchall()
conn.close()

df = pd.DataFrame(rows, columns=['customer_name','amount'])
df_full = pd.DataFrame(rows_full, columns=['customer_name','amount','date'])

print('Records:', len(df))
print('Unique customers:', df['customer_name'].nunique())
print('Unique dates:', df_full['date'].nunique())

try:
    clusters = run_kmeans(df)
    print('K-Means OK, clusters:', len(clusters))
except Exception as e:
    print('K-Means ERROR:', traceback.format_exc())

try:
    lr = train_logistic_regression(df_full)
    print('LR OK, accuracy:', lr['accuracy'])
except Exception as e:
    print('LR ERROR:', traceback.format_exc())
