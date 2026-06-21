import sys
from flask import Flask
app = Flask(__name__, template_folder='templates', static_folder='static')
with app.app_context():
    try:
        t = app.jinja_env.get_template('dashboard.html')
        result = t.render(
            kpi={'total_orders': 5, 'total_sales': 1000.0, 'total_customers': 3, 'avg_order_value': 200.0},
            sales_time=[{'date': '2024-01-01', 'sales': 100}],
            sales_product=[{'product': 'Laptop', 'sales': 500}],
            sales_region=[{'region': 'North', 'sales': 300}],
            start_date='', end_date=''
        )
        print('RENDER OK, length:', len(result))
    except Exception as e:
        import traceback
        print(traceback.format_exc())
