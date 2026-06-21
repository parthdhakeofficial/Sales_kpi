import os, sqlite3, sys
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, g, jsonify
from werkzeug.utils import secure_filename
# model/kmeans_model.py
import matplotlib
import seaborn as sns

from model.kmeans_model import run_kmeans
from model.logistic_regression_model import train_logistic_regression

app = Flask(__name__)
app.secret_key = 'sales_kpi_secret'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
DATABASE = os.path.join(os.path.dirname(__file__), 'sales_kpi.db')

# ── DB helpers ────────────────────────────────────────────────
def get_db():
    if '_db' not in g:
        g._db = sqlite3.connect(DATABASE)
        g._db.row_factory = sqlite3.Row
    return g._db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop('_db', None)
    if db: db.close()

def init_db():
    """Ensure the database file exists."""
    conn = sqlite3.connect(DATABASE)
    conn.commit(); conn.close()

def save_df(df):
    """Drop old data and insert new dataframe with ALL its columns into DB."""
    conn = sqlite3.connect(DATABASE)
    # Clean column names: lowercase, underscores, no special chars
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace(' ', '_', regex=False)
                  .str.replace(r'[^a-z0-9_]', '', regex=True))
    # Remove fully empty columns
    df = df.dropna(axis=1, how='all')
    # Replace old table entirely
    df.to_sql('sales_data', conn, if_exists='replace', index=False)
    conn.commit(); conn.close()

def _get_columns_info(db):
    """Auto-detect column types from actual data in the table."""
    try:
        row = db.execute("SELECT * FROM sales_data LIMIT 1").fetchone()
        if not row:
            return [], [], [], []
        cols = list(row.keys())
        # Remove internal sqlite rowid if present
        cols = [c for c in cols if c not in ('index',)]

        numeric = []
        categorical = []
        temporal = []

        for col in cols:
            # Sample up to 50 values to detect type
            samples = [r[col] for r in db.execute(f'SELECT DISTINCT "{col}" FROM sales_data WHERE "{col}" IS NOT NULL LIMIT 50').fetchall()]
            if not samples:
                categorical.append(col)
                continue

            # Check if temporal (looks like date)
            is_date = False
            if all(isinstance(v, str) for v in samples):
                try:
                    # pd.to_datetime(samples[:10])
                    pd.to_datetime(
                    samples[:10],
                    format='%Y-%m-%d',
                    errors='coerce'
                )
                    is_date = True
                except Exception:
                    pass
            if is_date:
                temporal.append(col)
                continue

            # Check if numeric
            is_numeric = True
            for v in samples[:20]:
                if v is None:
                    continue
                if isinstance(v, (int, float)):
                    continue
                try:
                    float(str(v).replace(',', ''))
                except (ValueError, TypeError):
                    is_numeric = False
                    break

            if is_numeric:
                numeric.append(col)
            else:
                categorical.append(col)

        return cols, categorical, numeric, temporal
    except Exception:
        return [], [], [], []


# ── ML Models - Train once at startup ────────────────────────
_lr_results = None


# ── ROUTES ────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load-sample')
def load_sample():
    flash('Sample data was removed. Please upload your CSV instead.', 'info')
    return redirect(url_for('upload'))


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            flash('No file selected.', 'error')
            return redirect(request.url)
        if not file.filename.lower().endswith('.csv'):
            flash('Only CSV files are allowed.', 'error')
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Try multiple encodings
            df = None
            for enc in ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']:
                try:
                    df = pd.read_csv(filepath, encoding=enc)
                    break
                except Exception:
                    continue

            if df is None or df.empty:
                flash('Could not read the CSV file. Please check the format.', 'error')
                return redirect(request.url)

            # Clean up: try to convert date-like columns
            for col in df.columns:
                col_lower = col.strip().lower()
                if any(kw in col_lower for kw in ['date', 'time', 'timestamp', 'created']):
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%Y-%m-%d')
                    except Exception:
                        pass
            # Try to convert numeric-looking columns
            for col in df.columns:
                if df[col].dtype == object:
                    try:
                        converted = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')
                        if converted.notna().sum() > len(df) * 0.5:  # >50% are numbers
                            df[col] = converted
                    except Exception:
                        pass

            save_df(df)
            flash(f'{len(df)} records × {len(df.columns)} columns uploaded successfully!', 'success')
            return redirect(url_for('dashboard'))

        except Exception as e:
            flash('Error processing file: ' + str(e), 'error')
            return redirect(request.url)

    return render_template('upload.html')

@app.route('/api/columns')
def api_columns():
    """Return column names available in the dataset with auto-detected types."""
    try:
        db = get_db()
        cols, categorical, numeric, temporal = _get_columns_info(db)
        return jsonify(columns=cols, categorical=categorical, numeric=numeric, temporal=temporal)
    except Exception:
        return jsonify(columns=[], categorical=[], numeric=[], temporal=[])

@app.route('/api/chart-data')
def api_chart_data():
    """Return aggregated data for a user-defined chart."""
    x_axis      = request.args.get('x_axis', '')
    y_axis      = request.args.get('y_axis', '')
    aggregation = request.args.get('aggregation', 'SUM').upper()
    start_date  = request.args.get('start_date', '')
    end_date    = request.args.get('end_date', '')
    date_col    = request.args.get('date_col', '')
    # Optional single slice filter (Power BI-like filtering)
    filter_col  = request.args.get('filter_col', '')
    filter_value = request.args.get('filter_value', '')
    limit       = request.args.get('limit', '50')


    if aggregation not in ('SUM','AVG','COUNT','MIN','MAX'):
        aggregation = 'SUM'

    try:
        db = get_db()
        # Validate columns exist
        cols, categorical, numeric, temporal = _get_columns_info(db)
        if x_axis not in cols:
            return jsonify(error=f'Invalid x_axis: {x_axis}'), 400

        where  = "WHERE 1=1"
        params = []
        # Apply date filter if a date column is identified
        if date_col and date_col in temporal:
            if start_date: where += f' AND "{date_col}" >= ?'; params.append(start_date)
            if end_date:   where += f' AND "{date_col}" <= ?'; params.append(end_date)

        # Apply optional slice filter
        if filter_col and filter_col in cols and filter_value != '':
            where += f' AND "{filter_col}" = ?'
            params.append(filter_value)


        # For temporal x-axis, group by month
        if x_axis in temporal:
            x_expr = f"strftime('%Y-%m', \"{x_axis}\")"
            order  = "ORDER BY grp ASC"
        else:
            x_expr = f'"{x_axis}"'
            order  = "ORDER BY val DESC"

        if aggregation == 'COUNT':
            sql = f'SELECT {x_expr} grp, COUNT(*) val FROM sales_data {where} GROUP BY grp {order} LIMIT ?'
        else:
            sql = f'SELECT {x_expr} grp, ROUND({aggregation}("{y_axis}"),2) val FROM sales_data {where} GROUP BY grp {order} LIMIT ?'

        params.append(int(limit))
        rows = [dict(r) for r in db.execute(sql, params)]
        labels = [str(r['grp']) if r['grp'] is not None else 'N/A' for r in rows]
        values = [r['val'] for r in rows]

        return jsonify(labels=labels, values=values, x_axis=x_axis, y_axis=y_axis, aggregation=aggregation)
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route('/api/insights')
def api_insights():
    """Auto-generate Power BI-style text insights from the data — works with ANY dataset."""
    try:
        db = get_db()
        cols, categorical, numeric, temporal = _get_columns_info(db)
        if not cols:
            return jsonify(insights=[])

        insights = []
        total_rows = db.execute("SELECT COUNT(*) c FROM sales_data").fetchone()['c']
        insights.append({"icon": "🔢", "type": "stat", "title": "Dataset Size",
            "text": f"Your dataset contains {total_rows:,} records across {len(cols)} columns."})

        # Insights for each numeric column
        for ncol in numeric[:3]:  # limit to first 3 numeric columns
            r = db.execute(f'SELECT ROUND(SUM("{ncol}"),2) s, ROUND(AVG("{ncol}"),2) a, ROUND(MIN("{ncol}"),2) mn, ROUND(MAX("{ncol}"),2) mx FROM sales_data').fetchone()
            if r and r['s'] is not None:
                pretty = ncol.replace('_', ' ').title()
                insights.append({"icon": "📊", "type": "stat", "title": f"{pretty} Overview",
                    "text": f"Total: ₹{r['s']:,.0f} | Avg: ₹{r['a']:,.0f} | Range: ₹{r['mn']:,.0f} – ₹{r['mx']:,.0f}"})

        # Top category for each categorical column + first numeric
        if numeric:
            main_num = numeric[0]
            pretty_num = main_num.replace('_', ' ').title()
            for ccol in categorical[:3]:
                r = db.execute(f'SELECT "{ccol}" cat, ROUND(SUM("{main_num}"),2) total FROM sales_data GROUP BY "{ccol}" ORDER BY total DESC LIMIT 1').fetchone()
                if r and r['cat']:
                    pretty_cat = ccol.replace('_', ' ').title()
                    insights.append({"icon": "🏆", "type": "top", "title": f"Top {pretty_cat}",
                        "text": f'"{r["cat"]}" leads with ₹{r["total"]:,.0f} total {pretty_num}.'})

            # Bottom category for first categorical
            if categorical:
                ccol = categorical[0]
                r = db.execute(f'SELECT "{ccol}" cat, ROUND(SUM("{main_num}"),2) total FROM sales_data GROUP BY "{ccol}" ORDER BY total ASC LIMIT 1').fetchone()
                if r and r['cat']:
                    pretty_cat = ccol.replace('_', ' ').title()
                    insights.append({"icon": "📉", "type": "low", "title": f"Lowest {pretty_cat}",
                        "text": f'"{r["cat"]}" has the lowest {pretty_num} at ₹{r["total"]:,.0f}.'})

        # Unique counts for categorical columns
        for ccol in categorical[:4]:
            r = db.execute(f'SELECT COUNT(DISTINCT "{ccol}") cnt FROM sales_data').fetchone()
            if r:
                pretty = ccol.replace('_', ' ').title()
                insights.append({"icon": "📦", "type": "stat", "title": f"{pretty} Diversity",
                    "text": f"{r['cnt']} unique values found in the {pretty} column."})

        # Temporal trend
        if temporal and numeric:
            tcol = temporal[0]
            ncol = numeric[0]
            months = [dict(r) for r in db.execute(
                f'SELECT strftime(\'%Y-%m\', "{tcol}") month, SUM("{ncol}") total '
                f'FROM sales_data WHERE "{tcol}" IS NOT NULL GROUP BY month ORDER BY month')]
            if len(months) >= 2:
                last = months[-1]['total']
                prev = months[-2]['total']
                change = round(((last - prev) / prev) * 100, 1) if prev else 0
                direction = "up" if change > 0 else "down"
                insights.append({"icon": "📈" if change > 0 else "📉", "type": "trend",
                    "title": "Monthly Trend",
                    "text": f"{ncol.replace('_',' ').title()} went {direction} {abs(change)}% from {months[-2]['month']} to {months[-1]['month']}."})

        return jsonify(insights=insights)
    except Exception as e:
        return jsonify(insights=[], error=str(e))

@app.route('/api/pivot-table')
def api_pivot_table():
    """Return a pivot table matrix from ANY dataset.

    Query params:
      - row_dim (required)
      - col_dim (optional; if empty => no column split)
      - value_measure (required; numeric)
      - agg (SUM|AVG|COUNT|MIN|MAX)

    Optional slicer / date filters:
      - filter_col + filter_value
      - start_date + end_date (applies only if a temporal column exists matching date_col='date_col' OR first temporal)

    Output:
      {
        row_labels: [],
        col_labels: [],
        matrix: [[...]],
        row_dim, col_dim, value_measure, agg
      }
    """
    try:
        row_dim = request.args.get('row_dim', '').strip()
        col_dim = request.args.get('col_dim', '').strip()
        value_measure = request.args.get('value_measure', '').strip()
        agg = request.args.get('agg', 'SUM').upper()

        if not row_dim:
            return jsonify(error='row_dim is required'), 400
        if agg not in ('SUM','AVG','COUNT','MIN','MAX'):
            agg = 'SUM'

        db = get_db()
        cols, categorical, numeric, temporal = _get_columns_info(db)
        if not cols:
            return jsonify(error='No data available'), 400

        if row_dim not in cols:
            return jsonify(error=f'Invalid row_dim: {row_dim}'), 400
        if col_dim and col_dim not in cols:
            return jsonify(error=f'Invalid col_dim: {col_dim}'), 400

        # COUNT can work without numeric measure, but for simplicity keep value_measure required.
        if value_measure and value_measure not in numeric:
            return jsonify(error=f'Invalid value_measure: {value_measure}'), 400

        if not value_measure:
            # fall back to first numeric if present
            if numeric:
                value_measure = numeric[0]
            else:
                return jsonify(error='No numeric measures available'), 400

        # Date filtering: use explicit date_col if provided, else first temporal
        start_date = request.args.get('start_date', '')
        end_date = request.args.get('end_date', '')
        date_col = request.args.get('date_col', '')
        if not date_col and temporal:
            date_col = temporal[0]

        filter_col = request.args.get('filter_col', '').strip()
        filter_value = request.args.get('filter_value', '').strip()

        where = 'WHERE 1=1'
        params = []

        if date_col and date_col in temporal:
            if start_date:
                where += f' AND "{date_col}" >= ?'
                params.append(start_date)
            if end_date:
                where += f' AND "{date_col}" <= ?'
                params.append(end_date)

        if filter_col and filter_col in cols and filter_value != '':
            where += f' AND "{filter_col}" = ?'
            params.append(filter_value)

        # Build distinct labels
        row_vals = [r['v'] for r in db.execute(
            f'SELECT DISTINCT "{row_dim}" v FROM sales_data {where} ORDER BY v'
        , params).fetchall()]
        row_labels = [str(v) if v is not None else 'N/A' for v in row_vals]

        if col_dim:
            col_vals = [r['v'] for r in db.execute(
                f'SELECT DISTINCT "{col_dim}" v FROM sales_data {where} ORDER BY v'
            , params).fetchall()]
            col_labels = [str(v) if v is not None else 'N/A' for v in col_vals]
        else:
            col_vals = []
            col_labels = []

        # Pre-fill matrix with zeros/nulls
        # We'll compute as numeric aggregation except COUNT.
        matrix = []
        for _ in row_labels:
            matrix.append([0 for __ in col_labels])

        # Query aggregated values
        if col_dim:
            if agg == 'COUNT':
                sql = f'SELECT "{row_dim}" r, "{col_dim}" c, COUNT(*) val FROM sales_data {where} GROUP BY r, c'
            else:
                sql = f'SELECT "{row_dim}" r, "{col_dim}" c, ROUND({agg}("{value_measure}"),2) val FROM sales_data {where} GROUP BY r, c'
            rows = [dict(r) for r in db.execute(sql, params).fetchall()]
            row_index = {str(v) if v is not None else 'N/A': i for i, v in enumerate(row_vals)}
            col_index = {str(v) if v is not None else 'N/A': j for j, v in enumerate(col_vals)}
            for r in rows:
                rr = str(r['r']) if r['r'] is not None else 'N/A'
                cc = str(r['c']) if r['c'] is not None else 'N/A'
                if rr in row_index and cc in col_index:
                    matrix[row_index[rr]][col_index[cc]] = r['val'] if r['val'] is not None else 0
        else:
            # single-column pivot (like table by rows only)
            if agg == 'COUNT':
                sql = f'SELECT "{row_dim}" r, COUNT(*) val FROM sales_data {where} GROUP BY r'
            else:
                sql = f'SELECT "{row_dim}" r, ROUND({agg}("{value_measure}"),2) val FROM sales_data {where} GROUP BY r'
            rows = [dict(r) for r in db.execute(sql, params).fetchall()]
            row_index = {str(v) if v is not None else 'N/A': i for i, v in enumerate(row_vals)}
            # matrix has no columns; represent as [value] per row with empty col_labels
            for r in rows:
                rr = str(r['r']) if r['r'] is not None else 'N/A'
                if rr in row_index:
                    matrix[row_index[rr]] = [r['val'] if r['val'] is not None else 0]

        # If no col_dim, keep matrix as Nx1 for rendering convenience
        if not col_dim:
            col_labels = ['Total']
            for i in range(len(row_labels)):
                matrix[i] = [matrix[i][0] if matrix[i] else 0]

        return jsonify(
            row_labels=row_labels,
            col_labels=col_labels,
            matrix=matrix,
            row_dim=row_dim,
            col_dim=col_dim,
            value_measure=value_measure,
            agg=agg
        )
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route('/api/summary-stats')
def api_summary_stats():
    """Return descriptive statistics for a numeric column (auto-detected or specified)."""
    col = request.args.get('column', '')
    try:
        db = get_db()
        cols, categorical, numeric, temporal = _get_columns_info(db)
        if not col and numeric:
            col = numeric[0]
        if col not in numeric:
            return jsonify(stats={}, numeric_columns=numeric)

        rows = db.execute(f'SELECT "{col}" val FROM sales_data WHERE "{col}" IS NOT NULL').fetchall()
        if not rows:
            return jsonify(stats={}, column=col, numeric_columns=numeric)
        amounts = sorted([r['val'] for r in rows if r['val'] is not None])
        amounts = [float(v) for v in amounts]
        n = len(amounts)
        if n == 0:
            return jsonify(stats={}, column=col, numeric_columns=numeric)
        total = sum(amounts)
        mean = round(total / n, 2)
        min_val = amounts[0]
        max_val = amounts[-1]
        median = amounts[n // 2] if n % 2 != 0 else round((amounts[n//2 - 1] + amounts[n//2]) / 2, 2)
        variance = sum((x - mean) ** 2 for x in amounts) / n
        std_dev = round(variance ** 0.5, 2)
        q1 = amounts[n // 4]
        q3 = amounts[(3 * n) // 4]

        return jsonify(stats={
            "count": n, "sum": round(total, 2), "mean": mean,
            "median": median, "min": min_val, "max": max_val,
            "std_dev": std_dev, "q1": q1, "q3": q3,
            "range": round(max_val - min_val, 2)
        }, column=col, numeric_columns=numeric)
    except Exception as e:
        return jsonify(stats={}, error=str(e))

@app.route('/api/data-table')
def api_data_table():
    """Return paginated, sortable, searchable raw data — works with ANY columns."""
    page     = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 15))
    sort_by  = request.args.get('sort_by', '')
    sort_dir = request.args.get('sort_dir', 'DESC')
    search   = request.args.get('search', '').strip()
    # Optional single slice filter
    filter_col  = request.args.get('filter_col', '')
    filter_value = request.args.get('filter_value', '')


    if sort_dir.upper() not in ('ASC','DESC'): sort_dir = 'DESC'

    try:
        db = get_db()
        cols, categorical, numeric, temporal = _get_columns_info(db)
        if not cols:
            return jsonify(data=[], total=0, columns=[])

        # Validate sort column
        if sort_by not in cols:
            sort_by = cols[0]

        where = "WHERE 1=1"
        params = []
        if search:
            # Search across all text/categorical columns
            search_cols = categorical + temporal
            if search_cols:
                clauses = [f'CAST("{c}" AS TEXT) LIKE ?' for c in search_cols]
                where += " AND (" + " OR ".join(clauses) + ")"
                params.extend([f'%{search}%'] * len(search_cols))

        # Apply optional slice filter
        if filter_col and filter_col in cols and filter_value != '':
            where += f' AND "{filter_col}" = ?'
            params.append(filter_value)


        total = db.execute(f"SELECT COUNT(*) c FROM sales_data {where}", params).fetchone()['c']
        offset = (page - 1) * per_page
        col_list = ', '.join(f'"{c}"' for c in cols)
        rows = [dict(r) for r in db.execute(
            f'SELECT {col_list} FROM sales_data {where} ORDER BY "{sort_by}" {sort_dir} LIMIT ? OFFSET ?',
            params + [per_page, offset]
        )]

        return jsonify(data=rows, total=total, page=page, per_page=per_page,
                       total_pages=(total + per_page - 1) // per_page,
                       columns=cols, numeric=numeric, categorical=categorical, temporal=temporal)
    except Exception as e:
        return jsonify(data=[], total=0, error=str(e))

@app.route('/dashboard')
def dashboard():
    start_date = request.args.get('start_date', '')
    end_date   = request.args.get('end_date', '')
    try:
        db = get_db()
        cols, categorical, numeric, temporal = _get_columns_info(db)

        if not cols:
            return render_template('dashboard.html', kpi=None, columns=[],
                categorical=[], numeric=[], temporal=[],
                start_date='', end_date='')

        # Build dynamic KPI cards
        kpi = {}
        kpi['total_records'] = db.execute("SELECT COUNT(*) c FROM sales_data").fetchone()['c']
        kpi['total_columns'] = len(cols)

        # For each numeric column, compute sum and avg
        kpi['numeric_stats'] = {}
        for ncol in numeric[:4]:  # up to 4 numeric columns for KPI cards
            r = db.execute(f'SELECT ROUND(SUM("{ncol}"),2) s, ROUND(AVG("{ncol}"),2) a FROM sales_data').fetchone()
            kpi['numeric_stats'][ncol] = {'sum': r['s'] or 0, 'avg': r['a'] or 0}

        # For each categorical column, count unique values
        kpi['category_counts'] = {}
        for ccol in categorical[:4]:
            r = db.execute(f'SELECT COUNT(DISTINCT "{ccol}") c FROM sales_data').fetchone()
            kpi['category_counts'][ccol] = r['c'] or 0

        return render_template('dashboard.html',
            kpi=kpi, columns=cols,
            categorical=categorical, numeric=numeric, temporal=temporal,
            start_date=start_date, end_date=end_date)

    except Exception as e:
        # Avoid hard dashboard failure when the dataset is missing/empty.
        flash('Dashboard error: ' + str(e), 'error')
        return render_template(
            'dashboard.html',
            kpi={'total_records': 0, 'total_columns': 0, 'numeric_stats': {}, 'category_counts': {}},
            columns=[], categorical=[], numeric=[], temporal=[],
            start_date=start_date,
            end_date=end_date
        )


@app.route('/ml')
def ml_results():
    try:
        db = get_db()
        cols, categorical, numeric, temporal = _get_columns_info(db)

        # ML needs at least one categorical and one numeric column
        if not numeric or not categorical:
            flash('ML analysis requires at least one categorical and one numeric column.', 'error')
            return redirect(url_for('upload'))

        cat_col = categorical[0]
        num_col = numeric[0]

        rows = db.execute(f'SELECT "{cat_col}", "{num_col}" FROM sales_data').fetchall()
        if not rows:
            flash('No data found. Please upload a CSV or load sample data first.', 'error')
            return redirect(url_for('upload'))
        df = pd.DataFrame(rows, columns=['customer_name', 'amount'])

        # For LR, try to find a date column too
        date_col = temporal[0] if temporal else None
        if date_col:
            rows_full = db.execute(f'SELECT "{cat_col}", "{num_col}", "{date_col}" FROM sales_data').fetchall()
            df_full = pd.DataFrame(rows_full, columns=['customer_name', 'amount', 'date'])
        else:
            df_full = None

        # Run K-Means
        clusters = run_kmeans(df)

        # Logistic Regression (only if date column exists)
        global _lr_results
        lr_acc = None
        if df_full is not None:
            if _lr_results is None:
                _lr_results = train_logistic_regression(df_full)
            lr_acc = _lr_results['accuracy'] if _lr_results else None

        return render_template('ml.html',
                             clusters=clusters,
                             rf_accuracy=None,
                             rf_results=None,
                             lr_accuracy=lr_acc,
                             lr_results=_lr_results)
    except Exception as e:
        flash('ML error: ' + str(e), 'error')
        return render_template('ml.html', clusters=[], rf_accuracy=None, rf_results=None, lr_accuracy=None, lr_results=None)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    init_db()
    print("Server starting at http://127.0.0.1:5000")
    app.run(debug=True)
