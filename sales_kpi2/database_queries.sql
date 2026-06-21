-- ============================================
-- Sales KPI Analytics - SQL Queries
-- Database: SQLite (sales_kpi.db)
-- ============================================

-- 1. CREATE TABLE
-- Creates the main sales_data table
CREATE TABLE IF NOT EXISTS sales_data (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_name TEXT,
    product       TEXT,
    amount        REAL,
    date          TEXT,
    region        TEXT
);

-- ============================================
-- 2. INSERT SAMPLE DATA
-- Insert sample records into sales_data table
INSERT INTO sales_data (customer_name, product, amount, date, region) VALUES
('Alice Johnson', 'Laptop', 99000.00, '2024-01-05', 'North'),
('Bob Smith', 'Phone', 70000.00, '2024-01-07', 'South'),
('Carol White', 'Tablet', 37000.00, '2024-01-10', 'East'),
('David Brown', 'Laptop', 111000.00, '2024-01-12', 'West'),
('Eve Davis', 'Monitor', 26000.00, '2024-01-15', 'North');

-- ============================================
-- 3. DELETE ALL DATA
-- Clears all records from sales_data table
DELETE FROM sales_data;

-- ============================================
-- 4. KPI QUERIES
-- Get total orders, total sales, and unique customers
SELECT 
    COUNT(*) AS total_orders, 
    COALESCE(SUM(amount), 0) AS total_sales, 
    COUNT(DISTINCT customer_name) AS total_customers 
FROM sales_data 
WHERE 1=1;

-- With date filter
SELECT 
    COUNT(*) AS total_orders, 
    COALESCE(SUM(amount), 0) AS total_sales, 
    COUNT(DISTINCT customer_name) AS total_customers 
FROM sales_data 
WHERE date >= '2024-01-01' 
  AND date <= '2024-03-31';

-- ============================================
-- 5. SALES OVER TIME
-- Get daily sales totals
SELECT 
    date, 
    ROUND(SUM(amount), 2) AS sales 
FROM sales_data 
WHERE 1=1 
GROUP BY date 
ORDER BY date;

-- ============================================
-- 6. SALES BY PRODUCT
-- Get top 10 products by sales
SELECT 
    product, 
    ROUND(SUM(amount), 2) AS sales 
FROM sales_data 
WHERE 1=1 
GROUP BY product 
ORDER BY sales DESC 
LIMIT 10;

-- ============================================
-- 7. SALES BY REGION
-- Get sales totals by region
SELECT 
    region, 
    ROUND(SUM(amount), 2) AS sales 
FROM sales_data 
WHERE 1=1 
GROUP BY region;

-- ============================================
-- 8. ML/CLUSTERING DATA
-- Get customer purchase data for machine learning
SELECT 
    customer_name, 
    amount 
FROM sales_data;

-- ============================================
-- 9. CUSTOMER ANALYSIS
-- Get customer purchase frequency and total spend
SELECT 
    customer_name,
    COUNT(*) AS frequency,
    SUM(amount) AS total_amount,
    AVG(amount) AS avg_amount,
    MIN(date) AS first_purchase,
    MAX(date) AS last_purchase
FROM sales_data
GROUP BY customer_name
ORDER BY total_amount DESC;

-- ============================================
-- 10. PRODUCT PERFORMANCE
-- Get detailed product statistics
SELECT 
    product,
    COUNT(*) AS total_orders,
    SUM(amount) AS total_sales,
    AVG(amount) AS avg_price,
    MIN(amount) AS min_price,
    MAX(amount) AS max_price
FROM sales_data
GROUP BY product
ORDER BY total_sales DESC;

-- ============================================
-- 11. REGION PERFORMANCE
-- Get detailed region statistics
SELECT 
    region,
    COUNT(*) AS total_orders,
    SUM(amount) AS total_sales,
    COUNT(DISTINCT customer_name) AS unique_customers,
    AVG(amount) AS avg_order_value
FROM sales_data
GROUP BY region
ORDER BY total_sales DESC;

-- ============================================
-- 12. MONTHLY SALES TREND
-- Get sales by month
SELECT 
    strftime('%Y-%m', date) AS month,
    COUNT(*) AS orders,
    ROUND(SUM(amount), 2) AS sales
FROM sales_data
GROUP BY month
ORDER BY month;

-- ============================================
-- 13. TOP CUSTOMERS
-- Get top 10 customers by total spend
SELECT 
    customer_name,
    COUNT(*) AS purchases,
    ROUND(SUM(amount), 2) AS total_spent,
    ROUND(AVG(amount), 2) AS avg_per_purchase
FROM sales_data
GROUP BY customer_name
ORDER BY total_spent DESC
LIMIT 10;

-- ============================================
-- 14. VIEW ALL DATA
-- Select all records from sales_data
SELECT * FROM sales_data;

-- ============================================
-- 15. COUNT RECORDS
-- Get total number of records
SELECT COUNT(*) AS total_records FROM sales_data;

-- ============================================
-- 16. DATE RANGE QUERY
-- Get data within specific date range
SELECT * 
FROM sales_data 
WHERE date BETWEEN '2024-01-01' AND '2024-12-31'
ORDER BY date DESC;

-- ============================================
-- 17. SEARCH BY CUSTOMER
-- Find all purchases by specific customer
SELECT * 
FROM sales_data 
WHERE customer_name LIKE '%Alice%'
ORDER BY date DESC;

-- ============================================
-- 18. SEARCH BY PRODUCT
-- Find all sales of specific product
SELECT * 
FROM sales_data 
WHERE product = 'Laptop'
ORDER BY date DESC;

-- ============================================
-- 19. HIGH VALUE TRANSACTIONS
-- Find transactions above certain amount
SELECT * 
FROM sales_data 
WHERE amount > 50000
ORDER BY amount DESC;

-- ============================================
-- 20. DROP TABLE (Use with caution!)
-- Completely removes the sales_data table
-- DROP TABLE IF EXISTS sales_data;
