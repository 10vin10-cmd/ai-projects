--Given a sales table with columns (order_id, customer_id, order_date, amount), 
--write a query to find each customer's 3-month rolling average order amount.

/*CREATE TABLE IF NOT EXISTS sales (
  order_id INT,
  customer_id INT,
  order_date DATE,
  amount DECIMAL(10, 2)
);
INSERT INTO sales VALUES
(1, 101, '2026-01-15', 100.00),
(2, 101, '2026-02-20', 150.00),
(3, 101, '2026-03-10', 200.00),
(4, 102, '2026-01-05', 50.00),
(5, 102, '2026-02-15', 75.00),
(6, 102, '2026-04-01', 125.00);*/

SELECT order_id, customer_id, order_date, amount, 
    ROUND(AVG(amount) 
    OVER (PARTITION BY customer_id  ORDER BY order_date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2)
    AS rolling_avg 
FROM sales ORDER BY customer_id, order_date;