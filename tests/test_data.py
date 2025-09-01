"""
Test data fixtures for NL2Q Agent comprehensive testing
This file contains realistic sample data for various test scenarios
"""

# Sample customer data
CUSTOMER_DATA = [
    {
        "id": 1,
        "first_name": "John",
        "last_name": "Doe",
        "email": "john.doe@example.com",
        "phone": "+1-555-0123",
        "city": "New York",
        "state": "NY",
        "country": "USA",
        "registration_date": "2023-01-15",
        "is_active": True,
        "customer_type": "Premium"
    },
    {
        "id": 2,
        "first_name": "Jane",
        "last_name": "Smith",
        "email": "jane.smith@example.com",
        "phone": "+1-555-0124",
        "city": "Los Angeles",
        "state": "CA",
        "country": "USA",
        "registration_date": "2023-02-20",
        "is_active": True,
        "customer_type": "Standard"
    },
    {
        "id": 3,
        "first_name": "Bob",
        "last_name": "Johnson",
        "email": "bob.johnson@example.com",
        "phone": "+1-555-0125",
        "city": "Chicago",
        "state": "IL",
        "country": "USA",
        "registration_date": "2023-03-10",
        "is_active": False,
        "customer_type": "Basic"
    },
    {
        "id": 4,
        "first_name": "Alice",
        "last_name": "Williams",
        "email": "alice.williams@example.com",
        "phone": "+1-555-0126",
        "city": "Houston",
        "state": "TX",
        "country": "USA",
        "registration_date": "2023-04-05",
        "is_active": True,
        "customer_type": "Premium"
    },
    {
        "id": 5,
        "first_name": "Charlie",
        "last_name": "Brown",
        "email": "charlie.brown@example.com",
        "phone": "+1-555-0127",
        "city": "Phoenix",
        "state": "AZ",
        "country": "USA",
        "registration_date": "2023-05-12",
        "is_active": True,
        "customer_type": "Standard"
    }
]

# Sample product data
PRODUCT_DATA = [
    {
        "id": 1,
        "name": "Wireless Bluetooth Headphones",
        "category": "Electronics",
        "price": 99.99,
        "cost": 60.00,
        "stock_quantity": 150,
        "supplier": "TechCorp",
        "rating": 4.5,
        "is_available": True
    },
    {
        "id": 2,
        "name": "Ergonomic Office Chair",
        "category": "Furniture",
        "price": 299.99,
        "cost": 150.00,
        "stock_quantity": 75,
        "supplier": "OfficePlus",
        "rating": 4.2,
        "is_available": True
    },
    {
        "id": 3,
        "name": "Organic Coffee Beans",
        "category": "Food & Beverage",
        "price": 24.99,
        "cost": 12.00,
        "stock_quantity": 200,
        "supplier": "GreenFarm",
        "rating": 4.8,
        "is_available": True
    },
    {
        "id": 4,
        "name": "Yoga Mat",
        "category": "Sports & Fitness",
        "price": 39.99,
        "cost": 15.00,
        "stock_quantity": 120,
        "supplier": "FitLife",
        "rating": 4.3,
        "is_available": False
    },
    {
        "id": 5,
        "name": "Smart Watch",
        "category": "Electronics",
        "price": 199.99,
        "cost": 100.00,
        "stock_quantity": 90,
        "supplier": "TechCorp",
        "rating": 4.6,
        "is_available": True
    }
]

# Sample order data
ORDER_DATA = [
    {
        "id": 1,
        "customer_id": 1,
        "order_date": "2023-09-01",
        "total_amount": 139.98,
        "status": "completed",
        "payment_method": "credit_card",
        "shipping_address": "123 Main St, New York, NY 10001"
    },
    {
        "id": 2,
        "customer_id": 2,
        "order_date": "2023-09-02",
        "total_amount": 299.99,
        "status": "processing",
        "payment_method": "paypal",
        "shipping_address": "456 Oak Ave, Los Angeles, CA 90210"
    },
    {
        "id": 3,
        "customer_id": 1,
        "order_date": "2023-09-03",
        "total_amount": 64.97,
        "status": "completed",
        "payment_method": "credit_card",
        "shipping_address": "123 Main St, New York, NY 10001"
    },
    {
        "id": 4,
        "customer_id": 4,
        "order_date": "2023-09-04",
        "total_amount": 199.99,
        "status": "shipped",
        "payment_method": "debit_card",
        "shipping_address": "789 Pine Rd, Houston, TX 77001"
    },
    {
        "id": 5,
        "customer_id": 5,
        "order_date": "2023-09-05",
        "total_amount": 39.99,
        "status": "cancelled",
        "payment_method": "credit_card",
        "shipping_address": "321 Elm St, Phoenix, AZ 85001"
    }
]

# Sample order items data
ORDER_ITEM_DATA = [
    {
        "id": 1,
        "order_id": 1,
        "product_id": 1,
        "quantity": 1,
        "unit_price": 99.99,
        "total_price": 99.99
    },
    {
        "id": 2,
        "order_id": 1,
        "product_id": 3,
        "quantity": 1,
        "unit_price": 24.99,
        "total_price": 24.99
    },
    {
        "id": 3,
        "order_id": 2,
        "product_id": 2,
        "quantity": 1,
        "unit_price": 299.99,
        "total_price": 299.99
    },
    {
        "id": 4,
        "order_id": 3,
        "product_id": 3,
        "quantity": 2,
        "unit_price": 24.99,
        "total_price": 49.98
    },
    {
        "id": 5,
        "order_id": 3,
        "product_id": 4,
        "quantity": 1,
        "unit_price": 39.99,
        "total_price": 39.99
    },
    {
        "id": 6,
        "order_id": 4,
        "product_id": 5,
        "quantity": 1,
        "unit_price": 199.99,
        "total_price": 199.99
    },
    {
        "id": 7,
        "order_id": 5,
        "product_id": 4,
        "quantity": 1,
        "unit_price": 39.99,
        "total_price": 39.99
    }
]

# Sample sales data for analytics
SALES_DATA = [
    {
        "id": 1,
        "date": "2023-09-01",
        "product_id": 1,
        "customer_id": 1,
        "quantity": 2,
        "unit_price": 99.99,
        "total_amount": 199.98,
        "discount": 0.00,
        "region": "Northeast"
    },
    {
        "id": 2,
        "date": "2023-09-01",
        "product_id": 2,
        "customer_id": 2,
        "quantity": 1,
        "unit_price": 299.99,
        "total_amount": 299.99,
        "discount": 10.00,
        "region": "West"
    },
    {
        "id": 3,
        "date": "2023-09-02",
        "product_id": 3,
        "customer_id": 3,
        "quantity": 3,
        "unit_price": 24.99,
        "total_amount": 74.97,
        "discount": 5.00,
        "region": "Midwest"
    },
    {
        "id": 4,
        "date": "2023-09-02",
        "product_id": 1,
        "customer_id": 4,
        "quantity": 1,
        "unit_price": 99.99,
        "total_amount": 99.99,
        "discount": 0.00,
        "region": "South"
    },
    {
        "id": 5,
        "date": "2023-09-03",
        "product_id": 5,
        "customer_id": 5,
        "quantity": 1,
        "unit_price": 199.99,
        "total_amount": 199.99,
        "discount": 15.00,
        "region": "West"
    }
]

# Sample employee data for HR queries
EMPLOYEE_DATA = [
    {
        "id": 1,
        "first_name": "Sarah",
        "last_name": "Johnson",
        "email": "sarah.johnson@company.com",
        "department": "Engineering",
        "position": "Senior Software Engineer",
        "salary": 120000,
        "hire_date": "2020-03-15",
        "manager_id": None,
        "gender": "F",
        "ethnicity": "Caucasian"
    },
    {
        "id": 2,
        "first_name": "Michael",
        "last_name": "Chen",
        "email": "michael.chen@company.com",
        "department": "Engineering",
        "position": "Software Engineer",
        "salary": 95000,
        "hire_date": "2021-06-01",
        "manager_id": 1,
        "gender": "M",
        "ethnicity": "Asian"
    },
    {
        "id": 3,
        "first_name": "Emily",
        "last_name": "Rodriguez",
        "email": "emily.rodriguez@company.com",
        "department": "Marketing",
        "position": "Marketing Manager",
        "salary": 85000,
        "hire_date": "2019-11-20",
        "manager_id": None,
        "gender": "F",
        "ethnicity": "Hispanic"
    },
    {
        "id": 4,
        "first_name": "David",
        "last_name": "Williams",
        "email": "david.williams@company.com",
        "department": "Sales",
        "position": "Sales Representative",
        "salary": 65000,
        "hire_date": "2022-01-10",
        "manager_id": None,
        "gender": "M",
        "ethnicity": "African American"
    },
    {
        "id": 5,
        "first_name": "Lisa",
        "last_name": "Thompson",
        "email": "lisa.thompson@company.com",
        "department": "HR",
        "position": "HR Specialist",
        "salary": 70000,
        "hire_date": "2020-08-15",
        "manager_id": None,
        "gender": "F",
        "ethnicity": "Caucasian"
    }
]

# Sample database schema for testing
SAMPLE_SCHEMA = {
    "customers": {
        "id": "INTEGER PRIMARY KEY",
        "first_name": "TEXT NOT NULL",
        "last_name": "TEXT NOT NULL",
        "email": "TEXT UNIQUE",
        "phone": "TEXT",
        "city": "TEXT",
        "state": "TEXT",
        "country": "TEXT",
        "registration_date": "DATE",
        "is_active": "BOOLEAN",
        "customer_type": "TEXT"
    },
    "products": {
        "id": "INTEGER PRIMARY KEY",
        "name": "TEXT NOT NULL",
        "category": "TEXT",
        "price": "REAL",
        "cost": "REAL",
        "stock_quantity": "INTEGER",
        "supplier": "TEXT",
        "rating": "REAL",
        "is_available": "BOOLEAN"
    },
    "orders": {
        "id": "INTEGER PRIMARY KEY",
        "customer_id": "INTEGER",
        "order_date": "DATE",
        "total_amount": "REAL",
        "status": "TEXT",
        "payment_method": "TEXT",
        "shipping_address": "TEXT",
        "FOREIGN KEY (customer_id)": "REFERENCES customers(id)"
    },
    "order_items": {
        "id": "INTEGER PRIMARY KEY",
        "order_id": "INTEGER",
        "product_id": "INTEGER",
        "quantity": "INTEGER",
        "unit_price": "REAL",
        "total_price": "REAL",
        "FOREIGN KEY (order_id)": "REFERENCES orders(id)",
        "FOREIGN KEY (product_id)": "REFERENCES products(id)"
    },
    "sales": {
        "id": "INTEGER PRIMARY KEY",
        "date": "DATE",
        "product_id": "INTEGER",
        "customer_id": "INTEGER",
        "quantity": "INTEGER",
        "unit_price": "REAL",
        "total_amount": "REAL",
        "discount": "REAL",
        "region": "TEXT",
        "FOREIGN KEY (product_id)": "REFERENCES products(id)",
        "FOREIGN KEY (customer_id)": "REFERENCES customers(id)"
    },
    "employees": {
        "id": "INTEGER PRIMARY KEY",
        "first_name": "TEXT NOT NULL",
        "last_name": "TEXT NOT NULL",
        "email": "TEXT UNIQUE",
        "department": "TEXT",
        "position": "TEXT",
        "salary": "REAL",
        "hire_date": "DATE",
        "manager_id": "INTEGER",
        "gender": "TEXT",
        "ethnicity": "TEXT",
        "FOREIGN KEY (manager_id)": "REFERENCES employees(id)"
    }
}

# Sample natural language queries for testing
SAMPLE_QUERIES = [
    "Show me all customers from New York",
    "What are the top 5 best-selling products?",
    "How many orders were placed in September 2023?",
    "Show me customers who spent more than $100",
    "What is the average salary by department?",
    "Find products with low stock (less than 100 units)",
    "Show me sales by region",
    "Who are the highest paid employees?",
    "List all cancelled orders",
    "Show me customer registration trends by month"
]

# Expected SQL results for the sample queries
EXPECTED_SQL_RESULTS = {
    "Show me all customers from New York": "SELECT * FROM customers WHERE city = 'New York'",
    "What are the top 5 best-selling products?": "SELECT p.name, SUM(oi.quantity) as total_sold FROM products p JOIN order_items oi ON p.id = oi.product_id GROUP BY p.id ORDER BY total_sold DESC LIMIT 5",
    "How many orders were placed in September 2023?": "SELECT COUNT(*) FROM orders WHERE strftime('%Y-%m', order_date) = '2023-09'",
    "Show me customers who spent more than $100": "SELECT c.*, SUM(o.total_amount) as total_spent FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id HAVING total_spent > 100",
    "What is the average salary by department?": "SELECT department, AVG(salary) as avg_salary FROM employees GROUP BY department",
    "Find products with low stock (less than 100 units)": "SELECT * FROM products WHERE stock_quantity < 100",
    "Show me sales by region": "SELECT region, SUM(total_amount) as total_sales FROM sales GROUP BY region",
    "Who are the highest paid employees?": "SELECT * FROM employees ORDER BY salary DESC",
    "List all cancelled orders": "SELECT * FROM orders WHERE status = 'cancelled'",
    "Show me customer registration trends by month": "SELECT strftime('%Y-%m', registration_date) as month, COUNT(*) as registrations FROM customers GROUP BY month ORDER BY month"
}

# Test scenarios for bias detection
BIAS_TEST_SCENARIOS = [
    {
        "name": "Gender Pay Gap",
        "data": EMPLOYEE_DATA,
        "query": "Compare salaries by gender",
        "expected_bias": True
    },
    {
        "name": "Regional Sales Distribution",
        "data": SALES_DATA,
        "query": "Show sales by region",
        "expected_bias": False
    },
    {
        "name": "Product Category Performance",
        "data": PRODUCT_DATA,
        "query": "Compare product ratings by category",
        "expected_bias": False
    }
]

# Performance test configurations
PERFORMANCE_TEST_CONFIGS = [
    {
        "name": "Simple Query",
        "query": "SELECT COUNT(*) FROM customers",
        "iterations": 100,
        "max_time": 10.0
    },
    {
        "name": "Complex Join",
        "query": "SELECT c.name, SUM(o.total_amount) FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id",
        "iterations": 50,
        "max_time": 15.0
    },
    {
        "name": "Aggregated Analytics",
        "query": "SELECT department, AVG(salary), COUNT(*) FROM employees GROUP BY department",
        "iterations": 75,
        "max_time": 12.0
    }
]

# Error test cases
ERROR_TEST_CASES = [
    {
        "name": "Invalid Table Name",
        "query": "SELECT * FROM nonexistent_table",
        "expected_error": "no such table"
    },
    {
        "name": "Invalid Column Name",
        "query": "SELECT invalid_column FROM customers",
        "expected_error": "no such column"
    },
    {
        "name": "Syntax Error",
        "query": "SELECT * FROM customers WHERE",
        "expected_error": "syntax error"
    },
    {
        "name": "Division by Zero",
        "query": "SELECT 1/0 FROM customers",
        "expected_error": "division by zero"
    }
]

# Integration test scenarios
INTEGRATION_TEST_SCENARIOS = [
    {
        "name": "Complete Order Workflow",
        "steps": [
            "Create customer",
            "Add products to inventory",
            "Place order",
            "Update inventory",
            "Generate invoice"
        ]
    },
    {
        "name": "User Authentication Flow",
        "steps": [
            "User registration",
            "Login attempt",
            "Token generation",
            "API access with token",
            "Logout"
        ]
    },
    {
        "name": "Analytics Dashboard",
        "steps": [
            "Generate sales data",
            "Run multiple queries",
            "Aggregate results",
            "Generate insights",
            "Export report"
        ]
    }
]
