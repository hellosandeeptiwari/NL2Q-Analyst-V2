# NL2Q Agent

## Quickstart

1. Install backend dependencies:
   ```sh
   pip install -r requirements.txt
   ```
2. Install frontend dependencies:
   ```sh
   cd frontend
   npm install
   ```
3. Set up `.env` with your DB and OpenAI credentials.
4. Start backend:
   ```sh
   uvicorn backend.main:app --reload
   ```
5. Start frontend:
   ```sh
   cd frontend
   npm start
   ```

## API Endpoints

- `GET /health` — DB health status
- `GET /schema` — DB schema snapshot
- `POST /query` — Run NL→SQL query (requires auth token)
- `GET /csv/{jobId}` — Download CSV (requires auth token)
- `POST /insights` — Generate insights from stored data (requires auth token)
- `GET /events/status` — SSE DB health
- `GET /logs` — Audit log (requires auth token)

## Example curl

```sh
curl http://localhost:8000/health
curl http://localhost:8000/schema
curl -X POST http://localhost:8000/query -H "Authorization: Bearer your_auth_token" -H "Content-Type: application/json" -d '{"natural_language": "Show top 5 products", "job_id": "job123"}'
curl -H "Authorization: Bearer your_auth_token" http://localhost:8000/csv/job123
curl -X POST http://localhost:8000/insights -H "Authorization: Bearer your_auth_token" -H "Content-Type: application/json" -d '{"location": "backend/exports/job123.csv", "query": "What are the trends?"}'
curl http://localhost:8000/logs -H "Authorization: Bearer your_auth_token"
```

## Features

- **NL→SQL Generation**: Safe SQL from natural language with GPT-4.1-mini
- **Multi-DB Support**: Postgres, Snowflake, extensible to others
- **Data Storage**: S3, SharePoint, or local folder for large datasets
- **Inline Visualizations**: Plotly charts in chat window
- **Insights Generation**: LLM-powered analysis of stored data
- **Authentication**: Token-based security
- **Audit Logging**: Comprehensive query logs
- **Health Monitoring**: Real-time DB status via SSE

## Example NL Prompts & Expected SQL

1. "Show all orders from last month."
   → `SELECT * FROM orders WHERE order_date >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month') LIMIT 100`
2. "List customers who spent more than $1000."
   → `SELECT customer_id, SUM(amount) FROM transactions GROUP BY customer_id HAVING SUM(amount) > 1000 LIMIT 100`
3. "Get product sales by category."
   → `SELECT category, SUM(sales) FROM products GROUP BY category LIMIT 100`
4. "Show daily active users for the past week."
   → `SELECT date, COUNT(DISTINCT user_id) FROM user_activity WHERE date >= CURRENT_DATE - INTERVAL '7 days' GROUP BY date LIMIT 100`
5. "Find top 5 selling products."
   → `SELECT product_id, SUM(sales) FROM sales GROUP BY product_id ORDER BY SUM(sales) DESC LIMIT 5`

## One Round-Trip Run

1. POST `/query` with NL prompt and jobId
2. SQL generated, executed, data stored, visualization returned
3. View chart inline in UI
4. Generate insights with POST `/insights`
5. Download CSV via `/csv/{jobId}`

## Tests

Run all tests:
```sh
pytest tests/
```

## Configuration

See `.env` for DB, OpenAI, storage, and auth settings.
