import boto3
import os
from pathlib import Path
from typing import List, Dict, Any
import json
import pandas as pd
from io import StringIO

class DataStorage:
    def __init__(self, storage_type: str = "local"):
        self.storage_type = storage_type
        self.s3_client = None
        self.sharepoint_client = None
        if storage_type == "s3":
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=os.getenv("AWS_REGION", "us-east-1")
            )
        elif storage_type == "sharepoint":
            # Placeholder for SharePoint integration (e.g., using office365-rest-python-client)
            pass

    def save_data(self, data: List[Dict[str, Any]], job_id: str, format: str = "csv") -> str:
        if len(data) < 1000:  # Small data, keep in memory
            return "memory"
        if self.storage_type == "s3":
            bucket = os.getenv("S3_BUCKET", "nl2q-results")
            key = f"{job_id}.{format}"
            df = pd.DataFrame(data)
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            self.s3_client.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())
            return f"s3://{bucket}/{key}"
        elif self.storage_type == "sharepoint":
            # Implement SharePoint upload
            return f"sharepoint://{job_id}.{format}"
        else:  # local
            path = Path("backend/exports") / f"{job_id}.{format}"
            path.parent.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(data)
            df.to_csv(path, index=False)
            return str(path)

    def load_data(self, location: str) -> List[Dict[str, Any]]:
        if location == "memory":
            return []  # Assume data is passed separately
        if location.startswith("s3://"):
            bucket, key = location.replace("s3://", "").split("/", 1)
            obj = self.s3_client.get_object(Bucket=bucket, Key=key)
            df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
            return df.to_dict('records')
        elif location.startswith("sharepoint://"):
            # Implement SharePoint download
            return []
        else:  # local
            df = pd.read_csv(location)
            return df.to_dict('records')

    def generate_insights(self, data: List[Dict[str, Any]], query: str) -> str:
        # Use LLM to generate insights from data
        import openai
        prompt = f"Analyze this data and answer: {query}\nData: {json.dumps(data[:10])}"  # Sample first 10 rows
        response = openai.ChatCompletion.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            messages=[{"role": "user", "content": prompt}]
        )
        insight = response["choices"][0]["message"]["content"]
        # Add explainability: feature impact
        if data:
            columns = list(data[0].keys())
            explain_prompt = f"Explain the key features/columns influencing the results for query: {query}\nColumns: {columns}\nData sample: {json.dumps(data[:5])}"
            explain_response = openai.ChatCompletion.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
                messages=[{"role": "user", "content": explain_prompt}]
            )
            explanation = explain_response["choices"][0]["message"]["content"]
            return f"Insight: {insight}\n\nExplainability: {explanation}"
        return insight
