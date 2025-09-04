import openai
import os
from typing import List, Dict, Any, Union

class BiasDetector:
    def __init__(self):
        self.protected_features = ["gender", "race", "age", "ethnicity", "income"]  # Configurable

    def detect_bias(self, data: List[Any], query: str) -> str:
        if not data:
            return "No data to analyze for bias."

        # Handle both dict and tuple data formats
        columns = []
        if isinstance(data[0], dict):
            columns = list(data[0].keys())
        elif isinstance(data[0], (tuple, list)):
            # For tuple/list data, we can't easily determine column names for bias detection
            return "Bias detection requires structured data with column names."
        else:
            return "Unsupported data format for bias detection."

        protected_in_data = [col for col in columns if col.lower() in self.protected_features]

        if not protected_in_data:
            return "No protected features detected in data."

        prompt = f"""
        Analyze the following data for potential bias in the context of the query: "{query}".
        Protected features present: {protected_in_data}
        Data sample: {data[:10]}
        Check for disparities in outcomes across protected groups. Provide a summary of any detected biases and recommendations.
        """
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
