import openai
import os
from typing import List, Dict, Any

class BiasDetector:
    def __init__(self):
        self.protected_features = ["gender", "race", "age", "ethnicity", "income"]  # Configurable

    def detect_bias(self, data: List[Dict[str, Any]], query: str) -> str:
        if not data:
            return "No data to analyze for bias."

        columns = list(data[0].keys())
        protected_in_data = [col for col in columns if col.lower() in self.protected_features]

        if not protected_in_data:
            return "No protected features detected in data."

        prompt = f"""
        Analyze the following data for potential bias in the context of the query: "{query}".
        Protected features present: {protected_in_data}
        Data sample: {data[:10]}
        Check for disparities in outcomes across protected groups. Provide a summary of any detected biases and recommendations.
        """
        response = openai.ChatCompletion.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            messages=[{"role": "user", "content": prompt}]
        )
        return response["choices"][0]["message"]["content"]
