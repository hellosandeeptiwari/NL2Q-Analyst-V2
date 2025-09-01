import pandas as pd
import plotly.express as px

def auto_plot_spec(df: pd.DataFrame) -> dict:
    # Infer x/time col, y numeric cols, categorical
    if df.empty:
        return {}
    time_cols = [col for col in df.columns if 'date' in col or 'time' in col]
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    x = time_cols[0] if time_cols else (cat_cols[0] if cat_cols else df.columns[0])
    y = numeric_cols[0] if numeric_cols else df.columns[1] if len(df.columns) > 1 else df.columns[0]
    fig = px.line(df, x=x, y=y) if time_cols else px.bar(df, x=x, y=y)
    return fig.to_dict()
