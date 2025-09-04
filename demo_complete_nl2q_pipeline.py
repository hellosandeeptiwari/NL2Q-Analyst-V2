"""
Complete NL2Q Pipeline Demo: Data → SQLite → LLM-SQL → LLM-Python → Visualization
Shows the full flow: Load data in SQLite → Generate SQL → Transform data → Generate Python → Create visualization
"""

import sqlite3
import pandas as pd
import json
import os
from datetime import datetime

# Import visualization libraries
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    px = None
    make_subplots = None

def step1_load_data_to_sqlite():
    """Step 1: Load pharma data into SQLite (simulating data ingestion)"""
    
    print("🔄 STEP 1: Loading Pharma Data into SQLite")
    print("="*60)
    
    # Simulate loading data from various sources into SQLite
    pharma_data = {
        'provider_id': ['PRV001', 'PRV002', 'PRV003', 'PRV004', 'PRV005', 'PRV006', 'PRV007', 'PRV008'],
        'provider_name': [
            'Dr. Sarah Chen', 'Dr. Mike Johnson', 'Dr. Lisa Wang', 'Dr. John Smith', 
            'Dr. Emma Davis', 'Dr. Robert Chen', 'Dr. Maria Rodriguez', 'Dr. James Wilson'
        ],
        'recommended_message': [
            'Clinical Trial Enrollment Opportunity',
            'New Treatment Protocol Available',
            'Patient Education Resources', 
            'Clinical Trial Enrollment Opportunity',
            'Dosing Guidelines Update',
            'Real-world Evidence Insights',
            'Biomarker Testing Guidelines',
            'Combination Therapy Insights'
        ],
        'provider_input': [
            'Interested in oncology trials',
            'Looking for diabetes management',
            'Needs patient education materials',
            'Seeking trial opportunities', 
            'Requesting dosing information',
            'Interested in outcomes data',
            'Seeking biomarker guidance',
            'Exploring combination treatments'
        ],
        'action_effect': [
            'High engagement expected',
            'Medium engagement expected',
            'Educational value high',
            'High engagement expected',
            'Clinical utility high',
            'High engagement expected',
            'Clinical utility high',
            'High engagement expected'
        ],
        'therapeutic_area': ['Oncology', 'Cardiology', 'Endocrinology', 'Oncology', 'Nephrology', 'Cardiology', 'Oncology', 'Endocrinology'],
        'engagement_score': [0.95, 0.67, 0.82, 0.78, 0.74, 0.89, 0.91, 0.86],
        'region': ['US-East', 'US-West', 'EU-Central', 'US-East', 'APAC', 'US-West', 'US-East', 'EU-Central']
    }
    
    # Load into SQLite
    conn = sqlite3.connect('pharma_analysis.db')
    df = pd.DataFrame(pharma_data)
    df.to_sql('nba_output', conn, if_exists='replace', index=False)
    
    print(f"✅ Loaded {len(df)} records into SQLite table 'nba_output'")
    print(f"📊 Columns: {list(df.columns)}")
    
    conn.close()
    return len(df)

def step2_llm_generate_sql(user_query: str):
    """Step 2: LLM generates SQL for the user's request"""
    
    print(f"\n🧠 STEP 2: LLM Generating SQL for User Query")
    print("="*60)
    print(f"📝 User Query: {user_query}")
    
    # Simulate LLM generating SQL based on user query
    # In real system, this would call OpenAI/Claude with schema context
    
    if "frequency" in user_query.lower() and "top 5" in user_query.lower():
        generated_sql = """
-- LLM-Generated SQL for frequency analysis with top 5 rows
SELECT 
    provider_name,
    recommended_message,
    provider_input, 
    action_effect,
    therapeutic_area,
    engagement_score
FROM nba_output 
ORDER BY engagement_score DESC 
LIMIT 5;

-- Additional frequency analysis queries
SELECT 
    recommended_message,
    COUNT(*) as frequency,
    AVG(engagement_score) as avg_engagement
FROM nba_output 
GROUP BY recommended_message 
ORDER BY frequency DESC;

SELECT 
    action_effect,
    COUNT(*) as frequency 
FROM nba_output 
GROUP BY action_effect 
ORDER BY frequency DESC;
"""
    else:
        generated_sql = """
SELECT * FROM nba_output LIMIT 10;
"""
    
    print("🔍 LLM-Generated SQL:")
    print(generated_sql)
    
    return generated_sql

def step3_execute_sql_transformations(sql_queries: str):
    """Step 3: Execute LLM-generated SQL on SQLite for data transformations"""
    
    print(f"\n⚡ STEP 3: Executing SQL Transformations on SQLite")
    print("="*60)
    
    conn = sqlite3.connect('pharma_analysis.db')
    
    # Manually split the queries (since they have complex formatting)
    individual_queries = [
        """SELECT 
    provider_name,
    recommended_message,
    provider_input, 
    action_effect,
    therapeutic_area,
    engagement_score
FROM nba_output 
ORDER BY engagement_score DESC 
LIMIT 5""",
        """SELECT 
    recommended_message,
    COUNT(*) as frequency,
    AVG(engagement_score) as avg_engagement
FROM nba_output 
GROUP BY recommended_message 
ORDER BY frequency DESC""",
        """SELECT 
    action_effect,
    COUNT(*) as frequency 
FROM nba_output 
GROUP BY action_effect 
ORDER BY frequency DESC"""
    ]
    
    print(f"🔍 Executing {len(individual_queries)} SQL queries")
    
    results = {}
    
    for i, query in enumerate(individual_queries, 1):
        try:
            print(f"🔄 Executing Query {i}...")
            print(f"📝 SQL: {query[:100]}...")
            df = pd.read_sql_query(query, conn)
            results[f"query_{i}"] = df
            print(f"✅ Query {i} returned {len(df)} rows, {len(df.columns)} columns")
            
            # Show preview
            if len(df) > 0:
                print(f"📊 Preview:")
                print(df.head(3).to_string())
                print()
                
        except Exception as e:
            print(f"❌ Query {i} failed: {e}")
            print(f"   SQL was: {query}")
            print()
    
    conn.close()
    return results

def step4_llm_generate_visualization_code(query_results: dict, user_query: str):
    """Step 4: LLM generates Python visualization code based on transformed data"""
    
    print(f"\n🎨 STEP 4: LLM Generating Python Visualization Code")
    print("="*60)
    
    # Simulate LLM generating Python code for visualization
    # In real system, this would analyze the data and generate appropriate charts
    
    python_viz_code = '''
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

def create_pharma_frequency_visualization(data_dict):
    """LLM-Generated Python code for pharma frequency visualization"""
    
    # Get the frequency data from SQL results
    freq_data = data_dict.get("query_2")  # Recommended message frequency
    effect_data = data_dict.get("query_3")  # Action effect frequency
    top_data = data_dict.get("query_1")    # Top 5 rows
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Top 5 Providers by Engagement", 
                       "Recommended Message Frequency",
                       "Action Effect Distribution", 
                       "Engagement Score Distribution"],
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "pie"}, {"type": "histogram"}]]
    )
    
    # Chart 1: Top 5 providers
    if top_data is not None and len(top_data) > 0:
        fig.add_trace(
            go.Bar(x=top_data["provider_name"], 
                   y=top_data["engagement_score"],
                   name="Engagement Score",
                   marker_color="lightblue"),
            row=1, col=1
        )
    
    # Chart 2: Message frequency  
    if freq_data is not None and len(freq_data) > 0:
        fig.add_trace(
            go.Bar(x=freq_data["recommended_message"],
                   y=freq_data["frequency"], 
                   name="Message Frequency",
                   marker_color="lightgreen"),
            row=1, col=2
        )
    
    # Chart 3: Action effect pie chart
    if effect_data is not None and len(effect_data) > 0:
        fig.add_trace(
            go.Pie(labels=effect_data["action_effect"],
                   values=effect_data["frequency"],
                   name="Action Effects"),
            row=2, col=1
        )
    
    # Chart 4: Engagement distribution
    if top_data is not None and len(top_data) > 0:
        fig.add_trace(
            go.Histogram(x=top_data["engagement_score"],
                        name="Engagement Distribution",
                        marker_color="orange"),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        title="Pharma GCO Analysis: Message Frequency & Provider Engagement",
        showlegend=False,
        height=800,
        width=1200
    )
    
    return fig

# Execute the visualization
fig = create_pharma_frequency_visualization(query_results)
fig.show()
fig.write_html("pharma_gco_analysis.html")
print("✅ Visualization saved as pharma_gco_analysis.html")
'''
    
    print("🔍 LLM-Generated Python Visualization Code:")
    print(python_viz_code[:500] + "..." if len(python_viz_code) > 500 else python_viz_code)
    
    return python_viz_code

def step5_execute_visualization(viz_code: str, query_results: dict):
    """Step 5: Execute LLM-generated Python code to create visualization"""
    
    print(f"\n📊 STEP 5: Executing Python Visualization Code")
    print("="*60)
    
    if not PLOTLY_AVAILABLE:
        print("❌ Plotly not available, skipping visualization")
        return False
    
    try:
        # Create a global environment with all required modules
        viz_globals = {
            'go': go,
            'px': px, 
            'make_subplots': make_subplots,
            'pd': pd,
            'query_results': query_results
        }
        
        # Execute the LLM-generated code
        exec(viz_code, viz_globals)
        
        print("✅ Visualization code executed successfully")
        print("📁 Output files:")
        if os.path.exists("pharma_gco_analysis.html"):
            print("   • pharma_gco_analysis.html (Interactive visualization)")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demonstrate_complete_nl2q_pipeline():
    """Demonstrate the complete NL2Q pipeline"""
    
    print("🏥 COMPLETE NL2Q PIPELINE DEMONSTRATION")
    print("🎯 User Query: 'read table final nba output python and fetch top 5 rows and create a visualization with frequency of recommended message and input and action effect'")
    print("="*80)
    
    user_query = "read table final nba output python and fetch top 5 rows and create a visualization with frequency of recommended message and input and action effect"
    
    # Step 1: Load data into SQLite
    record_count = step1_load_data_to_sqlite()
    
    # Step 2: LLM generates SQL
    sql_queries = step2_llm_generate_sql(user_query)
    
    # Step 3: Execute SQL transformations
    query_results = step3_execute_sql_transformations(sql_queries)
    
    # Step 4: LLM generates Python visualization code
    viz_code = step4_llm_generate_visualization_code(query_results, user_query)
    
    # Step 5: Execute visualization
    viz_success = step5_execute_visualization(viz_code, query_results)
    
    # Summary
    print(f"\n🎉 PIPELINE EXECUTION SUMMARY")
    print("="*60)
    print(f"✅ Data loaded into SQLite: {record_count} records")
    print(f"✅ SQL queries generated and executed: {len(query_results)}")
    print(f"✅ Python visualization code generated: {len(viz_code)} characters")
    print(f"✅ Visualization created: {'Yes' if viz_success else 'No'}")
    
    print(f"\n🔄 COMPLETE FLOW CONFIRMED:")
    print("   1. 📥 Data → SQLite (local processing engine)")
    print("   2. 🧠 LLM → SQL (for aggregations/filtering)")  
    print("   3. ⚡ SQLite → Transformed data")
    print("   4. 🧠 LLM → Python (for visualization)")
    print("   5. 📊 Python → Interactive charts")
    
    return query_results, viz_success

if __name__ == "__main__":
    demonstrate_complete_nl2q_pipeline()
