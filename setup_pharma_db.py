"""
Pharma GCO Database Setup - Mock NBA tables for testing
Creates sample tables that match the real pharma NBA output structure
"""

import sqlite3
import pandas as pd
import os
from pathlib import Path

def create_mock_pharma_database():
    """Create a mock SQLite database with NBA-style pharma tables"""
    
    db_path = "pharma_gco_test.db"
    
    # Remove existing database
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # Create new database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("🏥 Creating mock pharma GCO database...")
    
    # Create the main NBA output table with pharma-realistic structure
    cursor.execute("""
        CREATE TABLE NBA_PHASE2_SEP2024_SIMILARITY_OUTPUT_FINAL_PYTHON (
            provider_id TEXT PRIMARY KEY,
            provider_name TEXT,
            hcp_specialty TEXT,
            therapeutic_area TEXT,
            recommended_message TEXT,
            provider_input TEXT,
            action_effect TEXT,
            recommendation_score REAL,
            confidence_level TEXT,
            engagement_history TEXT,
            region TEXT,
            hcp_tier TEXT,
            last_interaction_date TEXT,
            preferred_channel TEXT,
            clinical_interests TEXT,
            prescription_volume_rank INTEGER,
            patient_population_size INTEGER,
            digital_engagement_score REAL,
            content_preferences TEXT,
            timestamp TEXT
        )
    """)
    
    # Insert realistic pharma GCO data
    sample_data = [
        ('PRV001', 'Dr. Sarah Chen', 'Oncology', 'Oncology', 'Clinical Trial Enrollment Opportunity', 
         'Interested in oncology trials', 'High engagement expected', 0.95, 'High', 
         'Previous trial participant', 'US-East', 'Tier 1', '2025-08-15', 'Email', 
         'Immuno-oncology, CAR-T therapy', 1, 450, 0.87, 'Scientific publications', '2025-09-04 10:30:00'),
        
        ('PRV002', 'Dr. Mike Johnson', 'Cardiology', 'Cardiology', 'New Treatment Protocol Available',
         'Looking for diabetes management', 'Medium engagement expected', 0.87, 'Medium',
         'Regular medical updates', 'US-West', 'Tier 2', '2025-08-20', 'In-person', 
         'Heart failure, Diabetes comorbidity', 3, 320, 0.72, 'Clinical guidelines', '2025-09-04 10:30:00'),
        
        ('PRV003', 'Dr. Lisa Wang', 'Endocrinology', 'Endocrinology', 'Patient Education Resources',
         'Needs patient education materials', 'Educational value high', 0.82, 'High',
         'Education focused interactions', 'EU-Central', 'Tier 1', '2025-08-25', 'Digital portal',
         'Type 2 diabetes, GLP-1 therapies', 2, 380, 0.91, 'Patient education tools', '2025-09-04 10:30:00'),
        
        ('PRV004', 'Dr. John Smith', 'Oncology', 'Oncology', 'Clinical Trial Enrollment Opportunity',
         'Seeking trial opportunities', 'High engagement expected', 0.78, 'High',
         'Active trial investigator', 'US-East', 'Tier 1', '2025-08-10', 'Phone',
         'Lung cancer, Targeted therapy', 1, 520, 0.83, 'Research protocols', '2025-09-04 10:30:00'),
        
        ('PRV005', 'Dr. Emma Davis', 'Nephrology', 'Nephrology', 'Dosing Guidelines Update',
         'Requesting dosing information', 'Clinical utility high', 0.74, 'Medium',
         'Clinical guidance seeker', 'APAC', 'Tier 2', '2025-08-30', 'Email',
         'Chronic kidney disease, Dialysis', 4, 280, 0.65, 'Dosing recommendations', '2025-09-04 10:30:00'),
        
        ('PRV006', 'Dr. Robert Chen', 'Cardiology', 'Cardiology', 'Real-world Evidence Insights',
         'Interested in outcomes data', 'High engagement expected', 0.89, 'High',
         'Data-driven decision maker', 'US-West', 'Tier 1', '2025-08-18', 'Digital portal',
         'SGLT2 inhibitors, Heart failure', 2, 420, 0.88, 'RWE studies', '2025-09-04 10:30:00'),
        
        ('PRV007', 'Dr. Maria Rodriguez', 'Oncology', 'Oncology', 'Biomarker Testing Guidelines',
         'Seeking biomarker guidance', 'Clinical utility high', 0.91, 'High',
         'Precision medicine advocate', 'US-East', 'Tier 1', '2025-08-12', 'In-person',
         'Breast cancer, Biomarker testing', 1, 390, 0.84, 'Diagnostic guidance', '2025-09-04 10:30:00'),
        
        ('PRV008', 'Dr. James Wilson', 'Endocrinology', 'Endocrinology', 'Combination Therapy Insights',
         'Exploring combination treatments', 'High engagement expected', 0.86, 'High',
         'Innovation early adopter', 'EU-Central', 'Tier 1', '2025-08-22', 'Email',
         'Diabetes, Combination therapies', 2, 350, 0.79, 'Combination protocols', '2025-09-04 10:30:00')
    ]
    
    cursor.executemany("""
        INSERT INTO NBA_PHASE2_SEP2024_SIMILARITY_OUTPUT_FINAL_PYTHON VALUES 
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, sample_data)
    
    # Create additional tables that might exist in a real pharma system
    cursor.execute("""
        CREATE TABLE PROVIDER_MASTER (
            provider_id TEXT PRIMARY KEY,
            npi_number TEXT,
            first_name TEXT,
            last_name TEXT,
            specialty TEXT,
            sub_specialty TEXT,
            practice_type TEXT,
            address TEXT,
            city TEXT,
            state TEXT,
            zip_code TEXT,
            phone TEXT,
            email TEXT
        )
    """)
    
    cursor.execute("""
        CREATE TABLE THERAPEUTIC_AREAS (
            area_id INTEGER PRIMARY KEY,
            area_name TEXT,
            therapeutic_class TEXT,
            indication TEXT,
            primary_drug TEXT,
            market_size TEXT,
            growth_rate REAL
        )
    """)
    
    # Insert some master data
    provider_master_data = [
        ('PRV001', 'NPI001', 'Sarah', 'Chen', 'Oncology', 'Medical Oncology', 'Hospital', '123 Medical Ave', 'Boston', 'MA', '02101', '555-0101', 'schen@hospital.com'),
        ('PRV002', 'NPI002', 'Mike', 'Johnson', 'Cardiology', 'Interventional Cardiology', 'Private Practice', '456 Heart St', 'Los Angeles', 'CA', '90210', '555-0102', 'mjohnson@cardio.com'),
        ('PRV003', 'NPI003', 'Lisa', 'Wang', 'Endocrinology', 'Diabetes & Metabolism', 'Academic Center', '789 Endo Blvd', 'Berlin', 'DE', '10115', '555-0103', 'lwang@university.de'),
        ('PRV004', 'NPI004', 'John', 'Smith', 'Oncology', 'Thoracic Oncology', 'Cancer Center', '321 Oncology Way', 'New York', 'NY', '10001', '555-0104', 'jsmith@cancer.org'),
        ('PRV005', 'NPI005', 'Emma', 'Davis', 'Nephrology', 'Dialysis', 'Specialty Clinic', '654 Kidney Lane', 'Sydney', 'NSW', '2000', '555-0105', 'edavis@nephro.au')
    ]
    
    cursor.executemany("""
        INSERT INTO PROVIDER_MASTER VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, provider_master_data)
    
    therapeutic_areas_data = [
        (1, 'Oncology', 'Immuno-oncology', 'Non-small cell lung cancer', 'Pembrolizumab', 'Large', 12.5),
        (2, 'Cardiology', 'Heart failure', 'Heart failure with reduced ejection fraction', 'Sacubitril/Valsartan', 'Medium', 8.3),
        (3, 'Endocrinology', 'Diabetes', 'Type 2 diabetes mellitus', 'GLP-1 agonists', 'Large', 15.7),
        (4, 'Nephrology', 'Chronic kidney disease', 'CKD with diabetes', 'SGLT2 inhibitors', 'Medium', 9.2)
    ]
    
    cursor.executemany("""
        INSERT INTO THERAPEUTIC_AREAS VALUES (?, ?, ?, ?, ?, ?, ?)
    """, therapeutic_areas_data)
    
    conn.commit()
    conn.close()
    
    print(f"✅ Created mock pharma database: {db_path}")
    print(f"📊 Tables created:")
    print(f"   • NBA_PHASE2_SEP2024_SIMILARITY_OUTPUT_FINAL_PYTHON ({len(sample_data)} records)")
    print(f"   • PROVIDER_MASTER ({len(provider_master_data)} records)")
    print(f"   • THERAPEUTIC_AREAS ({len(therapeutic_areas_data)} records)")
    
    return db_path

def test_database_connection():
    """Test the database connection and show sample data"""
    
    db_path = create_mock_pharma_database()
    
    # Test connection and show data
    conn = sqlite3.connect(db_path)
    
    print(f"\n🔍 Testing database connection...")
    
    # Show tables
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f"📋 Available tables: {[table[0] for table in tables]}")
    
    # Test the main NBA table
    print(f"\n📊 Sample data from NBA output table:")
    df = pd.read_sql("SELECT * FROM NBA_PHASE2_SEP2024_SIMILARITY_OUTPUT_FINAL_PYTHON LIMIT 3", conn)
    print(df[['provider_name', 'recommended_message', 'provider_input', 'action_effect']].to_string())
    
    conn.close()
    return db_path

if __name__ == "__main__":
    test_database_connection()
