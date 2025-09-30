"""
Schema Semantic Analyzer - Deep understanding of database schema semantics

This module provides semantic analysis of database schemas to understand
table purposes, column meanings, and potential relationships.
"""

import logging
from typing import List, Dict, Any, Optional, Set
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TableSemanticProfile:
    """Semantic profile of a database table"""
    domain_entities: List[str]
    data_types: List[str]
    relationship_types: List[str]
    column_semantics: Dict[str, str]
    business_purpose: str
    confidence_score: float

class SchemaSemanticAnalyzer:
    """
    Analyzes database schemas to extract semantic meaning from table and column names,
    data types, and relationships.
    """
    
    def __init__(self):
        self.domain_patterns = self._initialize_domain_patterns()
        self.data_type_patterns = self._initialize_data_type_patterns()
        self.relationship_patterns = self._initialize_relationship_patterns()
        
    def _initialize_domain_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for domain entity recognition"""
        return {
            'prescriber': [
                'prescriber', 'doctor', 'physician', 'provider', 'practitioner',
                'npi', 'dea', 'hcp', 'healthcare_provider'
            ],
            'prescription': [
                'prescription', 'rx', 'drug', 'medication', 'pharma', 'therapeutic',
                'ndc', 'dosage', 'strength', 'formulation'
            ],
            'financial': [
                'revenue', 'sales', 'profit', 'cost', 'price', 'amount', 'value',
                'billing', 'payment', 'invoice', 'financial'
            ],
            'geographic': [
                'geography', 'geo', 'region', 'territory', 'state', 'city', 'zip',
                'location', 'address', 'country'
            ],
            'temporal': [
                'date', 'time', 'month', 'year', 'quarter', 'period', 'timestamp',
                'created', 'updated', 'modified'
            ],
            'patient': [
                'patient', 'member', 'beneficiary', 'enrollee', 'individual',
                'person', 'demographic'
            ],
            'organization': [
                'organization', 'org', 'company', 'institution', 'facility',
                'hospital', 'clinic', 'pharmacy'
            ]
        }
    
    def _initialize_data_type_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for data type recognition"""
        return {
            'numeric': [
                'int', 'integer', 'decimal', 'float', 'double', 'number',
                'count', 'sum', 'total', 'amount', 'quantity'
            ],
            'categorical': [
                'varchar', 'char', 'string', 'text', 'category', 'type',
                'status', 'flag', 'indicator'
            ],
            'temporal': [
                'date', 'datetime', 'timestamp', 'time', 'year', 'month'
            ],
            'identifier': [
                'id', 'key', 'code', 'number', 'npi', 'dea', 'ndc'
            ],
            'financial': [
                'money', 'currency', 'dollar', 'cost', 'price', 'revenue'
            ]
        }
    
    def _initialize_relationship_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for relationship type recognition"""
        return {
            'grouping': [
                'by', 'per', 'group', 'category', 'classification'
            ],
            'comparison': [
                'vs', 'versus', 'compare', 'against', 'difference'
            ],
            'temporal': [
                'over_time', 'trend', 'historical', 'change', 'growth'
            ],
            'hierarchical': [
                'parent', 'child', 'level', 'hierarchy', 'rollup'
            ],
            'aggregation': [
                'sum', 'count', 'average', 'total', 'max', 'min'
            ]
        }
    
    def analyze_table_semantics(self, table_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the semantic meaning of a table based on its metadata.
        
        Args:
            table_info: Table metadata from vector store
            
        Returns:
            Semantic profile dictionary
        """
        try:
            table_name = table_info.get('table_name', '').lower()
            columns = table_info.get('columns', [])
            
            # Extract domain entities
            domain_entities = self._extract_domain_entities(table_name, columns)
            
            # Analyze data types
            data_types = self._analyze_data_types(columns)
            
            # Identify relationship types
            relationship_types = self._identify_relationship_types(table_name, columns)
            
            # Analyze column semantics
            column_semantics = self._analyze_column_semantics(columns)
            
            # Determine business purpose
            business_purpose = self._determine_business_purpose(table_name, domain_entities, columns)
            
            # Calculate confidence score
            confidence_score = self._calculate_semantic_confidence(
                domain_entities, data_types, relationship_types, columns
            )
            
            return {
                'domain_entities': domain_entities,
                'data_types': data_types,
                'relationship_types': relationship_types,
                'column_semantics': column_semantics,
                'business_purpose': business_purpose,
                'confidence_score': confidence_score
            }
            
        except Exception as e:
            logger.error(f"Error analyzing table semantics: {str(e)}")
            return self._create_default_semantic_profile()
    
    def _extract_domain_entities(self, table_name: str, columns: List[Dict[str, Any]]) -> List[str]:
        """Extract domain entities from table name and column names"""
        entities = set()
        
        # Analyze table name
        for domain, patterns in self.domain_patterns.items():
            for pattern in patterns:
                if pattern in table_name:
                    entities.add(domain)
        
        # Analyze column names
        for column in columns:
            column_name = column.get('column_name', '').lower()
            for domain, patterns in self.domain_patterns.items():
                for pattern in patterns:
                    if pattern in column_name:
                        entities.add(domain)
        
        return list(entities)
    
    def _analyze_data_types(self, columns: List[Dict[str, Any]]) -> List[str]:
        """Analyze data types present in the table"""
        data_types = set()
        
        for column in columns:
            column_name = column.get('column_name', '').lower()
            data_type = column.get('data_type', '').lower()
            
            # Map database types to semantic types
            for semantic_type, patterns in self.data_type_patterns.items():
                for pattern in patterns:
                    if pattern in data_type or pattern in column_name:
                        data_types.add(semantic_type)
        
        return list(data_types)
    
    def _identify_relationship_types(self, table_name: str, columns: List[Dict[str, Any]]) -> List[str]:
        """Identify types of relationships this table can support"""
        relationship_types = set()
        
        # Check for foreign key patterns
        for column in columns:
            column_name = column.get('column_name', '').lower()
            
            if column_name.endswith('_id') or column_name.endswith('_key'):
                relationship_types.add('hierarchical')
            
            if any(term in column_name for term in ['date', 'time', 'created', 'updated']):
                relationship_types.add('temporal')
            
            if any(term in column_name for term in ['category', 'type', 'group', 'class']):
                relationship_types.add('grouping')
        
        # Check for aggregation support (numeric columns)
        has_numeric = any(
            any(num_pattern in col.get('data_type', '').lower() 
                for num_pattern in ['int', 'decimal', 'float', 'number'])
            for col in columns
        )
        
        if has_numeric:
            relationship_types.add('aggregation')
        
        return list(relationship_types)
    
    def _analyze_column_semantics(self, columns: List[Dict[str, Any]]) -> Dict[str, str]:
        """Analyze semantic meaning of individual columns"""
        column_semantics = {}
        
        for column in columns:
            column_name = column.get('column_name', '')
            semantic_type = self._classify_column_semantic_type(column)
            column_semantics[column_name] = semantic_type
        
        return column_semantics
    
    def _classify_column_semantic_type(self, column: Dict[str, Any]) -> str:
        """Classify the semantic type of a column"""
        column_name = column.get('column_name', '').lower()
        data_type = column.get('data_type', '').lower()
        
        # Check for identifiers
        if any(pattern in column_name for pattern in ['id', 'key', 'npi', 'dea', 'ndc']):
            return 'identifier'
        
        # Check for temporal columns
        if any(pattern in column_name for pattern in ['date', 'time', 'created', 'updated']):
            return 'temporal'
        
        # Check for financial columns
        if any(pattern in column_name for pattern in ['amount', 'cost', 'price', 'revenue', 'value']):
            return 'financial'
        
        # Check for categorical columns
        if any(pattern in column_name for pattern in ['type', 'category', 'status', 'flag']):
            return 'categorical'
        
        # Check for geographic columns
        if any(pattern in column_name for pattern in ['state', 'city', 'zip', 'region']):
            return 'geographic'
        
        # Check for measurement columns
        if any(pattern in data_type for pattern in ['int', 'decimal', 'float']):
            return 'numeric'
        
        return 'general'
    
    def _determine_business_purpose(
        self, 
        table_name: str, 
        domain_entities: List[str], 
        columns: List[Dict[str, Any]]
    ) -> str:
        """Determine the business purpose of the table"""
        
        # Analyze table name for business purpose indicators
        if 'reporting' in table_name:
            return 'Reporting and Analytics'
        elif 'transaction' in table_name:
            return 'Transactional Data'
        elif 'master' in table_name or 'reference' in table_name:
            return 'Master/Reference Data'
        elif 'log' in table_name or 'audit' in table_name:
            return 'Logging and Auditing'
        
        # Analyze domain entities
        if 'prescriber' in domain_entities and 'financial' in domain_entities:
            return 'Prescriber Financial Analytics'
        elif 'prescription' in domain_entities and 'patient' in domain_entities:
            return 'Prescription Management'
        elif 'geographic' in domain_entities and 'financial' in domain_entities:
            return 'Geographic Sales Analysis'
        
        # Default based on primary domain entity
        if domain_entities:
            primary_entity = domain_entities[0]
            return f"{primary_entity.title()} Data Management"
        
        return 'General Business Data'
    
    def _calculate_semantic_confidence(
        self, 
        domain_entities: List[str], 
        data_types: List[str], 
        relationship_types: List[str], 
        columns: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence score for semantic analysis"""
        
        confidence = 0.0
        
        # Domain entity recognition confidence
        if domain_entities:
            confidence += 0.3 * min(len(domain_entities) / 2, 1.0)
        
        # Data type diversity confidence
        if data_types:
            confidence += 0.2 * min(len(data_types) / 3, 1.0)
        
        # Relationship capability confidence
        if relationship_types:
            confidence += 0.2 * min(len(relationship_types) / 2, 1.0)
        
        # Column analysis confidence
        if columns:
            analyzed_columns = sum(1 for col in columns if self._has_semantic_meaning(col))
            confidence += 0.3 * (analyzed_columns / len(columns))
        
        return min(confidence, 1.0)
    
    def _has_semantic_meaning(self, column: Dict[str, Any]) -> bool:
        """Check if a column has clear semantic meaning"""
        column_name = column.get('column_name', '').lower()
        
        # Check against all domain patterns
        for patterns in self.domain_patterns.values():
            if any(pattern in column_name for pattern in patterns):
                return True
        
        # Check against data type patterns
        for patterns in self.data_type_patterns.values():
            if any(pattern in column_name for pattern in patterns):
                return True
        
        return False
    
    def find_potential_joins(
        self, 
        table1_info: Dict[str, Any], 
        table2_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find potential join relationships between two tables"""
        
        joins = []
        
        table1_columns = table1_info.get('columns', [])
        table2_columns = table2_info.get('columns', [])
        
        # Look for exact column name matches
        for col1 in table1_columns:
            col1_name = col1.get('column_name', '').lower()
            for col2 in table2_columns:
                col2_name = col2.get('column_name', '').lower()
                
                if col1_name == col2_name and self._is_join_candidate(col1, col2):
                    joins.append({
                        'type': 'exact_match',
                        'columns': [col1_name, col2_name],
                        'confidence': 0.9
                    })
        
        # Look for semantic joins (e.g., prescriber_id -> npi)
        semantic_joins = self._find_semantic_joins(table1_columns, table2_columns)
        joins.extend(semantic_joins)
        
        return joins
    
    def _is_join_candidate(self, col1: Dict[str, Any], col2: Dict[str, Any]) -> bool:
        """Check if two columns are good candidates for joining"""
        
        # Check if columns are identifiers
        col1_name = col1.get('column_name', '').lower()
        col2_name = col2.get('column_name', '').lower()
        
        if any(pattern in col1_name for pattern in ['id', 'key', 'npi', 'dea']):
            return True
        
        if any(pattern in col2_name for pattern in ['id', 'key', 'npi', 'dea']):
            return True
        
        return False
    
    def _find_semantic_joins(
        self, 
        table1_columns: List[Dict[str, Any]], 
        table2_columns: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find semantic join relationships between tables"""
        
        semantic_joins = []
        
        # Define semantic equivalencies
        semantic_equivalencies = {
            'prescriber_id': ['npi', 'provider_id', 'doctor_id'],
            'npi': ['prescriber_id', 'provider_id'],
            'patient_id': ['member_id', 'beneficiary_id'],
            'drug_id': ['ndc', 'medication_id'],
            'geography_id': ['region_id', 'territory_id']
        }
        
        for col1 in table1_columns:
            col1_name = col1.get('column_name', '').lower()
            
            for col2 in table2_columns:
                col2_name = col2.get('column_name', '').lower()
                
                # Check semantic equivalencies
                for base_term, equivalents in semantic_equivalencies.items():
                    if (col1_name == base_term and col2_name in equivalents) or \
                       (col2_name == base_term and col1_name in equivalents):
                        semantic_joins.append({
                            'type': 'semantic_match',
                            'columns': [col1_name, col2_name],
                            'confidence': 0.7
                        })
        
        return semantic_joins
    
    def _create_default_semantic_profile(self) -> Dict[str, Any]:
        """Create a default semantic profile when analysis fails"""
        return {
            'domain_entities': ['general'],
            'data_types': ['general'],
            'relationship_types': ['basic'],
            'column_semantics': {},
            'business_purpose': 'Unknown',
            'confidence_score': 0.1
        }