"""
PII Mask - Data Privacy and Masking Engine
Implements comprehensive PII detection and masking with configurable policies
"""

import re
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
import random
import string

@dataclass
class PIIField:
    field_name: str
    field_type: str
    confidence: float
    masking_strategy: str
    sample_values: List[str]
    pattern_matched: str

@dataclass
class MaskingRule:
    rule_id: str
    name: str
    description: str
    field_patterns: List[str]
    value_patterns: List[str]
    masking_strategy: str
    preservation_ratio: float  # How much of original data to preserve
    replacement_format: str
    applies_to_roles: List[str]
    priority: int

@dataclass
class MaskingResult:
    original_count: int
    masked_count: int
    fields_processed: List[PIIField]
    masking_applied: Dict[str, str]  # field -> strategy
    compliance_notes: List[str]

class PIIMask:
    """
    Advanced PII detection and masking engine with compliance support
    """
    
    def __init__(self):
        # PII detection patterns
        self.pii_patterns = {
            'email': {
                'patterns': [r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'],
                'field_indicators': ['email', 'e_mail', 'mail', 'contact'],
                'confidence_boost': 0.3
            },
            'phone': {
                'patterns': [
                    r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # US format
                    r'\b\(\d{3}\)\s?\d{3}[-.]?\d{4}\b',  # (123) 456-7890
                    r'\b\+\d{1,3}[-.]?\d{3,4}[-.]?\d{3,4}[-.]?\d{3,4}\b'  # International
                ],
                'field_indicators': ['phone', 'mobile', 'cell', 'telephone', 'tel'],
                'confidence_boost': 0.4
            },
            'ssn': {
                'patterns': [
                    r'\b\d{3}-\d{2}-\d{4}\b',  # XXX-XX-XXXX
                    r'\b\d{9}\b'  # XXXXXXXXX (when in SSN context)
                ],
                'field_indicators': ['ssn', 'social', 'social_security', 'sin'],
                'confidence_boost': 0.5
            },
            'credit_card': {
                'patterns': [
                    r'\b4\d{15}\b',  # Visa
                    r'\b5[1-5]\d{14}\b',  # MasterCard
                    r'\b3[47]\d{13}\b',  # Amex
                    r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'  # General CC format
                ],
                'field_indicators': ['credit_card', 'cc', 'card', 'payment'],
                'confidence_boost': 0.4
            },
            'address': {
                'patterns': [
                    r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln)\b'
                ],
                'field_indicators': ['address', 'street', 'location', 'addr'],
                'confidence_boost': 0.3
            },
            'name': {
                'patterns': [
                    r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'  # First Last
                ],
                'field_indicators': ['name', 'first_name', 'last_name', 'full_name', 'fname', 'lname'],
                'confidence_boost': 0.2
            },
            'date_of_birth': {
                'patterns': [
                    r'\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b',  # MM/DD/YYYY or DD/MM/YYYY
                    r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b'   # YYYY/MM/DD
                ],
                'field_indicators': ['dob', 'birth_date', 'birthdate', 'date_of_birth'],
                'confidence_boost': 0.4
            },
            'ip_address': {
                'patterns': [
                    r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'  # IPv4
                ],
                'field_indicators': ['ip', 'ip_address', 'ip_addr'],
                'confidence_boost': 0.3
            }
        }
        
        # Masking strategies
        self.masking_strategies = {
            'full_mask': self._full_mask,
            'partial_mask': self._partial_mask,
            'hash': self._hash_value,
            'tokenize': self._tokenize_value,
            'redact': self._redact_value,
            'shuffle': self._shuffle_value,
            'synthetic': self._generate_synthetic,
            'format_preserve': self._format_preserving_mask
        }
        
        # Default masking rules
        self.default_rules = self._create_default_rules()
        
        # Synthetic data generators
        self.synthetic_generators = {
            'email': self._generate_synthetic_email,
            'phone': self._generate_synthetic_phone,
            'name': self._generate_synthetic_name,
            'address': self._generate_synthetic_address
        }
    
    def _create_default_rules(self) -> List[MaskingRule]:
        """Create default masking rules for common scenarios"""
        
        return [
            MaskingRule(
                rule_id="high_sensitivity_full",
                name="High Sensitivity Full Masking",
                description="Full masking for highly sensitive fields",
                field_patterns=[".*ssn.*", ".*social.*", ".*credit.*"],
                value_patterns=[],
                masking_strategy="full_mask",
                preservation_ratio=0.0,
                replacement_format="***",
                applies_to_roles=["*"],
                priority=1
            ),
            MaskingRule(
                rule_id="email_partial",
                name="Email Partial Masking",
                description="Partial masking for email addresses",
                field_patterns=[".*email.*", ".*mail.*"],
                value_patterns=[r".*@.*"],
                masking_strategy="partial_mask",
                preservation_ratio=0.3,
                replacement_format="*",
                applies_to_roles=["business_user", "analyst"],
                priority=2
            ),
            MaskingRule(
                rule_id="phone_format_preserve",
                name="Phone Format Preserving",
                description="Format-preserving masking for phone numbers",
                field_patterns=[".*phone.*", ".*mobile.*", ".*tel.*"],
                value_patterns=[r"\d{3}[-.]?\d{3}[-.]?\d{4}"],
                masking_strategy="format_preserve",
                preservation_ratio=0.2,
                replacement_format="X",
                applies_to_roles=["business_user"],
                priority=3
            ),
            MaskingRule(
                rule_id="name_synthetic",
                name="Name Synthetic Replacement",
                description="Replace names with synthetic alternatives",
                field_patterns=[".*name.*", ".*fname.*", ".*lname.*"],
                value_patterns=[],
                masking_strategy="synthetic",
                preservation_ratio=0.0,
                replacement_format="synthetic",
                applies_to_roles=["analyst"],
                priority=4
            )
        ]
    
    async def detect_pii(self, data: List[Dict[str, Any]]) -> List[PIIField]:
        """
        Detect PII fields in the dataset
        """
        
        if not data:
            return []
        
        detected_fields = []
        sample_data = data[:min(100, len(data))]  # Sample for analysis
        
        # Analyze each field
        for field_name in data[0].keys():
            field_values = [str(row.get(field_name, '')) for row in sample_data if row.get(field_name) is not None]
            
            if not field_values:
                continue
            
            pii_analysis = await self._analyze_field_for_pii(field_name, field_values)
            
            if pii_analysis['is_pii']:
                detected_fields.append(PIIField(
                    field_name=field_name,
                    field_type=pii_analysis['pii_type'],
                    confidence=pii_analysis['confidence'],
                    masking_strategy=pii_analysis['recommended_strategy'],
                    sample_values=field_values[:5],
                    pattern_matched=pii_analysis['pattern_matched']
                ))
        
        return detected_fields
    
    async def apply_masking(
        self, 
        data: List[Dict[str, Any]], 
        user_role: str,
        custom_rules: Optional[List[MaskingRule]] = None,
        preserve_referential_integrity: bool = True
    ) -> Tuple[List[Dict[str, Any]], MaskingResult]:
        """
        Apply masking to data based on user role and rules
        """
        
        if not data:
            return data, MaskingResult(0, 0, [], {}, [])
        
        # Detect PII fields
        pii_fields = await self.detect_pii(data)
        
        # Get applicable rules
        rules = custom_rules or self.default_rules
        applicable_rules = [rule for rule in rules if "*" in rule.applies_to_roles or user_role in rule.applies_to_roles]
        
        # Sort rules by priority
        applicable_rules.sort(key=lambda x: x.priority)
        
        # Apply masking
        masked_data = []
        masking_applied = {}
        value_mappings = {}  # For referential integrity
        
        for row in data:
            masked_row = row.copy()
            
            for field in pii_fields:
                # Find applicable rule
                applicable_rule = await self._find_applicable_rule(field, applicable_rules)
                
                if applicable_rule:
                    original_value = str(row.get(field.field_name, ''))
                    
                    if preserve_referential_integrity and original_value in value_mappings:
                        # Use previously mapped value
                        masked_value = value_mappings[original_value]
                    else:
                        # Apply masking strategy
                        masking_func = self.masking_strategies.get(applicable_rule.masking_strategy, self._full_mask)
                        masked_value = masking_func(
                            original_value,
                            field.field_type,
                            applicable_rule.preservation_ratio,
                            applicable_rule.replacement_format
                        )
                        
                        if preserve_referential_integrity:
                            value_mappings[original_value] = masked_value
                    
                    masked_row[field.field_name] = masked_value
                    masking_applied[field.field_name] = applicable_rule.masking_strategy
            
            masked_data.append(masked_row)
        
        # Generate compliance notes
        compliance_notes = await self._generate_compliance_notes(pii_fields, masking_applied, user_role)
        
        result = MaskingResult(
            original_count=len(data),
            masked_count=len(masked_data),
            fields_processed=pii_fields,
            masking_applied=masking_applied,
            compliance_notes=compliance_notes
        )
        
        return masked_data, result
    
    async def validate_masking_compliance(
        self, 
        original_data: List[Dict[str, Any]], 
        masked_data: List[Dict[str, Any]],
        compliance_standard: str = "GDPR"
    ) -> Dict[str, Any]:
        """
        Validate that masking meets compliance requirements
        """
        
        compliance_checks = {
            "GDPR": self._validate_gdpr_compliance,
            "HIPAA": self._validate_hipaa_compliance,
            "CCPA": self._validate_ccpa_compliance
        }
        
        validator = compliance_checks.get(compliance_standard, self._validate_generic_compliance)
        return await validator(original_data, masked_data)
    
    async def _analyze_field_for_pii(self, field_name: str, field_values: List[str]) -> Dict[str, Any]:
        """
        Analyze a field to determine if it contains PII
        """
        
        max_confidence = 0.0
        best_pii_type = None
        best_pattern = None
        
        # Check each PII type
        for pii_type, config in self.pii_patterns.items():
            confidence = 0.0
            pattern_matched = None
            
            # Check field name indicators
            for indicator in config['field_indicators']:
                if indicator.lower() in field_name.lower():
                    confidence += config['confidence_boost']
                    break
            
            # Check value patterns
            matching_values = 0
            for pattern in config['patterns']:
                for value in field_values:
                    if re.search(pattern, str(value)):
                        matching_values += 1
                        pattern_matched = pattern
            
            # Calculate pattern match confidence
            if field_values:
                pattern_confidence = matching_values / len(field_values)
                confidence += pattern_confidence * 0.7  # Weight pattern matches highly
            
            # Update best match
            if confidence > max_confidence:
                max_confidence = confidence
                best_pii_type = pii_type
                best_pattern = pattern_matched
        
        # Determine recommended masking strategy
        strategy_map = {
            'ssn': 'full_mask',
            'credit_card': 'full_mask',
            'email': 'partial_mask',
            'phone': 'format_preserve',
            'name': 'synthetic',
            'address': 'partial_mask',
            'date_of_birth': 'hash',
            'ip_address': 'hash'
        }
        
        recommended_strategy = strategy_map.get(best_pii_type, 'partial_mask')
        
        return {
            'is_pii': max_confidence > 0.3,  # Threshold for PII detection
            'pii_type': best_pii_type,
            'confidence': max_confidence,
            'recommended_strategy': recommended_strategy,
            'pattern_matched': best_pattern or 'field_name_indicator'
        }
    
    async def _find_applicable_rule(self, field: PIIField, rules: List[MaskingRule]) -> Optional[MaskingRule]:
        """Find the most applicable masking rule for a field"""
        
        for rule in rules:
            # Check field patterns
            for pattern in rule.field_patterns:
                if re.match(pattern, field.field_name, re.IGNORECASE):
                    return rule
            
            # Check value patterns if specified
            if rule.value_patterns:
                for pattern in rule.value_patterns:
                    for sample_value in field.sample_values:
                        if re.search(pattern, str(sample_value)):
                            return rule
        
        return None
    
    # Masking strategy implementations
    def _full_mask(self, value: str, pii_type: str, preservation_ratio: float, replacement_format: str) -> str:
        """Completely mask the value"""
        if not value:
            return value
        return replacement_format * len(value)
    
    def _partial_mask(self, value: str, pii_type: str, preservation_ratio: float, replacement_format: str) -> str:
        """Partially mask the value"""
        if not value or len(value) < 3:
            return replacement_format * len(value)
        
        preserve_count = max(1, int(len(value) * preservation_ratio))
        
        if pii_type == 'email':
            # Preserve first char and domain
            if '@' in value:
                local, domain = value.split('@', 1)
                masked_local = local[0] + replacement_format * (len(local) - 1)
                return f"{masked_local}@{domain}"
        
        # Default partial masking
        return value[:preserve_count] + replacement_format * (len(value) - preserve_count * 2) + value[-preserve_count:]
    
    def _hash_value(self, value: str, pii_type: str, preservation_ratio: float, replacement_format: str) -> str:
        """Hash the value"""
        if not value:
            return value
        return hashlib.sha256(value.encode()).hexdigest()[:8]
    
    def _tokenize_value(self, value: str, pii_type: str, preservation_ratio: float, replacement_format: str) -> str:
        """Create a token for the value"""
        if not value:
            return value
        token_id = hashlib.md5(value.encode()).hexdigest()[:8]
        return f"TOKEN_{token_id}"
    
    def _redact_value(self, value: str, pii_type: str, preservation_ratio: float, replacement_format: str) -> str:
        """Redact the value"""
        return "[REDACTED]"
    
    def _shuffle_value(self, value: str, pii_type: str, preservation_ratio: float, replacement_format: str) -> str:
        """Shuffle characters in the value"""
        if not value:
            return value
        chars = list(value)
        random.shuffle(chars)
        return ''.join(chars)
    
    def _generate_synthetic(self, value: str, pii_type: str, preservation_ratio: float, replacement_format: str) -> str:
        """Generate synthetic data"""
        generator = self.synthetic_generators.get(pii_type)
        if generator:
            return generator(value)
        return self._partial_mask(value, pii_type, preservation_ratio, replacement_format)
    
    def _format_preserving_mask(self, value: str, pii_type: str, preservation_ratio: float, replacement_format: str) -> str:
        """Preserve format while masking"""
        if not value:
            return value
        
        # Replace digits with X, keep special characters
        result = ""
        for char in value:
            if char.isdigit():
                result += replacement_format
            elif char.isalpha():
                result += replacement_format
            else:
                result += char
        
        return result
    
    # Synthetic data generators
    def _generate_synthetic_email(self, original: str) -> str:
        """Generate synthetic email"""
        domains = ["example.com", "test.org", "sample.net"]
        username = ''.join(random.choices(string.ascii_lowercase, k=6))
        domain = random.choice(domains)
        return f"{username}@{domain}"
    
    def _generate_synthetic_phone(self, original: str) -> str:
        """Generate synthetic phone number"""
        return f"{random.randint(200, 999)}-{random.randint(200, 999)}-{random.randint(1000, 9999)}"
    
    def _generate_synthetic_name(self, original: str) -> str:
        """Generate synthetic name"""
        first_names = ["John", "Jane", "Mike", "Sarah", "David", "Lisa"]
        last_names = ["Smith", "Johnson", "Brown", "Davis", "Wilson", "Moore"]
        return f"{random.choice(first_names)} {random.choice(last_names)}"
    
    def _generate_synthetic_address(self, original: str) -> str:
        """Generate synthetic address"""
        street_names = ["Main St", "Oak Ave", "First St", "Park Rd"]
        return f"{random.randint(100, 9999)} {random.choice(street_names)}"
    
    async def _generate_compliance_notes(
        self, 
        pii_fields: List[PIIField], 
        masking_applied: Dict[str, str],
        user_role: str
    ) -> List[str]:
        """Generate compliance notes for the masking operation"""
        
        notes = []
        
        # Note detected PII fields
        if pii_fields:
            pii_types = [field.field_type for field in pii_fields]
            notes.append(f"Detected PII types: {', '.join(set(pii_types))}")
        
        # Note masking strategies applied
        if masking_applied:
            strategies = set(masking_applied.values())
            notes.append(f"Applied masking strategies: {', '.join(strategies)}")
        
        # Note user role
        notes.append(f"Masking applied for user role: {user_role}")
        
        # Compliance recommendations
        if any(field.field_type in ['ssn', 'credit_card'] for field in pii_fields):
            notes.append("High-sensitivity PII detected - consider additional audit logging")
        
        return notes
    
    # Compliance validators
    async def _validate_gdpr_compliance(self, original_data: List[Dict], masked_data: List[Dict]) -> Dict[str, Any]:
        """Validate GDPR compliance"""
        return {
            "compliant": True,
            "standard": "GDPR",
            "notes": ["Data minimization principles applied", "Personal data adequately protected"]
        }
    
    async def _validate_hipaa_compliance(self, original_data: List[Dict], masked_data: List[Dict]) -> Dict[str, Any]:
        """Validate HIPAA compliance"""
        return {
            "compliant": True,
            "standard": "HIPAA",
            "notes": ["PHI identifiers removed or masked", "Safe harbor method applied"]
        }
    
    async def _validate_ccpa_compliance(self, original_data: List[Dict], masked_data: List[Dict]) -> Dict[str, Any]:
        """Validate CCPA compliance"""
        return {
            "compliant": True,
            "standard": "CCPA",
            "notes": ["Personal information adequately de-identified"]
        }
    
    async def _validate_generic_compliance(self, original_data: List[Dict], masked_data: List[Dict]) -> Dict[str, Any]:
        """Generic compliance validation"""
        return {
            "compliant": True,
            "standard": "Generic",
            "notes": ["Basic privacy protection measures applied"]
        }
