"""
Enhanced N-gram and Substring Matching for Table Names
Provides precise matching using 2-3 word combinations
"""
import re
from typing import List, Dict, Tuple
from difflib import SequenceMatcher

class EnhancedTableMatcher:
    def __init__(self):
        self.stopwords = {'the', 'and', 'or', 'of', 'to', 'in', 'a', 'an', 'is', 'are', 'was', 'were'}
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text"""
        # Convert to lowercase and handle underscores and camelCase
        # First, replace underscores with spaces for proper word splitting
        text = text.replace('_', ' ')
        
        # Handle camelCase by inserting spaces before capital letters
        import re
        # Insert space before capital letters that follow lowercase letters
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # Now extract words
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        # Remove stopwords and short words
        keywords = [w for w in words if len(w) > 2 and w not in self.stopwords]
        
        return keywords
    
    def generate_ngrams(self, keywords: List[str], n: int = 2) -> List[str]:
        """Generate n-grams from keywords"""
        if len(keywords) < n:
            return keywords
        
        ngrams = []
        for i in range(len(keywords) - n + 1):
            ngram = '_'.join(keywords[i:i+n])
            ngrams.append(ngram)
        
        return ngrams
    
    def calculate_ngram_similarity(self, query: str, table_name: str) -> float:
        """Calculate similarity using n-gram matching"""
        # Extract keywords
        query_keywords = self.extract_keywords(query)
        table_keywords = self.extract_keywords(table_name)
        
        if not query_keywords or not table_keywords:
            return 0.0
        
        # Generate 2-grams and 3-grams
        query_2grams = self.generate_ngrams(query_keywords, 2)
        query_3grams = self.generate_ngrams(query_keywords, 3)
        
        table_2grams = self.generate_ngrams(table_keywords, 2)
        table_3grams = self.generate_ngrams(table_keywords, 3)
        
        # Calculate matches
        total_score = 0.0
        total_weight = 0.0
        
        # Exact keyword matches (highest weight)
        exact_matches = len(set(query_keywords) & set(table_keywords))
        total_score += exact_matches * 3.0
        total_weight += len(query_keywords) * 3.0
        
        # 2-gram matches (medium weight)
        if query_2grams and table_2grams:
            bigram_matches = len(set(query_2grams) & set(table_2grams))
            total_score += bigram_matches * 2.0
            total_weight += len(query_2grams) * 2.0
        
        # 3-gram matches (high weight)
        if query_3grams and table_3grams:
            trigram_matches = len(set(query_3grams) & set(table_3grams))
            total_score += trigram_matches * 2.5
            total_weight += len(query_3grams) * 2.5
        
        # Substring matching (lower weight)
        substring_score = 0.0
        for q_word in query_keywords:
            for t_word in table_keywords:
                if q_word in t_word or t_word in q_word:
                    substring_score += 1.0
        total_score += substring_score * 1.0
        total_weight += len(query_keywords) * len(table_keywords) * 0.1
        
        # Avoid division by zero
        if total_weight == 0:
            return 0.0
        
        return min(total_score / total_weight, 1.0)
    
    def rank_tables_by_similarity(self, query: str, table_names: List[str], top_k: int = 5) -> List[Dict[str, any]]:
        """Rank tables by n-gram similarity"""
        results = []
        
        for table_name in table_names:
            similarity = self.calculate_ngram_similarity(query, table_name)
            
            if similarity > 0.1:  # Only include tables with some relevance
                # Get matching details
                query_keywords = self.extract_keywords(query)
                table_keywords = self.extract_keywords(table_name)
                common_keywords = list(set(query_keywords) & set(table_keywords))
                
                results.append({
                    'table_name': table_name,
                    'similarity_score': similarity,
                    'confidence': self._score_to_confidence(similarity),
                    'matching_keywords': common_keywords,
                    'match_type': 'ngram_similarity',
                    'reasons': [f"Keyword matches: {', '.join(common_keywords)}" if common_keywords else "Partial substring matches"]
                })
        
        # Sort by similarity score
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return results[:top_k]
    
    def _score_to_confidence(self, score: float) -> str:
        """Convert similarity score to confidence level"""
        if score >= 0.7:
            return "high"
        elif score >= 0.4:
            return "medium"
        elif score >= 0.2:
            return "low"
        else:
            return "very_low"
