"""
FAISS-based similarity search for intelligent table matching
Uses sentence transformers for semantic embeddings
"""
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import os
from typing import List, Tuple, Dict
import re

class FAISSTableMatcher:
    def __init__(self, cache_dir: str = "backend/storage"):
        self.cache_dir = cache_dir
        self.model_name = "all-MiniLM-L6-v2"  # Fast and efficient
        self.model = None
        self.index = None
        self.table_names = []
        self.embeddings_cache_file = os.path.join(cache_dir, "table_embeddings.pkl")
        self.index_cache_file = os.path.join(cache_dir, "faiss_index.bin")
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
    def _load_model(self):
        """Load the sentence transformer model"""
        if self.model is None:
            print("🔧 Loading sentence transformer model...")
            self.model = SentenceTransformer(self.model_name)
            
    def _preprocess_table_name(self, table_name: str) -> str:
        """Preprocess table name for better embeddings"""
        # Replace underscores with spaces
        processed = table_name.replace('_', ' ')
        # Add context keywords for Analytics tables
        if 'analytics' in processed.lower() or 'azure' in processed.lower():
            processed = f"Azure Analytics data {processed}"
        if any(term in processed.lower() for term in ['final', 'output', 'result']):
            processed = f"final results table {processed}"
        if any(term in processed.lower() for term in ['refresh', 'update']):
            processed = f"updated data {processed}"
        return processed
    
    def build_index(self, table_names: List[str], force_rebuild: bool = False):
        """Build FAISS index from table names"""
        self._load_model()
        
        # Check if cached embeddings exist and are valid
        if not force_rebuild and self._load_cached_embeddings(table_names):
            print(f"✅ Loaded cached FAISS index with {len(self.table_names)} tables")
            return
            
        print(f"🔧 Building FAISS index for {len(table_names)} tables...")
        
        # Preprocess table names for better semantic understanding
        processed_names = [self._preprocess_table_name(name) for name in table_names]
        
        # Generate embeddings
        embeddings = self.model.encode(processed_names, show_progress_bar=True)
        embeddings = embeddings.astype('float32')
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.index.add(embeddings)
        
        self.table_names = table_names
        
        # Cache the results
        self._save_cached_embeddings(embeddings)
        print(f"✅ FAISS index built and cached with {len(table_names)} tables")
        
    def _load_cached_embeddings(self, current_table_names: List[str]) -> bool:
        """Load cached embeddings if they exist and are valid"""
        try:
            if not (os.path.exists(self.embeddings_cache_file) and 
                   os.path.exists(self.index_cache_file)):
                return False
                
            with open(self.embeddings_cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                
            cached_table_names = cache_data.get('table_names', [])
            
            # Check if table names match
            if set(cached_table_names) != set(current_table_names):
                print("🔄 Table names changed, rebuilding index...")
                return False
                
            # Load the FAISS index
            self.index = faiss.read_index(self.index_cache_file)
            self.table_names = cached_table_names
            
            return True
            
        except Exception as e:
            print(f"⚠️ Error loading cached embeddings: {e}")
            return False
            
    def _save_cached_embeddings(self, embeddings: np.ndarray):
        """Save embeddings and index to cache"""
        try:
            # Save metadata
            cache_data = {
                'table_names': self.table_names,
                'model_name': self.model_name,
                'embeddings_shape': embeddings.shape
            }
            
            with open(self.embeddings_cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
                
            # Save FAISS index
            faiss.write_index(self.index, self.index_cache_file)
            
        except Exception as e:
            print(f"⚠️ Error saving cached embeddings: {e}")
    
    def find_similar_tables(self, query: str, top_k: int = 5, threshold: float = 0.3) -> List[Dict]:
        """Find similar tables using FAISS semantic search"""
        if self.index is None or self.model is None:
            return []
            
        # Preprocess query
        processed_query = self._preprocess_table_name(query)
        
        # Generate query embedding
        query_embedding = self.model.encode([processed_query]).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS index
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.table_names)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= threshold:  # Filter by similarity threshold
                table_name = self.table_names[idx]
                results.append({
                    'table_name': table_name,
                    'similarity_score': float(score),
                    'match_type': 'semantic',
                    'confidence': self._calculate_confidence(score, query, table_name)
                })
                
        return results
    
    def _calculate_confidence(self, similarity_score: float, query: str, table_name: str) -> str:
        """Calculate confidence level based on similarity score and patterns"""
        confidence_score = similarity_score
        
        # Boost confidence for exact word matches
        query_words = set(re.findall(r'\w+', query.lower()))
        table_words = set(re.findall(r'\w+', table_name.lower()))
        word_overlap = len(query_words.intersection(table_words)) / max(len(query_words), 1)
        
        if word_overlap > 0.5:
            confidence_score += 0.2
            
        # Azure Analytics-specific boosting
        if ('analytics' in query.lower() or 'azure' in query.lower()) and ('analytics' in table_name.lower() or 'azure' in table_name.lower()):
            confidence_score += 0.3
            
        if confidence_score >= 0.8:
            return "very_high"
        elif confidence_score >= 0.6:
            return "high"
        elif confidence_score >= 0.4:
            return "medium"
        else:
            return "low"
    
    def hybrid_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Combine FAISS semantic search with pattern matching"""
        results = []
        
        # Get semantic matches
        semantic_matches = self.find_similar_tables(query, top_k)
        results.extend(semantic_matches)
        
        # Add exact pattern matches (fallback)
        pattern_matches = self._pattern_based_search(query, top_k)
        
        # Combine and deduplicate
        seen_tables = {r['table_name'] for r in results}
        for match in pattern_matches:
            if match['table_name'] not in seen_tables:
                results.append(match)
                
        # Sort by confidence and similarity score
        results.sort(key=lambda x: (
            x.get('similarity_score', 0) if x['match_type'] == 'semantic' else 0.5,
            x['confidence'] == 'very_high',
            x['confidence'] == 'high'
        ), reverse=True)
        
        return results[:top_k]
    
    def _pattern_based_search(self, query: str, top_k: int) -> List[Dict]:
        """Fallback pattern-based search"""
        results = []
        query_words = re.findall(r'\w+', query.lower())
        
        for table_name in self.table_names:
            table_words = re.findall(r'\w+', table_name.lower())
            
            # Count word matches
            matches = sum(1 for word in query_words if word in table_words)
            if matches > 0:
                similarity = matches / len(query_words)
                confidence = "high" if similarity > 0.6 else "medium" if similarity > 0.3 else "low"
                
                results.append({
                    'table_name': table_name,
                    'similarity_score': similarity,
                    'match_type': 'pattern',
                    'confidence': confidence
                })
                
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results[:top_k]
