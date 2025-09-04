"""
Smart Schema Manager for Large Database Embeddings
Handles incremental processing, caching, and optimization
"""

import os
import json
import time
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta

class SmartSchemaManager:
    """Manages large schema embedding with smart optimization"""
    
    def __init__(self, cache_dir: str = "backend/storage"):
        self.cache_dir = cache_dir
        self.progress_file = os.path.join(cache_dir, "embedding_progress.json")
        self.schema_stats_file = os.path.join(cache_dir, "schema_stats.json")
        
    def should_use_incremental_processing(self, total_tables: int, total_items: int) -> bool:
        """Determine if incremental processing should be used"""
        return total_tables > 50 or total_items > 1000
    
    def save_progress(self, processed_tables: List[str], total_tables: int, 
                     current_batch: int, total_batches: int):
        """Save embedding progress for resume capability"""
        progress = {
            "timestamp": datetime.now().isoformat(),
            "processed_tables": processed_tables,
            "total_tables": total_tables,
            "current_batch": current_batch,
            "total_batches": total_batches,
            "completion_percentage": (len(processed_tables) / total_tables) * 100
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def load_progress(self) -> Optional[Dict]:
        """Load previous embedding progress"""
        if not os.path.exists(self.progress_file):
            return None
            
        try:
            with open(self.progress_file, 'r') as f:
                progress = json.load(f)
                
            # Check if progress is recent (within 24 hours)
            timestamp = datetime.fromisoformat(progress['timestamp'])
            if datetime.now() - timestamp > timedelta(hours=24):
                return None
                
            return progress
        except Exception:
            return None
    
    def clear_progress(self):
        """Clear progress file after successful completion"""
        if os.path.exists(self.progress_file):
            os.remove(self.progress_file)
    
    def save_schema_stats(self, stats: Dict):
        """Save schema statistics for optimization"""
        stats['last_updated'] = datetime.now().isoformat()
        
        with open(self.schema_stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def get_recommended_batch_size(self, total_items: int) -> int:
        """Get recommended batch size based on total items"""
        if total_items < 100:
            return 20
        elif total_items < 500:
            return 30
        elif total_items < 1000:
            return 40
        elif total_items < 3000:
            return 50
        else:
            return 75  # For very large schemas
    
    def estimate_processing_time(self, total_items: int, batch_size: int) -> str:
        """Estimate total processing time"""
        # Rough estimates based on OpenAI API performance
        items_per_minute = 200  # Conservative estimate
        total_minutes = total_items / items_per_minute
        
        if total_minutes < 60:
            return f"~{int(total_minutes)} minutes"
        else:
            hours = total_minutes / 60
            return f"~{hours:.1f} hours"
    
    def get_optimization_recommendations(self, total_tables: int, total_items: int) -> List[str]:
        """Get recommendations for optimizing large schema processing"""
        recommendations = []
        
        if total_tables > 100:
            recommendations.append(f"⚠️ Large schema detected ({total_tables} tables)")
            recommendations.append("🎯 Consider using max_tables parameter to limit processing")
            recommendations.append("📋 Use important_tables list to prioritize key tables")
        
        if total_items > 2000:
            recommendations.append(f"⚠️ Large item count ({total_items} items)")
            recommendations.append("🔄 Incremental processing will be used automatically")
            recommendations.append("💾 Progress will be saved for resume capability")
        
        if total_items > 5000:
            recommendations.append("🚀 Consider running during off-peak hours")
            recommendations.append("⏱️ Process may take several hours to complete")
        
        return recommendations

# Usage example function
def optimize_large_schema_embedding(vector_matcher, adapter, total_tables: int):
    """
    Example usage for optimizing large schema embedding
    """
    manager = SmartSchemaManager()
    
    # Check if we should use optimization
    if total_tables > 50:
        print(f"📊 Large schema detected: {total_tables} tables")
        
        # Get recommendations
        estimated_items = total_tables * 21  # Rough estimate
        recommendations = manager.get_optimization_recommendations(total_tables, estimated_items)
        
        for rec in recommendations:
            print(rec)
        
        # Recommended approach for large schemas
        important_tables = [
            # Add your most important table names here
            'NBA_FINAL_OUTPUT_PYTHON_DF',
            # Add more based on your specific schema
        ]
        
        # Use optimized initialization
        vector_matcher.initialize_from_database(
            adapter=adapter,
            force_rebuild=False,  # Use cache if available
            max_tables=100,       # Limit total tables
            important_tables=important_tables
        )
    else:
        # Standard processing for smaller schemas
        vector_matcher.initialize_from_database(adapter)
