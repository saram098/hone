"""
Smart caching system for ARC problems
Caches solutions for similar/identical problems to improve efficiency score
"""

import hashlib
import json
from typing import List, Dict, Optional
from loguru import logger
import time


class ARCCache:
    """
    LRU cache for ARC problem solutions
    """
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.ttl = ttl  # Time to live in seconds
        self.hits = 0
        self.misses = 0
        
    def _hash_problem(self, train_examples: List[Dict], test_input: List[List[int]]) -> str:
        """Create hash of problem for cache key"""
        # Convert to JSON string for hashing
        problem_data = {
            'train': train_examples,
            'test': test_input
        }
        problem_str = json.dumps(problem_data, sort_keys=True)
        return hashlib.sha256(problem_str.encode()).hexdigest()
    
    def _hash_pattern(self, train_examples: List[Dict]) -> str:
        """Hash only the pattern (for pattern-based caching)"""
        pattern_str = json.dumps(train_examples, sort_keys=True)
        return hashlib.sha256(pattern_str.encode()).hexdigest()
    
    def get(self, train_examples: List[Dict], test_input: List[List[int]]) -> Optional[List[List[int]]]:
        """Get cached solution if available"""
        # Try exact match first
        key = self._hash_problem(train_examples, test_input)
        
        if key in self.cache:
            entry = self.cache[key]
            # Check if expired
            if time.time() - entry['timestamp'] < self.ttl:
                self.hits += 1
                self.access_times[key] = time.time()
                logger.info(f"💾 Cache HIT! (hit rate: {self.get_hit_rate():.1%})")
                return entry['solution']
            else:
                # Expired
                del self.cache[key]
                del self.access_times[key]
        
        self.misses += 1
        return None
    
    def put(self, train_examples: List[Dict], test_input: List[List[int]], solution: List[List[int]]):
        """Store solution in cache"""
        key = self._hash_problem(train_examples, test_input)
        
        # Evict oldest if cache full
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        self.cache[key] = {
            'solution': solution,
            'timestamp': time.time()
        }
        self.access_times[key] = time.time()
        
        logger.debug(f"💾 Cached solution (cache size: {len(self.cache)}/{self.max_size})")
    
    def _evict_oldest(self):
        """Evict least recently used item"""
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
        logger.debug(f"💾 Evicted old cache entry")
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate"""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.get_hit_rate()
        }
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.access_times.clear()
        logger.info("💾 Cache cleared")


# Global cache instance
_global_cache = ARCCache(max_size=1000, ttl=3600)


def get_cached_solution(train_examples: List[Dict], test_input: List[List[int]]) -> Optional[List[List[int]]]:
    """Get solution from global cache"""
    return _global_cache.get(train_examples, test_input)


def cache_solution(train_examples: List[Dict], test_input: List[List[int]], solution: List[List[int]]):
    """Store solution in global cache"""
    _global_cache.put(train_examples, test_input, solution)


def get_cache_stats() -> Dict:
    """Get cache statistics"""
    return _global_cache.get_stats()

