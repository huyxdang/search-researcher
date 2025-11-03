# Advanced semantic search with query expansion and hybrid capabilities
# scripts/4_advanced_search.py

import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, Range
from openai import OpenAI
import re
from dataclasses import dataclass
from enum import Enum
import os
from dotenv import load_dotenv

load_dotenv()

class SearchMode(Enum):
    PURE_VECTOR = "pure_vector"
    HYBRID = "hybrid"
    FILTERED = "filtered"

@dataclass
class SearchQuery:
    """Structured search query with parsed components"""
    raw_query: str
    expanded_query: str
    mode: SearchMode
    filters: Dict
    boost_terms: List[str]
    semantic_concepts: List[str]
    exclude_filters: Dict  # For negation (NOT conditions)
    location_constraints: List[str]  # Explicit location requirements

class QueryParser:
    """Parse and understand search queries"""
    
    def __init__(self):
        self.nationality_terms = {
            'vietnamese': ['vietnam', 'hanoi', 'ho chi minh', 'hcmc', 'vinai', 'vnu'],
            'chinese': ['china', 'beijing', 'shanghai', 'tsinghua', 'chinese'],
            'korean': ['korea', 'seoul', 'kaist', 'korean'],
            'indian': ['india', 'bangalore', 'mumbai', 'iit', 'indian'],
            'american': ['usa', 'united states', 'mit', 'stanford', 'american'],
            'japanese': ['japan', 'tokyo', 'kyoto', 'japanese']
        }
        
        self.research_terms = {
            'nlp': ['natural language', 'nlp', 'text processing', 'language model', 'linguistics'],
            'vision': ['computer vision', 'image', 'visual', 'cnn', 'object detection'],
            'gui': ['gui agents', 'interface', 'ui automation', 'web agents', 'browser automation'],
            'multimodal': ['multimodal', 'vision-language', 'cross-modal', 'clip', 'vl'],
            'rl': ['reinforcement learning', 'rl', 'agent', 'reward', 'policy gradient'],
            'llm': ['large language model', 'llm', 'gpt', 'transformer', 'chatbot'],
            'robotics': ['robotics', 'robot', 'navigation', 'manipulation', 'ros']
        }
        
        self.temporal_terms = {
            'recent': ['recent', 'new', 'latest', 'current', 'nowadays'],
            'established': ['established', 'long-term', 'veteran', 'experienced'],
            'transition': ['switched', 'moved from', 'transitioned', 'shifted', 'changed from']
        }
    
    def parse(self, query: str) -> SearchQuery:
        """Parse query into structured components"""
        query_lower = query.lower()
        
        # Detect search mode
        mode = self._detect_mode(query_lower)
        
        # Extract filters
        filters = self._extract_filters(query_lower)
        
        # Extract exclude filters (negation)
        exclude_filters = self._extract_exclude_filters(query_lower)
        
        # Extract location constraints
        location_constraints = self._extract_location_constraints(query_lower)
        
        # Extract boost terms
        boost_terms = self._extract_boost_terms(query_lower)
        
        # Extract semantic concepts
        semantic_concepts = self._extract_semantic_concepts(query_lower)
        
        # Expand query
        expanded_query = self._expand_query(query, semantic_concepts, boost_terms)
        
        return SearchQuery(
            raw_query=query,
            expanded_query=expanded_query,
            mode=mode,
            filters=filters,
            boost_terms=boost_terms,
            semantic_concepts=semantic_concepts,
            exclude_filters=exclude_filters,
            location_constraints=location_constraints
        )
    
    def _detect_mode(self, query: str) -> SearchMode:
        """Detect the appropriate search mode"""
        # Check for specific filter indicators (strong filters)
        if any(term in query for term in ['exactly', 'must be', 'only', 'at least', 'more than', 'fewer than']):
            return SearchMode.FILTERED
        
        # Check for negation (requires filtering)
        if any(term in query for term in ['not', 'excluding', 'except', 'but not', 'without']):
            return SearchMode.FILTERED
        
        # Check for hybrid indicators (multiple conditions)
        if any(term in query for term in ['and', 'with', 'who also', 'also working', 'who are']):
            return SearchMode.HYBRID
        
        # Check for location constraints (should use hybrid for better matching)
        location_patterns = [' in ', ' from ', ' based in ', ' working in ', ' at ']
        if any(pattern in query.lower() for pattern in location_patterns):
            return SearchMode.HYBRID
        
        return SearchMode.PURE_VECTOR
    
    def _extract_filters(self, query: str) -> Dict:
        """Extract hard filters from query"""
        filters = {}
        
        # Paper count filters
        paper_count_patterns = [
            (r'(\d+)\+ papers', 'min'),
            (r'at least (\d+) papers', 'min'),
            (r'more than (\d+) papers', 'min'),
            (r'fewer than (\d+) papers', 'max'),
            (r'less than (\d+) papers', 'max')
        ]
        
        for pattern, filter_type in paper_count_patterns:
            match = re.search(pattern, query)
            if match:
                count = int(match.group(1))
                if filter_type == 'min':
                    filters['min_papers'] = count
                else:
                    filters['max_papers'] = count
        
        # Year filters
        year_match = re.search(r'(since|after|before|in) (20\d{2})', query)
        if year_match:
            relation = year_match.group(1)
            year = int(year_match.group(2))
            if relation in ['since', 'after']:
                filters['min_year'] = year
            elif relation == 'before':
                filters['max_year'] = year
            elif relation == 'in':
                filters['year'] = year
        
        return filters
    
    def _extract_exclude_filters(self, query: str) -> Dict:
        """Extract negation/exclusion filters (NOT conditions)"""
        exclude_filters = {}
        
        # Pattern for "NOT X", "excluding X", "except X", "but not X"
        negation_patterns = [
            (r'not\s+(?:a\s+)?(\w+)', 'general'),  # "not postdocs", "not a professor"
            (r'excluding\s+(\w+)', 'general'),
            (r'except\s+(\w+)', 'general'),
            (r'but\s+not\s+(\w+)', 'general'),
            (r'without\s+(\w+)', 'general'),
        ]
        
        for pattern, filter_type in negation_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                excluded_term = match.group(1).lower()
                
                # Check if it's a nationality
                for nationality, terms in self.nationality_terms.items():
                    if any(excluded_term in term or term in excluded_term for term in [nationality] + terms[:2]):
                        exclude_filters['nationality'] = nationality
                        break
        
        return exclude_filters
    
    def _extract_location_constraints(self, query: str) -> List[str]:
        """Extract explicit location constraints"""
        locations = []
        
        # Patterns for location specification
        location_patterns = [
            r'\bin\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',  # "in US", "in the US", "in New York"
            r'\bfrom\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',  # "from China"
            r'\bbased\s+in\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',  # "based in US"
            r'\bworking\s+in\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',  # "working in US"
            r'\bat\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',  # "at MIT", "at Stanford"
        ]
        
        for pattern in location_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                location = match.group(1).strip()
                # Normalize common variations
                location = location.replace('United States', 'USA').replace('United States of America', 'USA')
                location = location.replace('Vietnam', 'Vietnam').replace('Viet Nam', 'Vietnam')
                if location and len(location) < 50:  # Reasonable location length
                    locations.append(location)
        
        # Also check for explicit country mentions
        country_terms = {
            'USA': ['usa', 'united states', 'us', 'america', 'american'],
            'Vietnam': ['vietnam', 'viet nam', 'vn'],
            'China': ['china', 'chinese', 'prc'],
            'India': ['india', 'indian'],
            'Korea': ['korea', 'south korea', 'korean'],
            'Japan': ['japan', 'japanese'],
        }
        
        query_lower = query.lower()
        for country, variations in country_terms.items():
            if any(var in query_lower for var in variations):
                if country not in locations:
                    locations.append(country)
        
        return locations
    
    def _extract_boost_terms(self, query: str) -> List[str]:
        """Extract terms that should be boosted in results"""
        boost_terms = []
        
        # Add nationality boost terms
        for nationality, terms in self.nationality_terms.items():
            if nationality in query:
                boost_terms.extend(terms)
        
        # Add research area boost terms
        for area, terms in self.research_terms.items():
            if any(term in query for term in [area] + terms[:2]):  # Check main term + first 2 variants
                boost_terms.extend(terms)
        
        return list(set(boost_terms))  # Remove duplicates
    
    def _extract_semantic_concepts(self, query: str) -> List[str]:
        """Extract high-level semantic concepts"""
        concepts = []
        
        # Check nationalities
        for nationality, terms in self.nationality_terms.items():
            if any(term in query for term in [nationality] + terms[:3]):
                concepts.append(f"nationality:{nationality}")
        
        # Check research areas
        for area, terms in self.research_terms.items():
            if any(term in query for term in [area] + terms[:2]):
                concepts.append(f"research:{area}")
        
        # Check temporal aspects
        for temporal, terms in self.temporal_terms.items():
            if any(term in query for term in terms):
                concepts.append(f"temporal:{temporal}")
        
        return concepts
    
    def _expand_query(self, original: str, concepts: List[str], 
                     boost_terms: List[str]) -> str:
        """Expand query with related terms and concepts"""
        expanded_parts = [original]
        
        # Add concept expansions
        for concept in concepts:
            concept_type, concept_value = concept.split(':')
            
            if concept_type == 'nationality':
                if concept_value in self.nationality_terms:
                    # Add location-specific terms
                    expanded_parts.append(' '.join(self.nationality_terms[concept_value][:3]))
            
            elif concept_type == 'research':
                if concept_value in self.research_terms:
                    # Add research-specific terms
                    expanded_parts.append(' '.join(self.research_terms[concept_value][:3]))
            
        # Add boost terms (but don't duplicate)
        unique_boost = [term for term in boost_terms[:5] if term not in original.lower()]
        if unique_boost:
            expanded_parts.append(' '.join(unique_boost))
        
        return ' '.join(expanded_parts)

class SemanticAuthorSearch:
    """Advanced semantic search system for academic authors"""
    
    def __init__(self, 
                 collection_name: str = "authors",
                 embedding_model: str = "text-embedding-3-large"):
        
        self.qdrant = QdrantClient(path="./qdrant_data")
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.query_parser = QueryParser()
        
        # Load author profiles for additional processing
        profiles_path = 'data/author_profiles_enriched.json'
        if os.path.exists(profiles_path):
            with open(profiles_path, 'r') as f:
                self.profiles = json.load(f)
                self.profile_lookup = {p['name']: p for p in self.profiles}
        else:
            # Fallback to basic profiles
            profiles_path = 'data/author_profiles.json'
            if os.path.exists(profiles_path):
                with open(profiles_path, 'r') as f:
                    self.profiles = json.load(f)
                    self.profile_lookup = {p['name']: p for p in self.profiles}
            else:
                self.profiles = []
                self.profile_lookup = {}
    
    def _embed_text(self, text: str) -> List[float]:
        """Generate embedding using OpenAI"""
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding
    
    def search(self, 
               query: str, 
               limit: int = 20,
               mode: Optional[SearchMode] = None) -> List[Dict]:
        """
        Perform semantic search with query understanding
        """
        # Parse query
        parsed_query = self.query_parser.parse(query)
        
        if mode:
            parsed_query.mode = mode
        
        # Get initial results based on mode
        if parsed_query.mode == SearchMode.PURE_VECTOR:
            results = self._pure_vector_search(parsed_query, limit * 2)
        elif parsed_query.mode == SearchMode.HYBRID:
            results = self._hybrid_search(parsed_query, limit * 2)
        else:  # FILTERED
            results = self._filtered_search(parsed_query, limit * 2)
        
        # Apply boost scoring
        if parsed_query.boost_terms:
            results = self._apply_boost_scoring(results, parsed_query.boost_terms)
        
        # Apply exclusion filters (remove results matching exclude criteria)
        if parsed_query.exclude_filters:
            results = self._apply_exclude_filters(results, parsed_query.exclude_filters)
        
        # Apply location constraints (boost results matching location)
        if parsed_query.location_constraints:
            results = self._apply_location_constraints(results, parsed_query.location_constraints)
        
        # Re-rank results
        results = self._rerank_results(results, parsed_query, limit)
        
        # Enhance with explanations
        results = self._add_explanations(results, parsed_query)
        
        return results[:limit]
    
    def _pure_vector_search(self, query: SearchQuery, limit: int) -> List[Dict]:
        """Standard vector similarity search"""
        # Embed expanded query
        query_vector = self._embed_text(query.expanded_query)
        
        # Search
        search_results = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )
        
        results = []
        for hit in search_results:
            author_name = hit.payload.get('name')
            profile = self.profile_lookup.get(author_name, {})
            
            results.append({
                'name': author_name,
                'score': hit.score,
                'profile_text': hit.payload.get('profile_text', ''),
                'metadata': profile,
                'match_type': 'semantic'
            })
        
        return results
    
    def _hybrid_search(self, query: SearchQuery, limit: int) -> List[Dict]:
        """Combine vector search with keyword matching"""
        # Get vector results
        vector_results = self._pure_vector_search(query, limit)
        
        # Apply keyword filtering/boosting
        for result in vector_results:
            keyword_score = self._calculate_keyword_score(
                result['profile_text'], 
                query.boost_terms
            )
            # Combine scores (70% vector, 30% keyword)
            result['hybrid_score'] = (0.7 * result['score']) + (0.3 * keyword_score)
            result['match_type'] = 'hybrid'
        
        # Re-sort by hybrid score
        vector_results.sort(key=lambda x: x.get('hybrid_score', 0), reverse=True)
        
        return vector_results
    
    def _filtered_search(self, query: SearchQuery, limit: int) -> List[Dict]:
        """Apply hard filters before vector search"""
        # Build Qdrant filter conditions
        filter_conditions = []
        
        if 'min_papers' in query.filters:
            filter_conditions.append(
                FieldCondition(
                    key="paper_count",
                    range=Range(gte=query.filters['min_papers'])
                )
            )
        
        if 'max_papers' in query.filters:
            filter_conditions.append(
                FieldCondition(
                    key="paper_count",
                    range=Range(lte=query.filters['max_papers'])
                )
            )
        
        # Create filter
        search_filter = Filter(must=filter_conditions) if filter_conditions else None
        
        # Embed query
        query_vector = self._embed_text(query.expanded_query)
        
        # Search with filters
        search_results = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=search_filter,
            limit=limit
        )
        
        results = []
        for hit in search_results:
            author_name = hit.payload.get('name')
            profile = self.profile_lookup.get(author_name, {})
            
            results.append({
                'name': author_name,
                'score': hit.score,
                'profile_text': hit.payload.get('profile_text', ''),
                'metadata': profile,
                'match_type': 'filtered'
            })
        
        return results
    
    def _calculate_keyword_score(self, text: str, keywords: List[str]) -> float:
        """Calculate keyword matching score"""
        if not keywords:
            return 0.0
        
        text_lower = text.lower()
        matches = sum(1 for keyword in keywords if keyword.lower() in text_lower)
        return matches / len(keywords)
    
    def _apply_boost_scoring(self, results: List[Dict], 
                            boost_terms: List[str]) -> List[Dict]:
        """Apply boost to results containing specific terms"""
        for result in results:
            boost_score = self._calculate_keyword_score(
                result['profile_text'], 
                boost_terms
            )
            # Apply boost (up to 20% increase)
            result['boosted_score'] = result.get('score', 0) * (1 + 0.2 * boost_score)
        
        return results
    
    def _apply_exclude_filters(self, results: List[Dict], 
                               exclude_filters: Dict) -> List[Dict]:
        """Remove results that match exclusion criteria"""
        filtered_results = []
        
        for result in results:
            should_exclude = False
            metadata = result.get('metadata', {})
            
            # Exclude by nationality
            if 'nationality' in exclude_filters:
                excluded_nat = exclude_filters['nationality']
                nationality_signals = metadata.get('nationality_signals', [])
                if excluded_nat in nationality_signals:
                    should_exclude = True
            
            if not should_exclude:
                filtered_results.append(result)
        
        return filtered_results
    
    def _apply_location_constraints(self, results: List[Dict], 
                                    location_constraints: List[str]) -> List[Dict]:
        """Boost results that match location constraints"""
        for result in results:
            location_boost = 0
            metadata = result.get('metadata', {})
            profile_text = result.get('profile_text', '').lower()
            
            # Check affiliations and locations
            affiliations = ' '.join(metadata.get('affiliations', [])).lower()
            locations = ' '.join(metadata.get('locations', [])).lower()
            all_location_text = ' '.join([profile_text, affiliations, locations])
            
            for constraint in location_constraints:
                constraint_lower = constraint.lower()
                # Boost if location appears in profile
                if constraint_lower in all_location_text:
                    location_boost += 0.15
                
                # Extra boost for exact matches in locations field
                if constraint_lower in locations:
                    location_boost += 0.1
            
            # Apply location boost (up to 30% total)
            current_score = result.get('boosted_score', result.get('score', 0))
            result['location_boosted_score'] = current_score * (1 + min(location_boost, 0.3))
        
        # Re-sort by location-boosted score
        results.sort(key=lambda x: x.get('location_boosted_score', x.get('boosted_score', x.get('score', 0))), reverse=True)
        
        return results
    
    def _rerank_results(self, results: List[Dict], query: SearchQuery, 
                       limit: int) -> List[Dict]:
        """Re-rank results based on multiple factors"""
        for result in results:
            # Calculate final score (use location_boosted_score if available)
            base_score = result.get('location_boosted_score', 
                                   result.get('boosted_score', 
                                            result.get('score', 0)))
            
            # Boost for concept matches
            concept_boost = 0
            for concept in query.semantic_concepts:
                concept_type, concept_value = concept.split(':')
                
                if concept_type == 'nationality':
                    if concept_value in result['metadata'].get('nationality_signals', []):
                        concept_boost += 0.1
                
                elif concept_type == 'research':
                    # Check if research area appears in recent papers
                    recent_focus = result['metadata'].get('research_evolution', {}).get('recent_focus', [])
                    if any(concept_value in focus.lower() for focus in recent_focus):
                        concept_boost += 0.15
            
            result['final_score'] = base_score * (1 + concept_boost)
        
        # Sort by final score
        results.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        
        return results
    
    def _add_explanations(self, results: List[Dict], 
                         query: SearchQuery) -> List[Dict]:
        """Add explanations for why each result matched"""
        for result in results:
            explanations = []
            metadata = result.get('metadata', {})
            
            # Explain nationality signals
            for concept in query.semantic_concepts:
                if concept.startswith('nationality:'):
                    nationality = concept.split(':')[1]
                    if nationality in metadata.get('nationality_signals', []):
                        explanations.append(f"✓ Nationality indicator: {nationality}")
            
            # Explain research match
            for concept in query.semantic_concepts:
                if concept.startswith('research:'):
                    area = concept.split(':')[1]
                    evolution = metadata.get('research_evolution', {})
                    if any(area in str(focus).lower() for focus in evolution.get('recent_focus', [])):
                        explanations.append(f"✓ Research area: {area}")
            
            # Explain transition if relevant
            if 'transition' in query.raw_query.lower():
                evolution = metadata.get('research_evolution', {})
                if evolution.get('transition'):
                    early = ', '.join(evolution.get('early_focus', []))
                    recent = ', '.join(evolution.get('recent_focus', []))
                    explanations.append(f"✓ Research transition: {early} → {recent}")
            
            result['explanations'] = explanations
            result['relevance_summary'] = self._generate_relevance_summary(result, query)
        
        return results
    
    def _generate_relevance_summary(self, result: Dict, query: SearchQuery) -> str:
        """Generate a brief summary of why this result is relevant"""
        metadata = result.get('metadata', {})
        
        parts = []
        
        # Add years active
        years_active = metadata.get('years_active', 0)
        parts.append(f"Active researcher ({years_active} years)")
        
        # Add location if relevant
        if metadata.get('affiliations'):
            parts.append(f"at {metadata['affiliations'][0]}")
        
        # Add research focus
        evolution = metadata.get('research_evolution', {})
        if evolution.get('recent_focus'):
            focus = ', '.join(evolution['recent_focus'][:2])
            parts.append(f"researching {focus}")
        
        # Add metrics
        papers = metadata.get('paper_count', 0)
        citations = metadata.get('citation_count', 0)
        if citations > 100:
            parts.append(f"{papers} papers, {citations} citations")
        else:
            parts.append(f"{papers} papers")
        
        return ' | '.join(parts)

def demo_search():
    """Demonstrate the advanced search capabilities"""
    # Initialize search engine
    search = SemanticAuthorSearch()
    
    # Example queries showcasing different capabilities
    test_queries = [
        "Vietnamese researchers in the US working on GUI agents",
        "Researchers who transitioned from NLP to computer vision",
        "Researchers with 50+ papers in multimodal learning",
        "Chinese researchers working on reinforcement learning since 2022",
        "Researchers at VinAI or KAIST focusing on web automation",
        "Researchers who moved from robotics to LLMs",
        "Vietnamese researchers in US",
        "Researchers from China or India working on LLMs"
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}")
        
        results = search.search(query, limit=5)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['name']}")
            print(f"   Score: {result.get('final_score', result.get('score')):.3f}")
            print(f"   Match Type: {result.get('match_type')}")
            print(f"   Summary: {result.get('relevance_summary', 'N/A')}")
            
            if result.get('explanations'):
                print(f"   Why matched:")
                for explanation in result['explanations']:
                    print(f"     {explanation}")

if __name__ == "__main__":
    demo_search()