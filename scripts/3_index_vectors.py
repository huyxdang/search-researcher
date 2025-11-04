# Creates vector embeddings using OpenAI text-embedding-3-large and indexes author profiles in Qdrant
# Generates multiple embeddings (full profile, research focus, identity) with rich metadata for filtering

import json
import numpy as np
from typing import List, Dict, Any
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    CollectionInfo, PayloadSchemaType,
    TokenizerType, TextIndexParams
)
import hashlib
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv()

class VectorIndexer:
    """Enhanced indexer with OpenAI embeddings and rich metadata"""
    
    def __init__(self, 
                 collection_name: str = "authors",
                 embedding_model: str = "text-embedding-3-large"):
        
        self.collection_name = collection_name
        self.qdrant = QdrantClient(path="./qdrant_data")
        self.embedding_model = embedding_model
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Vector size for text-embedding-3-large is 3072
        self.vector_size = 3072
        
    def _embed_text(self, text: str) -> np.ndarray:
        """Generate embedding using OpenAI"""
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return np.array(response.data[0].embedding)
    
    def create_collection(self, recreate: bool = True):
        """Create Qdrant collection with proper configuration"""
        
        # Check if collection exists
        collections = self.qdrant.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if exists and recreate:
            print(f"Deleting existing collection: {self.collection_name}")
            self.qdrant.delete_collection(self.collection_name)
            exists = False
        
        if not exists:
            print(f"Creating collection: {self.collection_name}")
            
            # Create collection with vector configuration
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                    on_disk=False  # Keep in memory for speed
                ),
            )
            
            # Create payload indexes for filtering
            print("Creating payload indexes...")
            
            # Numeric indexes for range queries
            self.qdrant.create_payload_index(
                collection_name=self.collection_name,
                field_name="paper_count",
                field_schema=PayloadSchemaType.INTEGER
            )
            
            self.qdrant.create_payload_index(
                collection_name=self.collection_name,
                field_name="citation_count",
                field_schema=PayloadSchemaType.INTEGER
            )
            
            self.qdrant.create_payload_index(
                collection_name=self.collection_name,
                field_name="first_year",
                field_schema=PayloadSchemaType.INTEGER
            )
            
            self.qdrant.create_payload_index(
                collection_name=self.collection_name,
                field_name="last_year",
                field_schema=PayloadSchemaType.INTEGER
            )
            
            self.qdrant.create_payload_index(
                collection_name=self.collection_name,
                field_name="years_active",
                field_schema=PayloadSchemaType.INTEGER
            )
            
            # Text index for full-text search on profile
            self.qdrant.create_payload_index(
                collection_name=self.collection_name,
                field_name="profile_text",
                field_schema=TextIndexParams(
                    type="text",
                    tokenizer=TokenizerType.WORD,
                    min_token_len=2,
                    max_token_len=20,
                    lowercase=True
                )
            )
            
            print("Collection created with indexes")
        else:
            print(f"Collection {self.collection_name} already exists")
    
    def generate_enhanced_embedding(self, profile: Dict) -> np.ndarray:
        """
        Generate enhanced embedding by combining multiple strategies
        """
        embeddings = []
        weights = []
        
        # 1. Full profile embedding (highest weight)
        full_text = profile['profile_text']
        full_embedding = self._embed_text(full_text)
        embeddings.append(full_embedding)
        weights.append(0.5)  # 50% weight
        
        # 2. Research focus embedding (important for topical search)
        research_parts = []
        
        # Add recent paper titles
        for paper in profile.get('papers', [])[:5]:
            research_parts.append(paper['title'])
        
        # Add research evolution description
        evolution = profile.get('research_evolution', {})
        if evolution.get('recent_focus'):
            research_parts.append(f"Current research: {', '.join(evolution['recent_focus'])}")
        if evolution.get('early_focus'):
            research_parts.append(f"Previous research: {', '.join(evolution['early_focus'])}")
        
        if research_parts:
            research_text = ' '.join(research_parts)
            research_embedding = self._embed_text(research_text)
            embeddings.append(research_embedding)
            weights.append(0.3)  # 30% weight
        
        # 3. Identity embedding (name, affiliation, location)
        identity_parts = [profile['name']]
        
        if profile.get('affiliations'):
            identity_parts.extend(profile['affiliations'][:2])
        
        if profile.get('locations'):
            identity_parts.extend(profile['locations'][:2])
        
        # Add nationality signals
        if profile.get('nationality_signals'):
            for nationality in profile['nationality_signals']:
                identity_parts.append(f"{nationality} researcher")
        
        identity_text = ' '.join(identity_parts)
        identity_embedding = self._embed_text(identity_text)
        embeddings.append(identity_embedding)
        weights.append(0.2)  # 20% weight
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Combine embeddings with weights
        combined = np.zeros_like(embeddings[0])
        for emb, w in zip(embeddings, weights):
            combined += w * emb
        
        # Normalize final embedding
        combined = combined / np.linalg.norm(combined)
        
        return combined
    
    def prepare_payload(self, profile: Dict) -> Dict[str, Any]:
        """
        Prepare comprehensive payload for storage in Qdrant
        """
        # Extract paper years for filtering
        paper_years = []
        for paper in profile.get('papers', []):
            try:
                year = int(paper['published'][:4])
                paper_years.append(year)
            except:
                continue
        
        # Build payload with all searchable fields
        payload = {
            # Basic info
            'name': profile['name'],
            'profile_text': profile['profile_text'],
            
            # Numeric fields for filtering
            'paper_count': profile.get('paper_count', 0),
            'citation_count': profile.get('citation_count', 0),
            'first_year': profile.get('first_year', 2020),
            'last_year': profile.get('last_year', 2024),
            'years_active': profile.get('years_active', 0),
            'h_index': profile.get('h_index', 0),
            
            # Categorical fields
            
            # List fields
            'affiliations': profile.get('affiliations', []),
            'locations': profile.get('locations', []),
            'nationality_signals': profile.get('nationality_signals', []),
            'paper_years': paper_years,
            
            # Research evolution
            'research_areas': [],
            'research_transition': False
        }
        
        # Extract research areas from evolution
        evolution = profile.get('research_evolution', {})
        if evolution:
            all_areas = set()
            if evolution.get('early_focus'):
                all_areas.update(evolution['early_focus'])
            if evolution.get('recent_focus'):
                all_areas.update(evolution['recent_focus'])
            payload['research_areas'] = list(all_areas)
            payload['research_transition'] = evolution.get('transition', False)
        
        # Add top paper titles for additional context
        top_papers = []
        for paper in profile.get('papers', [])[:5]:
            top_papers.append(paper['title'])
        payload['top_papers'] = top_papers
        
        # Add searchable text summary (for keyword search)
        search_text_parts = [
            profile['name'],
            ' '.join(profile.get('affiliations', [])),
            ' '.join(profile.get('locations', [])),
            ' '.join(profile.get('nationality_signals', [])),
            ' '.join(payload.get('research_areas', [])),
        ]
        payload['search_text'] = ' '.join(filter(None, search_text_parts))
        
        return payload
    
    def index_profiles(self, profiles_path: str, batch_size: int = 100):
        """
        Index all profiles with enhanced embeddings and metadata
        """
        # Load profiles
        print(f"Loading profiles from {profiles_path}")
        with open(profiles_path, 'r') as f:
            profiles = json.load(f)
        
        print(f"Indexing {len(profiles)} profiles...")
        
        # Process in batches
        points = []
        
        for i, profile in enumerate(tqdm(profiles, desc="Generating embeddings")):
            # Generate unique ID
            author_id = hashlib.md5(profile['name'].encode()).hexdigest()
            
            # Generate enhanced embedding
            embedding = self.generate_enhanced_embedding(profile)
            
            # Prepare payload
            payload = self.prepare_payload(profile)
            
            # Create point
            point = PointStruct(
                id=author_id,
                vector=embedding.tolist(),
                payload=payload
            )
            points.append(point)
            
            # Upload in batches
            if len(points) >= batch_size:
                self.qdrant.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                points = []
        
        # Upload remaining points
        if points:
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=points
            )
        
        # Get collection info
        info = self.qdrant.get_collection(self.collection_name)
        print(f"\nIndexing complete!")
        print(f"Collection: {self.collection_name}")
        print(f"Vectors: {info.vectors_count}")
        print(f"Indexed properties: {info.payload_schema.keys() if info.payload_schema else 'N/A'}")
    
    def verify_index(self, sample_queries: List[str] = None):
        """
        Verify the index with sample searches
        """
        if sample_queries is None:
            sample_queries = [
                "Vietnamese researchers in computer vision",
                "PhD students working on reinforcement learning",
                "Senior professors with 50+ papers"
            ]
        
        print("\n" + "="*60)
        print("Verifying index with sample queries...")
        print("="*60)
        
        for query in sample_queries:
            print(f"\nQuery: {query}")
            print("-" * 40)
            
            # Encode query
            query_vector = self._embed_text(query).tolist()
            
            # Search
            results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=3
            )
            
            for i, hit in enumerate(results, 1):
                print(f"\n{i}. {hit.payload['name']}")
                print(f"   Score: {hit.score:.3f}")
                print(f"   Papers: {hit.payload.get('paper_count', 0)}")
                print(f"   Affiliations: {', '.join(hit.payload.get('affiliations', [])[:2])}")
                print(f"   Research: {', '.join(hit.payload.get('research_areas', [])[:3])}")

def main():
    """Main indexing pipeline"""
    
    # Configuration
    profiles_path = 'data/author_profiles_enriched.json'
    
    # Check if enriched profiles exist, otherwise use basic profiles
    if not os.path.exists(profiles_path):
        print(f"Enriched profiles not found at {profiles_path}")
        profiles_path = 'data/author_profiles.json'
        if not os.path.exists(profiles_path):
            print("No profiles found! Please run profile building script first.")
            return
    
    # Initialize indexer
    print("Initializing vector indexer...")
    indexer = VectorIndexer(
        collection_name="authors",
        embedding_model="text-embedding-3-large"
    )
    
    # Create collection
    indexer.create_collection(recreate=True)
    
    # Index profiles
    indexer.index_profiles(profiles_path)
    
    # Verify with test queries
    indexer.verify_index()
    
    print("\n✅ Indexing complete! Ready for semantic search.")

if __name__ == "__main__":
    main()