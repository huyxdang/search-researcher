# Enhanced profile builder with rich semantic signals
# scripts/2_build_profiles_enhanced.py

import json
from collections import defaultdict
from openai import OpenAI
from semanticscholar import SemanticScholar  # type: ignore
import os
import time
import re
from dotenv import load_dotenv
from typing import Dict, List, Optional, Set
from datetime import datetime

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize Semantic Scholar client
semantic_scholar_key = os.getenv('SEMANTIC_SCHOLAR_API_KEY')
sch = SemanticScholar(api_key=semantic_scholar_key) if semantic_scholar_key else SemanticScholar()

# Cache for API calls
author_cache = {}
institution_location_map = {}

# Location inference patterns
LOCATION_PATTERNS = {
    'vietnam': {
        'institutions': ['VinAI', 'VNU', 'HUST', 'HCMUT', 'Vietnam National University', 
                        'Hanoi University', 'Ho Chi Minh City University'],
        'cities': ['Hanoi', 'Ho Chi Minh', 'HCMC', 'Da Nang', 'Can Tho'],
        'country_variations': ['Vietnam', 'Viet Nam', 'VN'],
        'email_domains': ['.vn', 'vinai.io', 'vnu.edu.vn'],
        'name_patterns': ['Nguyen', 'Tran', 'Pham', 'Le', 'Hoang', 'Vo', 'Dang', 'Bui']
    },
    'chinese': {
        'institutions': ['Tsinghua', 'Peking University', 'CAS', 'USTC', 'Zhejiang University',
                        'Shanghai Jiao Tong', 'Fudan'],
        'cities': ['Beijing', 'Shanghai', 'Shenzhen', 'Hangzhou', 'Guangzhou'],
        'country_variations': ['China', 'CN', 'PRC'],
        'email_domains': ['.cn', 'tsinghua.edu.cn', 'pku.edu.cn'],
        'name_patterns': ['Wang', 'Zhang', 'Li', 'Liu', 'Chen', 'Yang', 'Zhao', 'Wu']
    },
    'korean': {
        'institutions': ['KAIST', 'Seoul National', 'Yonsei', 'POSTECH', 'SNU'],
        'cities': ['Seoul', 'Daejeon', 'Pohang', 'Busan'],
        'country_variations': ['Korea', 'South Korea', 'KR'],
        'email_domains': ['.kr', 'kaist.ac.kr', 'snu.ac.kr'],
        'name_patterns': ['Kim', 'Lee', 'Park', 'Choi', 'Jung', 'Kang', 'Cho', 'Yoon']
    },
    'indian': {
        'institutions': ['IIT', 'IISc', 'IIIT', 'ISI', 'TIFR'],
        'cities': ['Bangalore', 'Mumbai', 'Delhi', 'Chennai', 'Hyderabad', 'Pune'],
        'country_variations': ['India', 'IN'],
        'email_domains': ['.in', 'iisc.ac.in', 'iitb.ac.in'],
        'name_patterns': ['Kumar', 'Singh', 'Sharma', 'Patel', 'Gupta', 'Reddy']
    }
}

def infer_nationality_signals(author_name: str, affiliations: List[str], 
                             locations: Set[str]) -> Dict[str, List[str]]:
    """
    Infer nationality/geographic signals from multiple sources
    """
    signals = defaultdict(list)
    
    for region, patterns in LOCATION_PATTERNS.items():
        confidence_score = 0
        region_signals = []
        
        # Check name patterns (weak signal)
        for name_pattern in patterns['name_patterns']:
            if name_pattern in author_name:
                confidence_score += 0.3
                region_signals.append(f"name pattern suggests {region} origin ({name_pattern})")
                break
        
        # Check institutions (strong signal)
        for affiliation in affiliations:
            affiliation_lower = affiliation.lower()
            for inst_pattern in patterns['institutions']:
                if inst_pattern.lower() in affiliation_lower:
                    confidence_score += 1.0
                    region_signals.append(f"affiliated with {region} institution ({inst_pattern})")
                    break
        
        # Check cities (strong signal)
        for location in locations:
            location_lower = location.lower()
            for city in patterns['cities']:
                if city.lower() in location_lower:
                    confidence_score += 1.0
                    region_signals.append(f"based in {city}, {patterns['country_variations'][0]}")
                    break
        
        # Check country variations (strong signal)
        for location in locations:
            for country_var in patterns['country_variations']:
                if country_var.lower() in location.lower():
                    confidence_score += 1.0
                    region_signals.append(f"located in {country_var}")
                    break
        
        if confidence_score >= 0.5:  # Threshold for including signals
            signals[region] = region_signals
    
    return signals

def extract_research_evolution(papers: List[Dict]) -> Dict[str, any]:
    """
    Analyze how research topics evolved over time
    """
    evolution = {}
    
    # Sort papers by year
    papers_sorted = sorted(papers, key=lambda p: p['published'])
    
    # Split into early and recent periods
    midpoint = len(papers_sorted) // 2
    early_papers = papers_sorted[:midpoint]
    recent_papers = papers_sorted[midpoint:]
    
    # Extract topics using simple keyword extraction
    def extract_topics(paper_list):
        topics = defaultdict(int)
        for paper in paper_list:
            # Extract from title and abstract
            text = (paper['title'] + ' ' + paper['abstract']).lower()
            
            # Common ML/AI topics to look for
            topic_keywords = {
                'nlp': ['language', 'nlp', 'text', 'translation', 'bert', 'transformer', 'linguistic'],
                'computer vision': ['vision', 'image', 'visual', 'cnn', 'detection', 'segmentation'],
                'reinforcement learning': ['reinforcement', 'rl', 'agent', 'reward', 'policy'],
                'gui agents': ['gui', 'interface', 'ui', 'automation', 'web agent', 'browser'],
                'multimodal': ['multimodal', 'cross-modal', 'vision-language', 'clip'],
                'llm': ['language model', 'llm', 'gpt', 'chatbot', 'dialogue'],
                'robotics': ['robot', 'robotic', 'navigation', 'manipulation'],
                'graphs': ['graph', 'network', 'gnn', 'knowledge graph'],
                'generative': ['generative', 'gan', 'diffusion', 'synthesis', 'generation']
            }
            
            for topic, keywords in topic_keywords.items():
                if any(kw in text for kw in keywords):
                    topics[topic] += 1
        
        # Return top topics
        return [topic for topic, count in sorted(topics.items(), 
                key=lambda x: x[1], reverse=True)[:3]]
    
    early_topics = extract_topics(early_papers)
    recent_topics = extract_topics(recent_papers)
    
    evolution['early_focus'] = early_topics
    evolution['recent_focus'] = recent_topics
    
    # Detect transitions
    topics_added = set(recent_topics) - set(early_topics)
    topics_dropped = set(early_topics) - set(recent_topics)
    
    if topics_added or topics_dropped:
        evolution['transition'] = True
        evolution['topics_added'] = list(topics_added)
        evolution['topics_dropped'] = list(topics_dropped)
    else:
        evolution['transition'] = False
        evolution['consistent'] = True
    
    return evolution

def build_enriched_author_profile(author_name: str, papers: List[Dict]) -> Dict:
    """
    Build a semantically rich author profile with multiple inference signals
    """
    # Sort papers by date
    papers_sorted = sorted(papers, key=lambda p: p['published'], reverse=True)
    
    # Basic metrics
    paper_count = len(papers)
    years = [int(p['published'][:4]) for p in papers]
    first_year = min(years)
    last_year = max(years)
    years_active = last_year - first_year + 1
    
    # Fetch Semantic Scholar data (with verification)
    author_info = fetch_author_info_from_semantic_scholar(author_name, papers)
    
    # Extract affiliations and locations
    affiliations = author_info.get('affiliations', []) if author_info else []
    locations = set(author_info.get('locations', [])) if author_info else set()
    citation_count = author_info.get('citation_count', 0) if author_info else 0
    
    # Infer nationality signals
    nationality_signals = infer_nationality_signals(author_name, affiliations, locations)
    
    # Extract research evolution (only if we have enough papers for meaningful analysis)
    # Note: Career stage inference removed - not needed for research topic and location search
    if paper_count >= 3:
        research_evolution = extract_research_evolution(papers)
    else:
        # For authors with 1-2 papers, use simplified evolution
        research_evolution = {
            'early_focus': [],
            'recent_focus': [],
            'consistent': True,
            'transition': False
        }
    
    # Build comprehensive profile text
    profile_parts = []
    
    # Opening with name and basic info
    profile_parts.append(f"{author_name} is a researcher in computer science.")
    
    # Add affiliation and location (with redundancy for better matching)
    if affiliations:
        profile_parts.append(f"Currently affiliated with {affiliations[0]}.")
        for affil in affiliations[1:3]:  # Add up to 2 more affiliations
            profile_parts.append(f"Also associated with {affil}.")
    
    # Add nationality/geographic signals
    for region, signals in nationality_signals.items():
        for signal in signals:
            profile_parts.append(signal.capitalize() + ".")
    
    # Add location redundancy
    if locations:
        loc_list = list(locations)[:3]
        profile_parts.append(f"Geographic markers: {', '.join(loc_list)}.")
    
    # Publication metrics
    profile_parts.append(f"Published {paper_count} papers from {first_year} to {last_year}.")
    if citation_count > 0:
        profile_parts.append(f"Total citations: {citation_count}.")
    
    # Research evolution
    if research_evolution.get('transition'):
        profile_parts.append(f"Research evolution: Early work focused on {', '.join(research_evolution['early_focus'])}.")
        profile_parts.append(f"Recent work focuses on {', '.join(research_evolution['recent_focus'])}.")
        if research_evolution.get('topics_added'):
            profile_parts.append(f"Shifted towards: {', '.join(research_evolution['topics_added'])}.")
    else:
        all_topics = research_evolution.get('early_focus', []) + research_evolution.get('recent_focus', [])
        unique_topics = list(set(all_topics))
        if unique_topics:
            profile_parts.append(f"Consistent research focus on: {', '.join(unique_topics)}.")
    
    # Add collaboration patterns
    all_coauthors = set()
    for paper in papers_sorted:
        all_coauthors.update([a for a in paper['authors'] if a != author_name])
    
    if all_coauthors:
        frequent_collaborators = list(all_coauthors)[:5]
        profile_parts.append(f"Frequent collaborators: {', '.join(frequent_collaborators)}.")
    
    # Paper titles and abstracts for semantic richness
    profile_parts.append("\nKey publications:")
    for i, paper in enumerate(papers_sorted[:10], 1):
        year = paper['published'][:4]
        profile_parts.append(f"{i}. '{paper['title']}' ({year})")
        # Include abstract snippet for semantic matching
        abstract_snippet = paper['abstract'][:200].replace('\n', ' ')
        profile_parts.append(f"   Research on: {abstract_snippet}...")
    
    # Generate LLM summary for research focus
    try:
        paper_info = "\n".join([
            f"- {p['title']} ({p['published'][:4]}): {p['abstract'][:100]}..."
            for p in papers_sorted[:10]
        ])
        
        summary_prompt = f"""
        Based on these publications, provide a detailed 3-4 sentence description of {author_name}'s research focus and contributions.
        Include specific technical areas, methodologies, and application domains.
        
        Publications:
        {paper_info}
        
        Write in a factual, technical style that captures their research identity.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=200,
            temperature=0.3
        )
        research_summary = response.choices[0].message.content
        
        # Insert research summary early in profile
        profile_parts.insert(2, f"Research focus: {research_summary}")
        
    except Exception as e:
        print(f"Error generating LLM summary for {author_name}: {e}")
    
    # Compile final profile
    profile_text = "\n".join(profile_parts)
    
    # Add semantic keywords at the end for better matching
    semantic_keywords = []
    
    # Add nationality keywords
    for region in nationality_signals.keys():
        if region in LOCATION_PATTERNS:
            semantic_keywords.extend(LOCATION_PATTERNS[region]['country_variations'])
    
    if semantic_keywords:
        profile_text += f"\n\nSemantic markers: {', '.join(semantic_keywords)}."
    
    return {
        'name': author_name,
        'profile_text': profile_text,
        'paper_count': paper_count,
        'first_year': first_year,
        'last_year': last_year,
        'years_active': years_active,
        'nationality_signals': list(nationality_signals.keys()),
        'research_evolution': research_evolution,
        'affiliations': affiliations,
        'locations': list(locations),
        'citation_count': citation_count,
        'papers': papers_sorted,
        'metadata': {  # Add metadata about Semantic Scholar match
            'semantic_scholar_found': author_info is not None,
            'verified': author_info.get('verified', False) if author_info else False,
            'overlap_ratio': author_info.get('overlap_ratio', 0.0) if author_info else 0.0
        }
    }

def check_paper_overlap(arxiv_papers: List[Dict], s2_papers: List) -> float:
    """
    Check overlap between ArXiv papers and Semantic Scholar papers
    Returns overlap ratio (0.0 to 1.0)
    """
    if not arxiv_papers or not s2_papers:
        return 0.0
    
    # Extract ArXiv paper titles (normalized)
    arxiv_titles = set()
    for paper in arxiv_papers:
        title = paper.get('title', '').lower().strip()
        if title:
            arxiv_titles.add(title)
    
    # Extract S2 paper titles
    s2_titles = set()
    for paper in s2_papers[:20]:  # Check first 20 papers
        title = getattr(paper, 'title', None) or (paper.get('title') if isinstance(paper, dict) else None)
        if title:
            s2_titles.add(title.lower().strip())
    
    if not arxiv_titles or not s2_titles:
        return 0.0
    
    # Calculate overlap
    overlap = len(arxiv_titles & s2_titles)
    total_unique = len(arxiv_titles | s2_titles)
    
    return overlap / total_unique if total_unique > 0 else 0.0

def fetch_author_info_from_semantic_scholar(author_name: str, arxiv_papers: List[Dict] = None) -> Optional[Dict]:
    """
    Enhanced version with verification by checking paper overlap
    """
    if author_name in author_cache:
        return author_cache[author_name]
    
    try:
        # Search for author - get multiple candidates
        search_results = sch.search_author(author_name, limit=5)
        
        if not search_results:
            author_cache[author_name] = None
            return None
        
        # If we have ArXiv papers, verify the match
        best_match = None
        best_overlap = 0.0
        
        if arxiv_papers and len(arxiv_papers) > 0:
            # Check each candidate
            for candidate in search_results:
                author_id = candidate.authorId if hasattr(candidate, 'authorId') else candidate.get('authorId')
                if not author_id:
                    continue
                
                try:
                    # Get author details with papers
                    author_details = sch.get_author(author_id, fields=['papers'])
                    
                    # Get papers if available
                    s2_papers = []
                    if hasattr(author_details, 'papers'):
                        s2_papers = list(author_details.papers) if author_details.papers else []
                    elif isinstance(author_details, dict) and 'papers' in author_details:
                        s2_papers = author_details['papers']
                    
                    # Check overlap
                    overlap = check_paper_overlap(arxiv_papers, s2_papers)
                    
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_match = author_details
                        
                except Exception as e:
                    continue
            
            # Require minimum 20% overlap for verification, or use first result if no verification
            if best_overlap < 0.2:
                if len(search_results) > 0:
                    # Use first result but mark as unverified
                    author_id = search_results[0].authorId if hasattr(search_results[0], 'authorId') else search_results[0].get('authorId')
                    if author_id:
                        best_match = sch.get_author(author_id)
                        best_overlap = 0.0  # Mark as unverified
                else:
                    author_cache[author_name] = None
                    return None
        else:
            # No ArXiv papers for verification, use first result
            author_id = search_results[0].authorId if hasattr(search_results[0], 'authorId') else search_results[0].get('authorId')
            if author_id:
                best_match = sch.get_author(author_id)
            else:
                author_cache[author_name] = None
                return None
        
        if not best_match:
            author_cache[author_name] = None
            return None
        
        # Extract all available information
        affiliations = []
        locations = set()
        
        # Handle affiliations
        if hasattr(best_match, 'affiliations'):
            for affil in best_match.affiliations or []:
                if isinstance(affil, str):
                    affiliations.append(affil)
                elif hasattr(affil, 'name'):
                    affiliations.append(affil.name)
        elif isinstance(best_match, dict) and 'affiliations' in best_match:
            affiliations = best_match['affiliations']
        
        # Extract locations from affiliations
        for affil in affiliations:
            # Common patterns
            parts = affil.split(',')
            if len(parts) > 1:
                # Last part often contains location
                location = parts[-1].strip()
                if location and len(location) < 50:
                    locations.add(location)
            
            # Look for country indicators
            for country in ['USA', 'UK', 'China', 'India', 'Vietnam', 'Korea', 'Japan', 
                          'Germany', 'France', 'Canada', 'Australia']:
                if country in affil:
                    locations.add(country)
        
        # Get metrics
        citation_count = getattr(best_match, 'citationCount', 0) if hasattr(best_match, 'citationCount') else best_match.get('citationCount', 0)
        paper_count = getattr(best_match, 'paperCount', 0) if hasattr(best_match, 'paperCount') else best_match.get('paperCount', 0)
        h_index = getattr(best_match, 'hIndex', 0) if hasattr(best_match, 'hIndex') else best_match.get('hIndex', 0)
        
        result = {
            'affiliations': affiliations,
            'locations': list(locations),
            'citation_count': citation_count,
            'paper_count_s2': paper_count,
            'h_index': h_index,
            'verified': best_overlap >= 0.2,  # Track if verified
            'overlap_ratio': best_overlap
        }
        
        author_cache[author_name] = result
        time.sleep(0.3)  # Rate limiting
        
        return result
        
    except Exception as e:
        print(f"Error fetching data for {author_name}: {e}")
        author_cache[author_name] = None
        return None

def group_papers_by_author(papers: List[Dict]) -> Dict[str, List[Dict]]:
    """Group papers by author name"""
    author_papers = defaultdict(list)
    
    for paper in papers:
        for author in paper['authors']:
            author_papers[author].append(paper)
    
    return author_papers

if __name__ == "__main__":
    # Load papers
    with open('data/papers.json', 'r') as f:
        papers = json.load(f)
    
    print(f"Loaded {len(papers)} papers")
    
    # Group by author
    author_papers = group_papers_by_author(papers)
    print(f"Found {len(author_papers)} unique authors")
    
    # Filter authors with 1+ papers (no minimum threshold)
    author_papers_filtered = {
        author: papers 
        for author, papers in author_papers.items() 
        if len(papers) >= 1
    }
    print(f"Authors with 1+ papers: {len(author_papers_filtered)}")
    
    # Build enriched profiles
    profiles = []
    for i, (author, papers) in enumerate(author_papers_filtered.items()):
        try:
            profile = build_enriched_author_profile(author, papers)
            profiles.append(profile)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(author_papers_filtered)} authors...")
                
                # Sample output for monitoring
                if i == 9:  # Show first profile as example
                    print("\n--- Sample Profile ---")
                    print(f"Author: {profile['name']}")
                    print(f"Nationality Signals: {profile['nationality_signals']}")
                    print(f"Research Evolution: {profile['research_evolution']}")
                    print("--- End Sample ---\n")
                    
        except Exception as e:
            print(f"Error processing {author}: {e}")
            continue
    
    # Save profiles
    with open('data/author_profiles_enriched.json', 'w') as f:
        json.dump(profiles, f, indent=2)
    
    print(f"\nSaved {len(profiles)} enriched author profiles")
    
    # Calculate Semantic Scholar coverage statistics
    semantic_scholar_found = sum(1 for p in profiles if p.get('affiliations'))
    semantic_scholar_verified = sum(1 for p in profiles 
                                    if p.get('affiliations') and 
                                    p.get('metadata', {}).get('verified', False))
    
    coverage_rate = (semantic_scholar_found / len(profiles) * 100) if profiles else 0
    verification_rate = (semantic_scholar_verified / semantic_scholar_found * 100) if semantic_scholar_found > 0 else 0
    
    print("\n" + "="*60)
    print("📊 Semantic Scholar Coverage Statistics")
    print("="*60)
    print(f"Total profiles: {len(profiles)}")
    print(f"Profiles with Semantic Scholar data: {semantic_scholar_found} ({coverage_rate:.1f}%)")
    print(f"Verified matches (20%+ paper overlap): {semantic_scholar_verified} ({verification_rate:.1f}% of found)")
    print(f"Profiles without Semantic Scholar data: {len(profiles) - semantic_scholar_found}")
    print("="*60)
    
    # Print statistics
    nationalities = defaultdict(int)
    
    for profile in profiles:
        for nat in profile.get('nationality_signals', []):
            nationalities[nat] += 1
    
    print("\n--- Profile Statistics ---")
    print("\nInferred Nationalities:")
    for nat, count in nationalities.items():
        print(f"  {nat}: {count}")