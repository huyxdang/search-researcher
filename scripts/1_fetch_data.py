# Download from ArXiv
# scripts/1_fetch_data.py
import arxiv
import json
from datetime import datetime
import sys
import os
import time

# Load categories from JSON file
config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts/categories.json')
with open(config_path, 'r') as f:
    config = json.load(f)
    CATEGORIES = config['categories']
    START_YEAR = config['start_year']

def fetch_papers_by_category(
    category: str,
    start_year: int,
    max_results_per_category: int = None  # None = fetch all available
):
    """
    Fetch papers from a single ArXiv category, year by year to avoid API pagination issues.
    Date filtering happens in the query, not after fetching.
    Note: Duplicate handling within category AND across categories is done here and in fetch_arxiv_papers()
    """
    all_papers = []
    seen_ids = set()  # Track duplicate paper IDs within this category
    
    current_year = datetime.now().year
    years_to_fetch = range(start_year, current_year + 1)
    
    print(f"  Fetching {category} (years {start_year}-{current_year})...")
    
    # Use Client for newer API with rate limiting
    client = arxiv.Client(
        page_size=100,
        delay_seconds=3.0,  # Be nice to ArXiv API
        num_retries=3
    )
    
    for year in years_to_fetch:
        for month in range(1, 13):  # 12 months
            papers_this_month = []
            
            # Calculate month boundaries
            days_in_month = 31 if month in [1,3,5,7,8,10,12] else 30 if month != 2 else 29 if year % 4 == 0 else 28
            
            date_from = f"{year}{month:02d}01000000"
            date_to = f"{year}{month:02d}{days_in_month}235959"
            
            query_string = f"cat:{category} AND submittedDate:[{date_from} TO {date_to}]"
            
            # Retry logic for UnexpectedEmptyPageError
            max_retries = 3
            retry_delay = 5.0  # seconds
            fetch_successful = False
            
            for attempt in range(max_retries):
                try:
                    search = arxiv.Search(
                        query=query_string,
                        max_results=10000,  # Should be enough per year for any category
                        sort_by=arxiv.SortCriterion.SubmittedDate,
                        sort_order=arxiv.SortOrder.Descending
                    )
                    
                    count = 0
                    papers_before = len(papers_this_month)
                    
                    for result in client.results(search):
                        count += 1
                        
                        arxiv_id = result.entry_id.split('/')[-1]
                        if arxiv_id in seen_ids:
                            continue
                        seen_ids.add(arxiv_id)
                        
                        # Use result.summary instead of result.abstract
                        abstract = result.summary if hasattr(result, 'summary') else getattr(result, 'abstract', '')
                        
                        paper = {
                            'arxiv_id': arxiv_id,
                            'title': result.title,
                            'abstract': abstract,
                            'authors': [author.name for author in result.authors],
                            'published': result.published.isoformat(),
                            'categories': list(result.categories),
                            'primary_category': result.primary_category
                        }
                        papers_this_month.append(paper)
                    
                    # If we got through the entire iteration without error, success!
                    fetch_successful = True
                    break
                    
                except arxiv.UnexpectedEmptyPageError:
                    papers_after = len(papers_this_month)
                    papers_fetched = papers_after - papers_before
                    
                    if attempt < max_retries - 1:
                        print(f"    Year {year}, Month {month}: Empty page encountered (got {papers_fetched} papers), retrying in {retry_delay:.1f}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        retry_delay *= 1.5  # Exponential backoff
                    else:
                        # Last attempt failed - try splitting into weeks
                        print(f"    Year {year}, Month {month}: Empty page after {papers_fetched} papers, splitting into weeks...")
                        # Preserve papers already fetched before resetting for week-by-week fetch
                        papers_already_fetched = papers_this_month.copy()
                        papers_this_month = []  # Reset, we'll fetch by weeks
                        
                        # Split month into 4 weeks
                        week_starts = [1, 8, 15, 22]
                        for week_num, week_start in enumerate(week_starts, 1):
                            if week_num < len(week_starts):
                                week_end = week_starts[week_num] - 1
                            else:
                                week_end = days_in_month
                            
                            week_date_from = f"{year}{month:02d}{week_start:02d}000000"
                            week_date_to = f"{year}{month:02d}{week_end:02d}235959"
                            week_query = f"cat:{category} AND submittedDate:[{week_date_from} TO {week_date_to}]"
                            
                            try:
                                week_search = arxiv.Search(
                                    query=week_query,
                                    max_results=10000,
                                    sort_by=arxiv.SortCriterion.SubmittedDate,
                                    sort_order=arxiv.SortOrder.Descending
                                )
                                
                                for result in client.results(week_search):
                                    arxiv_id = result.entry_id.split('/')[-1]
                                    if arxiv_id in seen_ids:
                                        continue
                                    seen_ids.add(arxiv_id)
                                    
                                    abstract = result.summary if hasattr(result, 'summary') else getattr(result, 'abstract', '')
                                    paper = {
                                        'arxiv_id': arxiv_id,
                                        'title': result.title,
                                        'abstract': abstract,
                                        'authors': [author.name for author in result.authors],
                                        'published': result.published.isoformat(),
                                        'categories': list(result.categories),
                                        'primary_category': result.primary_category
                                    }
                                    papers_this_month.append(paper)
                            except arxiv.UnexpectedEmptyPageError:
                                print(f"      Week {week_num} also hit empty page, skipping...")
                                continue
                            except Exception as e:
                                print(f"      Week {week_num} error: {e}, continuing...")
                                continue
                        
                        # Restore papers that were fetched before week splitting
                        papers_this_month = papers_already_fetched + papers_this_month
                        
                        break  # Exit retry loop after trying weeks
                        
                except Exception as e:
                    print(f"    Year {year}, Month {month}: Error - {e}, continuing...")
                    break
            
            if papers_this_month:
                all_papers.extend(papers_this_month)
                print(f"    Year {year}, Month {month}: {len(papers_this_month)} papers (total: {len(all_papers)})")
            else:
                print(f"    Year {year}, Month {month}: 0 papers")
    
    print(f"    ✅ {len(all_papers)} total papers from {category}")
    return all_papers

def fetch_arxiv_papers(
    categories=None,
    start_year=None,
    fetch_all=True  # If True, fetches all available papers by querying each category separately
):
    """
    Fetch papers from ArXiv in specified categories.
    
    Strategy: Fetch each category separately and combine results.
    This avoids the empty page issues that occur with large OR queries.
    
    Args:
        categories: List of arXiv categories. If None, uses categories from categories.py
        start_year: Year to start fetching from. If None, uses START_YEAR from categories.py
        fetch_all: If True, fetches all available papers by querying each category separately
    """
    # Use defaults from categories.py if not provided
    if categories is None:
        categories = CATEGORIES
    if start_year is None:
        start_year = START_YEAR
    
    all_papers = []
    seen_ids = set()  # Track duplicates across categories
    
    print(f"Fetching papers from ArXiv (categories: {', '.join(categories)})...")
    print(f"Start year: {start_year}")
    print(f"Strategy: Fetching each category separately to avoid API issues\n")
    
    # Fetch each category separately
    for category in categories:
        category_papers = fetch_papers_by_category(category, start_year)
        
        # Add papers, avoiding duplicates
        for paper in category_papers:
            if paper['arxiv_id'] not in seen_ids:
                seen_ids.add(paper['arxiv_id'])
                all_papers.append(paper)
        
        print(f"  Total unique papers so far: {len(all_papers)}\n")
    
    print(f"\n{'='*60}")
    print(f"✅ Total unique papers fetched: {len(all_papers)}")
    print(f"{'='*60}")
    
    return all_papers

if __name__ == "__main__":
    # Fetch all papers by querying each category separately
    # This avoids ArXiv API issues with large OR queries
    # Categories and start_year are loaded from categories.py by default
    papers = fetch_arxiv_papers(
        categories=['cs.AI'],
        start_year=2022,
        fetch_all=True  # Fetch all available papers
    )
    
    # Save to JSON
    if papers:
        with open('data/papers.json', 'w') as f:
            json.dump(papers, f, indent=2)
        print(f"\n✅ Saved {len(papers)} unique papers to data/papers.json")
    else:
        print("\n❌ No papers were fetched. Please try again later or check your query.")