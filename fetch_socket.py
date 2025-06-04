import xml.etree.ElementTree as ET
import asyncio
import json
import aiohttp
import logging
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
from typing import List, Dict, Any
import time
from flask import Flask
from flask_socketio import SocketIO, emit

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsScraperBroadcaster:
    def __init__(self, port: int = 8765):
        self.port = port
        self.news_data = []
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        self.setup_socket_events()

    def setup_socket_events(self):
        @self.socketio.on('connect')
        def handle_connect():
            logger.info("Client connected")
            if self.news_data:
                emit('initial_data', {
                    'type': 'initial_data',
                    'data': self.news_data[-50:],
                    'total_count': len(self.news_data),
                    'message': f'Connected! Showing latest 50 of {len(self.news_data)} articles'
                })

        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info("Client disconnected")

    def broadcast_batch(self, news_batch: List[Dict[str, Any]]):
        """Broadcast batch of news to all connected clients"""
        if news_batch:
            self.socketio.emit('news_batch', {
                'type': 'news_batch',
                'data': news_batch,
                'count': len(news_batch),
                'timestamp': datetime.now().isoformat()
            })

    async def fetch_sitemap_xml(self, session: aiohttp.ClientSession, sitemap_url: str) -> str:
        """Fetch XML sitemap content from URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            async with session.get(sitemap_url, headers=headers, timeout=15) as response:
                if response.status == 200:
                    content = await response.text()
                    logger.info(f"Successfully fetched sitemap: {sitemap_url}")
                    return content
                else:
                    logger.warning(f"Failed to fetch sitemap {sitemap_url}: Status {response.status}")
                    return ""
                    
        except Exception as e:
            logger.error(f"Error fetching sitemap {sitemap_url}: {e}")
            return ""

    def parse_sitemap_xml(self, xml_content: str, source_url: str = "") -> List[Dict[str, str]]:
        """Parse XML sitemap and extract URLs with metadata"""
        try:
            if not xml_content.strip():
                return []
                
            # Handle XML parsing errors gracefully
            try:
                root = ET.fromstring(xml_content)
            except ET.ParseError as e:
                logger.error(f"XML parsing error for {source_url}: {e}")
                return []
            
            urls = []
            
            # Define namespace mapping
            ns = {
                'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9',
                'news': 'http://www.google.com/schemas/sitemap-news/0.9'
            }
            
            # First try with namespace
            url_elements = root.findall('.//ns:url', ns)
            
            # If no URLs found, try without namespace
            if not url_elements:
                url_elements = root.findall('.//url')
            
            for url_elem in url_elements:
                url_data = {'source_sitemap': source_url}
                
                # Extract loc (required)
                loc_elem = url_elem.find('.//ns:loc', ns) or url_elem.find('.//loc')
                if loc_elem is not None and loc_elem.text:
                    url_data['url'] = loc_elem.text.strip()
                else:
                    continue  # Skip if no URL found
                
                # Extract lastmod
                lastmod_elem = url_elem.find('.//ns:lastmod', ns) or url_elem.find('.//lastmod')
                if lastmod_elem is not None and lastmod_elem.text:
                    url_data['lastmod'] = lastmod_elem.text.strip()
                
                # Extract priority
                priority_elem = url_elem.find('.//ns:priority', ns) or url_elem.find('.//priority')
                if priority_elem is not None and priority_elem.text:
                    url_data['priority'] = priority_elem.text.strip()
                
                # Extract changefreq
                changefreq_elem = url_elem.find('.//ns:changefreq', ns) or url_elem.find('.//changefreq')
                if changefreq_elem is not None and changefreq_elem.text:
                    url_data['changefreq'] = changefreq_elem.text.strip()
                
                # Add to URLs list
                urls.append(url_data)
            
            # Also check for sitemap index
            sitemap_elements = root.findall('.//ns:sitemap', ns) or root.findall('.//sitemap')
            
            nested_urls = []
            for sitemap_elem in sitemap_elements:
                loc_elem = sitemap_elem.find('.//ns:loc', ns) or sitemap_elem.find('.//loc')
                if loc_elem is not None and loc_elem.text:
                    nested_urls.append({
                        'url': loc_elem.text.strip(),
                        'type': 'nested_sitemap',
                        'source_sitemap': source_url
                    })
            
            total_urls = len(urls) + len(nested_urls)
            logger.info(f"Extracted {len(urls)} URLs and {len(nested_urls)} nested sitemaps from {source_url}")
            
            if total_urls == 0:
                logger.warning(f"No URLs found in sitemap: {source_url}")
            
            return urls + nested_urls
            
        except Exception as e:
            logger.error(f"Error parsing sitemap {source_url}: {e}")
            return []

    async def fetch_all_sitemaps(self, sitemap_urls: List[str]) -> List[Dict[str, str]]:
        """Fetch and parse all sitemap URLs"""
        all_urls = []
        nested_sitemaps = []
        
        connector = aiohttp.TCPConnector(limit=20, limit_per_host=10)
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # First, fetch all primary sitemaps
            tasks = [self.fetch_sitemap_xml(session, url) for url in sitemap_urls]
            sitemap_contents = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Parse primary sitemaps
            for i, content in enumerate(sitemap_contents):
                if isinstance(content, str) and content.strip():
                    parsed_urls = self.parse_sitemap_xml(content, sitemap_urls[i])
                    for url_data in parsed_urls:
                        if url_data.get('type') == 'nested_sitemap':
                            nested_sitemaps.append(url_data['url'])
                        else:
                            all_urls.append(url_data)
            
            # Fetch nested sitemaps if any
            if nested_sitemaps:
                logger.info(f"Found {len(nested_sitemaps)} nested sitemaps. Fetching...")
                nested_tasks = [self.fetch_sitemap_xml(session, url) for url in nested_sitemaps[:50]]  # Limit to 50 nested
                nested_contents = await asyncio.gather(*nested_tasks, return_exceptions=True)
                
                for i, content in enumerate(nested_contents):
                    if isinstance(content, str) and content.strip():
                        parsed_urls = self.parse_sitemap_xml(content, nested_sitemaps[i])
                        for url_data in parsed_urls:
                            if url_data.get('type') != 'nested_sitemap':
                                all_urls.append(url_data)
        
        logger.info(f"Total URLs collected from all sitemaps: {len(all_urls)}")
        return all_urls

    async def fetch_page_content(self, session: aiohttp.ClientSession, url_data: Dict[str, str]) -> Dict[str, Any]:
        """Fetch and parse content from a URL"""
        url = url_data['url']
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            async with session.get(url, headers=headers, timeout=15) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract title
                    title = ""
                    title_elem = soup.find('title') or soup.find('h1')
                    if title_elem:
                        title = title_elem.get_text().strip()
                    
                    # Use news title from sitemap if available
                    if url_data.get('news_title'):
                        title = url_data['news_title']
                    
                    # Extract meta description
                    description = ""
                    meta_desc = soup.find('meta', attrs={'name': 'description'}) or \
                               soup.find('meta', attrs={'property': 'og:description'})
                    if meta_desc:
                        description = meta_desc.get('content', '').strip()
                    
                    # Extract main content
                    content = ""
                    content_selectors = [
                        'article', '.article-content', '.content', '.post-content',
                        '.news-content', '.story-content', 'main', '.main-content',
                        '.entry-content', '.post-body', '.article-body'
                    ]
                    
                    for selector in content_selectors:
                        content_elem = soup.select_one(selector)
                        if content_elem:
                            for script in content_elem(["script", "style", "nav", "footer"]):
                                script.decompose()
                            content = content_elem.get_text().strip()
                            break
                    
                    if not content:
                        paragraphs = soup.find_all('p')
                        content = ' '.join([p.get_text().strip() for p in paragraphs[:5]])
                    
                    # Clean up content
                    content = re.sub(r'\s+', ' ', content)
                    content = content[:2000] + "..." if len(content) > 2000 else content
                    
                    # Extract category/topic from URL
                    category = self.extract_category_from_url(url)
                    
                    return {
                        'url': url,
                        'title': title[:300] if title else "No Title",
                        'description': description[:500] if description else "",
                        'content': content,
                        'category': category,
                        'scraped_at': datetime.now().isoformat(),
                        'lastmod': url_data.get('lastmod', ''),
                        'priority': url_data.get('priority', ''),
                        'changefreq': url_data.get('changefreq', ''),
                        'publication_date': url_data.get('publication_date', ''),
                        'source_sitemap': url_data.get('source_sitemap', ''),
                        'status': 'success'
                    }
                else:
                    logger.warning(f"Failed to fetch {url}: Status {response.status}")
                    return self.create_error_entry(url_data, f"HTTP {response.status}")
                    
        except asyncio.TimeoutError:
            logger.warning(f"Timeout fetching {url}")
            return self.create_error_entry(url_data, "Timeout")
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return self.create_error_entry(url_data, str(e))

    def create_error_entry(self, url_data: Dict[str, str], error: str) -> Dict[str, Any]:
        """Create error entry for failed URLs"""
        return {
            'url': url_data['url'],
            'title': f"Error: {error}",
            'description': "",
            'content': "",
            'category': self.extract_category_from_url(url_data['url']),
            'scraped_at': datetime.now().isoformat(),
            'lastmod': url_data.get('lastmod', ''),
            'priority': url_data.get('priority', ''),
            'changefreq': url_data.get('changefreq', ''),
            'publication_date': url_data.get('publication_date', ''),
            'source_sitemap': url_data.get('source_sitemap', ''),
            'status': 'error',
            'error': error
        }

    def extract_category_from_url(self, url: str) -> str:
        """Extract category/topic from URL"""
        try:
            path = urlparse(url).path.lower()
            domain = urlparse(url).netloc.lower()
            
            # Domain-based categories
            if 'shiksha.com' in domain:
                return 'Education India'
            elif 'leverageedu.com' in domain:
                return 'Study Abroad'
            elif 'studyabroad.com' in domain:
                return 'Study Abroad'
            elif 'opendoorsdata.org' in domain:
                return 'International Education Data'
            elif 'educations.com' in domain:
                return 'Global Education'
            elif 'thepienews.com' in domain:
                return 'Education News'
            elif 'timeshighereducation.com' in domain:
                return 'Higher Education'
            elif 'educationusa.state.gov' in domain:
                return 'US Education'
            elif 'usnews.com' in domain:
                return 'US News'
            
            # Path-based categories
            categories = {
                'college': 'Higher Education',
                'university': 'Higher Education',
                'medical': 'Medical Education',
                'medicine': 'Medical Education',
                'health': 'Health Sciences',
                'engineering': 'Engineering',
                'mba': 'Business Education',
                'news': 'Education News',
                'admission': 'Admissions',
                'exam': 'Examinations',
                'neet': 'Medical Entrance',
                'jee': 'Engineering Entrance',
                'abroad': 'Study Abroad',
                'visa': 'Student Visa',
                'scholarship': 'Scholarships',
                'ranking': 'University Rankings'
            }
            
            for keyword, category in categories.items():
                if keyword in path:
                    return category
            
            return 'General Education'
        except:
            return 'General Education'

    async def scrape_all_urls(self, url_data_list: List[Dict[str, str]], max_articles: int = 500) -> List[Dict[str, Any]]:
        """Scrape content from URLs concurrently"""
        # Limit the number of articles to scrape to avoid overwhelming
        limited_urls = url_data_list[:max_articles]
        logger.info(f"Scraping {len(limited_urls)} articles (limited from {len(url_data_list)} total)")
        
        connector = aiohttp.TCPConnector(limit=15, limit_per_host=8)
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Process in batches to avoid overwhelming servers
            batch_size = 20
            all_results = []
            
            for i in range(0, len(limited_urls), batch_size):
                batch = limited_urls[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(limited_urls) + batch_size - 1)//batch_size}")
                
                tasks = [self.fetch_page_content(session, url_data) for url_data in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                valid_results = []
                for result in batch_results:
                    if isinstance(result, dict):
                        valid_results.append(result)
                    else:
                        logger.error(f"Task failed with exception: {result}")
                
                all_results.extend(valid_results)
                
                # Broadcast batch immediately
                if valid_results:
                    await self.broadcast_batch(valid_results)
                
                # Small delay between batches
                await asyncio.sleep(2)
            
            return all_results

    def save_to_json(self, news_data: List[Dict[str, Any]], filename: str = None):
        """Save news data to JSON file in a format compatible with news_analyzer"""
        if filename is None:
            filename = "rss_output.json"
        
        try:
            # Transform the data into the format expected by news_analyzer
            formatted_articles = []
            for article in news_data:
                if article.get('status') == 'success':
                    formatted_article = {
                        'title': article.get('title', ''),
                        'source': article.get('url', ''),
                        'timestamp': article.get('scraped_at', ''),
                        'category': article.get('category', 'General'),
                        'content': article.get('content', ''),
                        'description': article.get('description', '')
                    }
                    formatted_articles.append(formatted_article)
            
            output_data = {
                'articles': formatted_articles,
                'metadata': {
                    'total_articles': len(formatted_articles),
                    'scrape_timestamp': datetime.now().isoformat(),
                    'categories': list(set(a.get('category', 'General') for a in formatted_articles))
                }
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"News data saved to {filename} with {len(formatted_articles)} articles")
            return filename
        except Exception as e:
            logger.error(f"Error saving to JSON: {e}")
            return None

    def start(self):
        """Start the Socket.IO server"""
        logger.info(f"Starting Socket.IO server on port {self.port}")
        self.socketio.run(self.app, host='0.0.0.0', port=self.port, debug=True, allow_unsafe_werkzeug=True)

async def process_sitemaps(scraper: NewsScraperBroadcaster, sitemap_urls: List[str]):
    """Process sitemaps asynchronously"""
    # Fetch all sitemap URLs and extract article URLs
    all_url_data = await scraper.fetch_all_sitemaps(sitemap_urls)
    
    if not all_url_data:
        logger.error("No URLs found in any sitemaps")
        return
    
    logger.info(f"Found {len(all_url_data)} total article URLs")
    
    # Scrape articles and broadcast in real-time
    news_data = await scraper.scrape_all_urls(all_url_data, max_articles=500)
    scraper.news_data = news_data
    
    # Save to JSON
    json_file = scraper.save_to_json(news_data, "rss_output.json")
    
    # Send completion message
    scraper.socketio.emit('scraping_complete', {
        'type': 'scraping_complete',
        'total_articles': len(news_data),
        'successful': len([a for a in news_data if a.get('status') == 'success']),
        'failed': len([a for a in news_data if a.get('status') == 'error']),
        'json_file': json_file,
        'timestamp': datetime.now().isoformat()
    })
    
    logger.info(f"Processing complete! Scraped {len(news_data)} articles, saved to {json_file}")
    return json_file

# Usage
def main():
    # Your sitemap URLs - keep the existing URLs list
    sitemap_urls = [
        "https://www.shiksha.com/updates.xml",
        "https://www.shiksha.com/NewsIndex1.xml",
        "https://www.shiksha.com/sitemap_index.xml",
        "https://opendoorsdata.org/sitemap_index.xml",
        "https://opendoorsdata.org/fact_sheets-sitemap.xml",
        "https://opendoorsdata.org/data-sitemap.xml",
        "https://opendoorsdata.org/fast_facts-sitemap.xml",
        "https://www.educations.com/sitemap-index.xml",
        "https://thepienews.com/sitemap_index.xml",
        "https://www.timeshighereducation.com/sitemap.xml",
        "https://www.timeshighereducation.com/sitemap-types-node-ranking_dataset.xml",
        "https://www.timeshighereducation.com/unijobs/sitemapindex.xml",
        "https://www.timeshighereducation.com/campus/sitemap.xml",
        "https://www.timeshighereducation.com/student/sitemap.xml",
        "https://www.timeshighereducation.com/googlenews.xml",
        "https://www.timeshighereducation.com/sitemaps/articles-sitemap/sitemap.xml",
        "https://educationusa.state.gov/sitemap.xml",
        "https://educationusa.state.gov/sitemap.xml?page=2",
        "https://educationusa.state.gov/sitemap.xml?page=1",
        "https://www.usnews.com/news/editorial-sitemap.xml",
        "https://www.usnews.com/news-sitemap.xml",
        "https://deals.usnews.com/sitemap.xml",
        "https://leverageedu.com/sitemap.xml",
        "https://leverageedu.com/blog/sitemap_index.xml",
        "https://leverageedu.com/blog/hi/sitemap_index.xml",
        "https://leverageedu.com/learn/sitemap_index.xml",
        "https://leverageedu.com/explore/sitemap_index.xml",
        "https://leverageedu.com/discover/sitemap_index.xml",
        "https://leverageedu.com/blog/hi/post-sitemap.xml",
        "https://leverageedu.com/blog/category-sitemap.xml",
        "https://leverageedu.com/blog/author-sitemap.xml",
        "https://leverageedu.com/blog/page-sitemap.xml",
        "https://leverageedu.com/blog/post-sitemap10.xml",
        "https://leverageedu.com/blog/post-sitemap5.xml",
        "https://leverageedu.com/blog/post-sitemap4.xml",
        "https://leverageedu.com/blog/post-sitemap3.xml",
        "https://leverageedu.com/blog/post-sitemap2.xml",
        "https://leverageedu.com/blog/post-sitemap1.xml",
        "https://leverageedu.com/blog/post-sitemap7.xml",
        "https://leverageedu.com/blog/post-sitemap6.xml",
        "https://www.studyabroad.com/sitemap_index.xml"
    ]
    
    # Create scraper instance
    scraper = NewsScraperBroadcaster(port=8765)
    
    # Create background task for processing sitemaps
    @scraper.app.before_first_request
    def start_background_task():
        asyncio.run(process_sitemaps(scraper, sitemap_urls))
    
    # Start the Socket.IO server
    scraper.start()

if __name__ == "__main__":
    main()
