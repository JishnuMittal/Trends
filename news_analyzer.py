import json
import time
import os
import requests
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from datetime import datetime
import logging
from flask import Flask, jsonify, render_template
from flask_socketio import SocketIO
import threading
import uuid
from multiprocessing import Pool, cpu_count
from functools import partial
from urllib.parse import urlparse

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("news_analyzer.log"),
                              logging.StreamHandler()])
logger = logging.getLogger(__name__)

class NewsAnalyzer:
    def __init__(self, model="llama3", watch_file="rss_output.json", watch_interval=10):
        """
        Initialize the News Analyzer application
        
        Args:
            model (str): The model to use ('llama3' or 'tinyllama')
            watch_file (str): The JSON file to watch for changes
            watch_interval (int): How often to check for file changes (seconds)
        """
        self.model = model
        self.watch_file = watch_file
        self.watch_interval = watch_interval
        self.last_modified = 0
        self.recent_keywords = []
        self.recent_titles = []  # Changed from recent_sentences to recent_titles
        self.max_history = 50  # Keep track of last 50 titles for trend analysis
        
        # Store analysis results for API access
        self.all_results = []
        self.trend_data = {"trending_keywords": [], "anomalies": [], "timestamp": ""}
        
        # Unique run ID for this instance
        self.run_id = str(uuid.uuid4())[:8]
        
        # Model selection parameters - Updated to use llama3 as default
        if model == "llama3":
            self.model_name = "llama3:latest"
        else:
            self.model_name = "tinyllama:latest"
            
        logger.info(f"Initialized NewsAnalyzer with model: {self.model_name}")
        
        # Verify Ollama is running
        self._check_ollama()
    
    def _check_ollama(self):
        """Check if Ollama is running and the model is available"""
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                
                if self.model_name not in model_names:
                    logger.warning(f"Model {self.model_name} not found in Ollama. Available models: {model_names}")
                    logger.info(f"Please run: ollama pull {self.model_name.split(':')[0]}")
                else:
                    logger.info(f"Connected to Ollama. Using model: {self.model_name}")
            else:
                logger.error("Couldn't connect to Ollama API")
        except requests.exceptions.ConnectionError:
            logger.error("Failed to connect to Ollama. Is it running? Start with 'ollama serve'")

    def analyze_text(self, title, source=""):
        """
        Send text to Ollama model and extract keywords and anomalies
        
        Args:
            title (str): The news title to analyze
            source (str): The source of the news title
            
        Returns:
            dict: Analysis results including keywords and anomalies
        """
        # IMPROVED PROMPT: More structured prompt for keyword extraction 
        keyword_prompt = f"""
        Task: Extract 3-5 of the most significant, specific, and informative keywords from the news title below.
        
        Guidelines:
        - Focus on entities, events, topics, and concepts that are central to the news item
        - Prefer specific terms over generic ones (e.g., "Tesla" over "company")
        - Include proper nouns, technical terms, and domain-specific vocabulary
        - Avoid common words, articles, and generic terms
        - Select keywords that would be useful for categorization and trend analysis
        
        News title: "{title}"

        Respond ONLY in this JSON format:
        {{
          "keywords": ["keyword1", "keyword2", "keyword3"]
        }}
        """
        
        # IMPROVED PROMPT: Much more detailed prompt for anomaly detection
        anomaly_prompt = f"""
        Task: Analyze the news title for anomalies - unusual, unexpected, or significant information that deviates from normal patterns or expectations.
        
        News title: "{title}"
        Source: "{source}"
        
        Context:
        - Recent keywords: {', '.join(self.recent_keywords[-30:])}
        - Recent news trends: {' | '.join(self.recent_titles[-5:])}
        
        Analysis guidelines:
        1. Identify information that contradicts established patterns or expectations
        2. Look for unexpected connections between entities or events
        3. Detect unusual shifts in sentiment, policy, or behavior
        4. Find statistical outliers or surprising numerical data
        5. Spot emerging trends that could signal important shifts
        6. Consider geopolitical, economic, technological, or social pattern breaks
        
        For each anomaly:
        - Be specific and descriptive
        - Explain why it's anomalous (contradicts expectations, breaks patterns, etc.)
        - Consider its potential significance for future trends
        
        Respond ONLY in this strict JSON format:
        {{
          "anomalies": [
            "Specific detailed anomaly 1", 
            "Specific detailed anomaly 2"
          ],
          "confidence": X.X  // Value between 0.0-1.0 representing confidence in anomaly detection
                        // 0.8-1.0: High confidence - clear pattern break
                        // 0.5-0.7: Medium confidence - possible anomaly
                        // 0.1-0.4: Low confidence - subtle deviation
                        // 0.0: No anomalies detected
        }}
        """
        
        try:
            # Extract keywords with improved prompt
            keyword_response = self._query_ollama(keyword_prompt)
            keywords = self._extract_json_from_response(keyword_response)
            
            # Extract anomalies with improved prompt
            anomaly_response = self._query_ollama(anomaly_prompt)
            anomalies = self._extract_json_from_response(anomaly_response)
            
            # Default confidence
            confidence = 0.0
            
            # Calculate confidence if not provided explicitly
            if anomalies:
                if 'confidence' in anomalies:
                    confidence = anomalies.get('confidence', 0.0)
                elif 'anomalies' in anomalies and len(anomalies['anomalies']) > 0:
                    # If anomalies exist but no confidence, assign default based on count
                    confidence = min(0.7, 0.3 + (len(anomalies['anomalies']) * 0.2))
            
            # Update recent keywords for context
            if keywords and 'keywords' in keywords:
                self.recent_keywords.extend(keywords['keywords'])
                self.recent_keywords = self.recent_keywords[-100:]  # Keep last 100 keywords
            
            # Update recent titles
            self.recent_titles.append(title)
            self.recent_titles = self.recent_titles[-self.max_history:]
            
            # Combine results
            result = {
                'title': title,  # Changed from 'sentence' to 'title'
                'source': source,  # Added source field
                'timestamp': datetime.now().isoformat(),
                'keywords': keywords.get('keywords', []) if keywords else [],
                'anomalies': anomalies.get('anomalies', []) if anomalies else [],
                'confidence': confidence
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            return {
                'title': title,  # Changed from 'sentence' to 'title'
                'source': source,  # Added source field
                'timestamp': datetime.now().isoformat(),
                'keywords': [],
                'anomalies': [],
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _query_ollama(self, prompt):
        """Send a prompt to Ollama and get the response"""
        try:
            # Added temperature and max_tokens parameters for more consistent outputs
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.1,  # Lower temperature for more deterministic responses
                    "num_predict": 512   # Limit token generation
                }
            )
            
            if response.status_code == 200:
                return response.json().get('response', '')
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return ""
                
        except Exception as e:
            logger.error(f"Error querying Ollama: {e}")
            return ""
    
    def _extract_json_from_response(self, text):
        """Extract JSON from Ollama's text response with improved robustness"""
        try:
            # First try to find JSON-like content in the response
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = text[start_idx:end_idx+1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # If direct extraction fails, try cleaning the string
                    cleaned_json = self._clean_json_string(json_str)
                    return json.loads(cleaned_json)
            else:
                # Try to extract a list if no JSON object found
                start_idx = text.find('[')
                end_idx = text.rfind(']')
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = text[start_idx:end_idx+1]
                    return {"keywords": json.loads(json_str)} 
                else:
                    # If we can't find JSON, create keywords from text
                    words = [w.strip().lower() for w in text.split() if len(w.strip()) > 3]
                    # Remove duplicates
                    words = list(dict.fromkeys(words))
                    return {"keywords": words[:5]}
                    
        except json.JSONDecodeError:
            logger.warning(f"Could not parse JSON from response: {text[:100]}...")
            # Fall back to simple word extraction
            words = [w.strip().lower() for w in text.split() if len(w.strip()) > 3]
            words = list(dict.fromkeys(words))
            return {"keywords": words[:5]}

    def _clean_json_string(self, json_str):
        """Clean and fix common JSON formatting issues"""
        # Replace single quotes with double quotes (common LLM mistake)
        json_str = json_str.replace("'", '"')
        
        # Fix trailing commas in arrays/objects (another common issue)
        json_str = json_str.replace(",]", "]").replace(",}", "}")
        
        # Fix missing quotes around keys
        import re
        json_str = re.sub(r'([{,])\s*([a-zA-Z0-9_]+):', r'\1"\2":', json_str)
        
        return json_str
    
    def analyze_file(self):
        """Analyze the news data file if it has been updated"""
        try:
            if not os.path.exists(self.watch_file):
                logger.warning(f"File {self.watch_file} not found")
                return None
                
            # Check if file has been modified
            current_modified = os.path.getmtime(self.watch_file)
            if current_modified <= self.last_modified:
                return None
                
            logger.info(f"Analyzing updated file: {self.watch_file}")
            self.last_modified = current_modified
            
            # Read and parse the JSON file
            try:
                with open(self.watch_file, 'r') as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON file: {e}")
                return None
            
            # Handle both new and old JSON formats
            articles = []
            if isinstance(data, dict):
                if 'articles' in data:
                    articles = data['articles']
                else:
                    # Try to convert URL data to article format
                    for item in data.values():
                        if isinstance(item, dict) and 'url' in item:
                            article = {
                                'title': item.get('title', item.get('url', '')),
                                'source': item.get('url', ''),
                                'timestamp': item.get('lastmod', datetime.now().isoformat()),
                                'category': self.extract_category_from_url(item.get('url', '')),
                                'content': '',
                                'description': ''
                            }
                            articles.append(article)
            elif isinstance(data, list):
                # Handle list format
                for item in data:
                    if isinstance(item, dict):
                        if 'url' in item:
                            # Convert URL data to article format
                            article = {
                                'title': item.get('title', item.get('url', '')),
                                'source': item.get('url', ''),
                                'timestamp': item.get('lastmod', datetime.now().isoformat()),
                                'category': self.extract_category_from_url(item.get('url', '')),
                                'content': '',
                                'description': ''
                            }
                            articles.append(article)
                        elif 'title' in item:
                            # Already in article format
                            articles.append(item)
            
            if not articles:
                logger.warning("No articles found in the JSON file")
                return None
            
            logger.info(f"Processing {len(articles)} articles using multiprocessing")
            
            # Use multiprocessing to analyze articles in parallel
            num_processes = min(cpu_count(), len(articles))
            
            with Pool(processes=num_processes) as pool:
                # Create a partial function with the model name
                analyze_func = partial(self._parallel_analyze, model_name=self.model_name)
                # Prepare data for parallel processing
                process_data = [(article.get('title', ''), article.get('source', '')) for article in articles]
                results = pool.starmap(analyze_func, process_data)
            
            # Filter out None results and combine with original article data
            valid_results = []
            for result, article in zip(results, articles):
                if result is not None:
                    # Combine analysis results with original article data
                    combined_result = {
                        'title': article.get('title', ''),
                        'source': article.get('source', ''),
                        'timestamp': article.get('timestamp', datetime.now().isoformat()),
                        'category': article.get('category', 'General'),
                        'content': article.get('content', ''),
                        'description': article.get('description', ''),
                        'keywords': result.get('keywords', []),
                        'anomalies': result.get('anomalies', []),
                        'confidence': result.get('confidence', 0.0),
                        'analysis_timestamp': result.get('timestamp', '')
                    }
                    valid_results.append(combined_result)
            
            # Save analysis results
            output_file = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump(valid_results, f, indent=2)
            
            # Update the in-memory results for API access
            self.all_results.extend(valid_results)
            # Keep only the most recent 100 results to avoid memory issues
            self.all_results = self.all_results[-100:]
                
            logger.info(f"Analysis complete. Results saved to {output_file}")
            
            # Generate trending report
            trend_report = self.generate_trending_report(valid_results)
            
            return valid_results
            
        except Exception as e:
            logger.error(f"Error analyzing file: {e}")
            return None

    def extract_category_from_url(self, url: str) -> str:
        """Extract category from URL"""
        try:
            path = urlparse(url).path.lower()
            
            # Extract category from path segments
            segments = [s for s in path.split('/') if s]
            
            # Common categories in educational news
            categories = {
                'engineering': 'Engineering',
                'medical': 'Medical',
                'medicine': 'Medical',
                'mba': 'Business',
                'law': 'Law',
                'science': 'Science',
                'arts': 'Arts',
                'commerce': 'Commerce',
                'education': 'Education',
                'teaching': 'Education',
                'design': 'Design',
                'technology': 'Technology',
                'management': 'Management',
                'nursing': 'Nursing',
                'pharmacy': 'Pharmacy',
                'architecture': 'Architecture'
            }
            
            # Check each segment for a category match
            for segment in segments:
                for key, value in categories.items():
                    if key in segment:
                        return value
            
            # Check if it's an exam-related URL
            exam_indicators = ['exam', 'test', 'entrance', 'admission']
            if any(indicator in url.lower() for indicator in exam_indicators):
                return 'Examinations'
            
            return 'General'
            
        except:
            return 'General'

    @staticmethod
    def _parallel_analyze(title, source, model_name):
        """Static method for parallel processing of articles"""
        try:
            # IMPROVED PROMPT: More structured prompt for keyword extraction 
            keyword_prompt = f"""
            Task: Extract 3-5 of the most significant, specific, and informative keywords from the news title below.
            
            Guidelines:
            - Focus on entities, events, topics, and concepts that are central to the news item
            - Prefer specific terms over generic ones (e.g., "Tesla" over "company")
            - Include proper nouns, technical terms, and domain-specific vocabulary
            - Avoid common words, articles, and generic terms
            - Select keywords that would be useful for categorization and trend analysis
            
            News title: "{title}"

            Respond ONLY in this JSON format:
            {{
              "keywords": ["keyword1", "keyword2", "keyword3"]
            }}
            """
            
            # IMPROVED PROMPT: Much more detailed prompt for anomaly detection
            anomaly_prompt = f"""
            Task: Analyze the news title for anomalies - unusual, unexpected, or significant information.
            
            News title: "{title}"
            Source: "{source}"
            
            Analysis guidelines:
            1. Identify information that contradicts established patterns
            2. Look for unexpected connections between entities or events
            3. Detect unusual shifts in sentiment, policy, or behavior
            4. Find statistical outliers or surprising numerical data
            
            Respond ONLY in this strict JSON format:
            {{
              "anomalies": [
                "Specific detailed anomaly 1", 
                "Specific detailed anomaly 2"
              ],
              "confidence": X.X
            }}
            """
            
            # Extract keywords
            keyword_response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model_name,
                    "prompt": keyword_prompt,
                    "stream": False,
                    "temperature": 0.1,
                    "num_predict": 512
                }
            ).json().get('response', '')
            
            # Extract anomalies
            anomaly_response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model_name,
                    "prompt": anomaly_prompt,
                    "stream": False,
                    "temperature": 0.1,
                    "num_predict": 512
                }
            ).json().get('response', '')
            
            # Process responses
            try:
                keywords = json.loads(keyword_response) if keyword_response else {"keywords": []}
                anomalies = json.loads(anomaly_response) if anomaly_response else {"anomalies": [], "confidence": 0.0}
            except json.JSONDecodeError:
                keywords = {"keywords": []}
                anomalies = {"anomalies": [], "confidence": 0.0}
            
            return {
                'title': title,
                'source': source,
                'timestamp': datetime.now().isoformat(),
                'keywords': keywords.get('keywords', []),
                'anomalies': anomalies.get('anomalies', []),
                'confidence': anomalies.get('confidence', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Error in parallel analysis: {e}")
            return None
    
    def generate_trending_report(self, results):
        """Generate a report of trending keywords and anomalies with improved analytics"""
        try:
            all_keywords = []
            all_anomalies = []
            anomaly_confidence = {}
            
            # Improved tracking - also track source title for each anomaly
            anomaly_sources = {}
            
            for result in results:
                # Extract and normalize keywords
                keywords = [k.lower().strip() for k in result.get('keywords', [])]
                all_keywords.extend(keywords)
                
                # Process anomalies and their confidence
                anomalies = result.get('anomalies', [])
                confidence = result.get('confidence', 0.0)
                title = result.get('title', '')  # Changed from 'sentence' to 'title'
                
                for anomaly in anomalies:
                    all_anomalies.append(anomaly)
                    # Store highest confidence for each unique anomaly
                    if anomaly in anomaly_confidence:
                        anomaly_confidence[anomaly] = max(anomaly_confidence[anomaly], confidence)
                    else:
                        anomaly_confidence[anomaly] = confidence
                        
                    # Store source title
                    if anomaly in anomaly_sources:
                        anomaly_sources[anomaly].append(title)
                    else:
                        anomaly_sources[anomaly] = [title]
            
            # Count keyword occurrences with improved normalization
            keyword_counts = {}
            for keyword in all_keywords:
                # Normalize keywords
                keyword = keyword.lower().strip()
                if keyword in keyword_counts:
                    keyword_counts[keyword] += 1
                else:
                    keyword_counts[keyword] = 1
            
            # Sort by count
            sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Format anomalies with confidence and sources
            formatted_anomalies = [
                {
                    "anomaly": anomaly, 
                    "confidence": confidence,
                    "sources": anomaly_sources.get(anomaly, [])[:2]  # Include up to 2 source titles
                } 
                for anomaly, confidence in anomaly_confidence.items()
            ]
            
            # Sort anomalies by confidence
            sorted_anomalies = sorted(formatted_anomalies, key=lambda x: x['confidence'], reverse=True)
            
            # Create trend report
            report = {
                "timestamp": datetime.now().isoformat(),
                "trending_keywords": [{"keyword": k, "count": c} for k, c in sorted_keywords[:10]],
                "anomalies": sorted_anomalies,
                "total_items_analyzed": len(results)  # Changed from 'total_sentences_analyzed'
            }
            
            # Save trend report
            trend_file = "trend_report.json"
            with open(trend_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Update stored trend data for API
            self.trend_data = report
                
            logger.info(f"Trend report generated: {trend_file}")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating trend report: {e}")
            return None
    
    def watch_for_updates(self):
        """Watch for file updates and process them"""
        logger.info(f"Watching for updates to {self.watch_file} every {self.watch_interval} seconds")
        
        while True:
            self.analyze_file()
            time.sleep(self.watch_interval)

class FileChangeHandler(FileSystemEventHandler):
    """Handler for file system events"""
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
        
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(self.analyzer.watch_file):
            logger.info(f"File {event.src_path} has been modified")
            self.analyzer.analyze_file()

class WebServer:
    def __init__(self, analyzer, port=8000):
        self.analyzer = analyzer
        self.port = port
        self.app = Flask(__name__, template_folder='templates')
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        self.setup_routes()
        self.setup_socket_events()
        
    def setup_routes(self):
        @self.app.route('/')
        def home():
            return render_template('index.html')
        
        @self.app.route('/api/info')
        def get_info():
            return jsonify({
                'model': self.analyzer.model_name,
                'file': self.analyzer.watch_file,
                'interval': self.analyzer.watch_interval,
                'run_id': self.analyzer.run_id
            })
        
        @self.app.route('/api/recent')
        def get_recent():
            return jsonify(self.analyzer.all_results)
        
        @self.app.route('/api/trends')
        def get_trends():
            return jsonify(self.analyzer.trend_data)
        
        @self.app.route('/api/report')
        def get_report():
            try:
                with open('trend_report.json', 'r') as f:
                    return jsonify(json.load(f))
            except:
                return jsonify({"error": "Report not found"}), 404

    def setup_socket_events(self):
        @self.socketio.on('connect')
        def handle_connect():
            logger.info("Client connected to analyzer")
    
    def start(self):
        """Start the web server with Socket.IO support"""
        self.socketio.run(self.app, host='0.0.0.0', port=self.port, debug=True, allow_unsafe_werkzeug=True)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='News Trend Analyzer')
    parser.add_argument('--model', type=str, default='llama3', choices=['llama3', 'tinyllama'],
                        help='LLM model to use (llama3 or tinyllama)')
    parser.add_argument('--file', type=str, default='rss_output.json',
                        help='JSON file to watch for news data')
    parser.add_argument('--interval', type=int, default=10,
                        help='Interval in seconds to check for file updates')
    parser.add_argument('--watch', action='store_true',
                        help='Use watchdog to monitor file changes instead of polling')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port for the web interface')
    
    args = parser.parse_args()
    
    analyzer = NewsAnalyzer(
        model=args.model,
        watch_file=args.file,
        watch_interval=args.interval
    )
    
    # Start the web server with Socket.IO
    web_server = WebServer(analyzer, port=args.port)
    
    # Start file watching in a background thread
    if args.watch:
        event_handler = FileChangeHandler(analyzer)
        observer = Observer()
        path = os.path.dirname(os.path.abspath(args.file)) or '.'
        observer.schedule(event_handler, path=path, recursive=False)
        observer.start()
        logger.info(f"Started file watcher for {args.file}")
    else:
        threading.Thread(target=analyzer.watch_for_updates, daemon=True).start()
        logger.info(f"Started polling for {args.file} every {args.interval} seconds")
    
    # Start the web server (this will block)
    web_server.start()

if __name__ == "__main__":
    main()