# # import json
# # import time
# # import os
# # import requests
# # import numpy as np
# # from watchdog.observers import Observer
# # from watchdog.events import FileSystemEventHandler
# # from datetime import datetime
# # import logging

# # # Set up logging
# # logging.basicConfig(level=logging.INFO, 
# #                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
# #                     handlers=[logging.FileHandler("news_analyzer.log"),
# #                               logging.StreamHandler()])
# # logger = logging.getLogger(__name__)

# # class NewsAnalyzer:
# #     def __init__(self, model="phi", watch_file="./Trends_detector-main/rss_output.json", watch_interval=10):
# #         """
# #         Initialize the News Analyzer application
        
# #         Args:
# #             model (str): The model to use ('phi' or 'tinyllama')
# #             watch_file (str): The JSON file to watch for changes
# #             watch_interval (int): How often to check for file changes (seconds)
# #         """
# #         self.model = model
# #         self.watch_file = watch_file
# #         self.watch_interval = watch_interval
# #         self.last_modified = 0
# #         self.recent_keywords = []
# #         self.recent_sentences = []
# #         self.max_history = 50  # Keep track of last 50 sentences for trend analysis
        
# #         # Model selection parameters
# #         if model == "phi":
# #             self.model_name = "phi"
# #         else:
# #             self.model_name = "tinyllama"
            
# #         logger.info(f"Initialized NewsAnalyzer with model: {self.model_name}")
        
# #         # Verify Ollama is running
# #         self._check_ollama()
    
# #     def _check_ollama(self):
# #         """Check if Ollama is running and the model is available"""
# #         try:
# #             response = requests.get("http://localhost:11434/api/tags")
# #             if response.status_code == 200:
# #                 models = response.json().get("models", [])
# #                 model_names = [m["name"] for m in models]
                
# #                 if self.model_name not in model_names:
# #                     logger.warning(f"Model {self.model_name} not found in Ollama. Available models: {model_names}")
# #                     logger.info(f"Please run: ollama pull {self.model_name}")
# #                 else:
# #                     logger.info(f"Connected to Ollama. Using model: {self.model_name}")
# #             else:
# #                 logger.error("Couldn't connect to Ollama API")
# #         except requests.exceptions.ConnectionError:
# #             logger.error("Failed to connect to Ollama. Is it running? Start with 'ollama serve'")

# #     def analyze_text(self, sentence):
# #         """
# #         Send text to Ollama model and extract keywords and anomalies
        
# #         Args:
# #             sentence (str): The news sentence to analyze
            
# #         Returns:
# #             dict: Analysis results including keywords and anomalies
# #         """
# #         # Prepare prompt for keyword extraction
# #         keyword_prompt = f"""
# #         Extract the 3-5 most important keywords from this news sentence. 
# #         Format as JSON with a 'keywords' array. Sentence: "{sentence}"
# #         """
        
# #         # Prepare prompt for anomaly detection
# #         anomaly_prompt = f"""
# #         Analyze this news sentence for unusual or anomalous information. 
# #         Format as JSON with 'anomalies' array and 'confidence' float (0-1).
# #         Sentence: "{sentence}"
# #         Context from recent news: {', '.join(self.recent_keywords[:20])}
# #         """
        
# #         try:
# #             # Extract keywords
# #             keyword_response = self._query_ollama(keyword_prompt)
# #             keywords = self._extract_json_from_response(keyword_response)
            
# #             # Extract anomalies
# #             anomaly_response = self._query_ollama(anomaly_prompt)
# #             anomalies = self._extract_json_from_response(anomaly_response)
            
# #             # Update recent keywords for context
# #             if keywords and 'keywords' in keywords:
# #                 self.recent_keywords.extend(keywords['keywords'])
# #                 self.recent_keywords = self.recent_keywords[-100:]  # Keep last 100 keywords
            
# #             # Update recent sentences
# #             self.recent_sentences.append(sentence)
# #             self.recent_sentences = self.recent_sentences[-self.max_history:]
            
# #             # Combine results
# #             result = {
# #                 'sentence': sentence,
# #                 'timestamp': datetime.now().isoformat(),
# #                 'keywords': keywords.get('keywords', []) if keywords else [],
# #                 'anomalies': anomalies.get('anomalies', []) if anomalies else [],
# #                 'confidence': anomalies.get('confidence', 0.0) if anomalies else 0.0
# #             }
            
# #             return result
            
# #         except Exception as e:
# #             logger.error(f"Error analyzing text: {e}")
# #             return {
# #                 'sentence': sentence,
# #                 'timestamp': datetime.now().isoformat(),
# #                 'keywords': [],
# #                 'anomalies': [],
# #                 'confidence': 0.0,
# #                 'error': str(e)
# #             }
    
# #     def _query_ollama(self, prompt):
# #         """Send a prompt to Ollama and get the response"""
# #         try:
# #             response = requests.post(
# #                 "http://localhost:11434/api/generate",
# #                 json={
# #                     "model": self.model_name,
# #                     "prompt": prompt,
# #                     "stream": False
# #                 }
# #             )
            
# #             if response.status_code == 200:
# #                 return response.json().get('response', '')
# #             else:
# #                 logger.error(f"Ollama API error: {response.status_code} - {response.text}")
# #                 return ""
                
# #         except Exception as e:
# #             logger.error(f"Error querying Ollama: {e}")
# #             return ""
    
# #     def _extract_json_from_response(self, text):
# #         """Extract JSON from Ollama's text response"""
# #         try:
# #             # Find JSON-like content in the response
# #             start_idx = text.find('{')
# #             end_idx = text.rfind('}')
            
# #             if start_idx >= 0 and end_idx > start_idx:
# #                 json_str = text[start_idx:end_idx+1]
# #                 return json.loads(json_str)
# #             else:
# #                 # Try to extract a list if no JSON object found
# #                 start_idx = text.find('[')
# #                 end_idx = text.rfind(']')
# #                 if start_idx >= 0 and end_idx > start_idx:
# #                     json_str = text[start_idx:end_idx+1]
# #                     return {"keywords": json.loads(json_str)} 
# #                 else:
# #                     # If we can't find JSON, create keywords from text
# #                     words = [w.strip() for w in text.split() if len(w.strip()) > 3]
# #                     return {"keywords": words[:5]}
                    
# #         except json.JSONDecodeError:
# #             logger.warning(f"Could not parse JSON from response: {text[:100]}...")
# #             # Fall back to simple word extraction
# #             words = [w.strip() for w in text.split() if len(w.strip()) > 3]
# #             return {"keywords": words[:5]}
    
# #     def analyze_file(self):
# #         """Analyze the news data file if it has been updated"""
# #         try:
# #             if not os.path.exists(self.watch_file):
# #                 logger.warning(f"File {self.watch_file} not found")
# #                 return None
                
# #             # Check if file has been modified
# #             current_modified = os.path.getmtime(self.watch_file)
# #             if current_modified <= self.last_modified:
# #                 return None
                
# #             logger.info(f"Analyzing updated file: {self.watch_file}")
# #             self.last_modified = current_modified
            
# #             # Read and parse the JSON file
# #             with open(self.watch_file, 'r') as f:
# #                 data = json.load(f)
            
# #             results = []
            
# #             # Process each news sentence
# #             if isinstance(data, list):
# #                 for item in data:
# #                     if isinstance(item, dict) and 'sentence' in item:
# #                         result = self.analyze_text(item['sentence'])
# #                         results.append(result)
# #             elif isinstance(data, dict):
# #                 # Handle case where JSON is a single object or has a different structure
# #                 if 'sentences' in data and isinstance(data['sentences'], list):
# #                     # Format with a sentences array
# #                     for sentence in data['sentences']:
# #                         if isinstance(sentence, str):
# #                             result = self.analyze_text(sentence)
# #                             results.append(result)
# #                         elif isinstance(sentence, dict) and 'text' in sentence:
# #                             result = self.analyze_text(sentence['text'])
# #                             results.append(result)
# #                 elif 'sentence' in data:
# #                     # Single sentence entry
# #                     result = self.analyze_text(data['sentence'])
# #                     results.append(result)
            
# #             # Save analysis results
# #             output_file = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
# #             with open(output_file, 'w') as f:
# #                 json.dump(results, f, indent=2)
                
# #             logger.info(f"Analysis complete. Results saved to {output_file}")
            
# #             # Generate trending report
# #             self.generate_trending_report(results)
            
# #             return results
            
# #         except Exception as e:
# #             logger.error(f"Error analyzing file: {e}")
# #             return None
    
# #     def generate_trending_report(self, results):
# #         """Generate a report of trending keywords and anomalies"""
# #         try:
# #             all_keywords = []
# #             all_anomalies = []
            
# #             for result in results:
# #                 all_keywords.extend(result.get('keywords', []))
# #                 all_anomalies.extend(result.get('anomalies', []))
            
# #             # Count keyword occurrences
# #             keyword_counts = {}
# #             for keyword in all_keywords:
# #                 if keyword in keyword_counts:
# #                     keyword_counts[keyword] += 1
# #                 else:
# #                     keyword_counts[keyword] = 1
            
# #             # Sort by count
# #             sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
            
# #             # Create trend report
# #             report = {
# #                 "timestamp": datetime.now().isoformat(),
# #                 "trending_keywords": [{"keyword": k, "count": c} for k, c in sorted_keywords[:10]],
# #                 "anomalies": all_anomalies,
# #                 "total_sentences_analyzed": len(results)
# #             }
            
# #             # Save trend report
# #             trend_file = "trend_report.json"
# #             with open(trend_file, 'w') as f:
# #                 json.dump(report, f, indent=2)
                
# #             logger.info(f"Trend report generated: {trend_file}")
            
# #         except Exception as e:
# #             logger.error(f"Error generating trend report: {e}")
    
# #     def watch_for_updates(self):
# #         """Watch for file updates and process them"""
# #         logger.info(f"Watching for updates to {self.watch_file} every {self.watch_interval} seconds")
        
# #         while True:
# #             self.analyze_file()
# #             time.sleep(self.watch_interval)

# # class FileChangeHandler(FileSystemEventHandler):
# #     """Handler for file system events"""
    
# #     def __init__(self, analyzer):
# #         self.analyzer = analyzer
        
# #     def on_modified(self, event):
# #         if not event.is_directory and event.src_path.endswith(self.analyzer.watch_file):
# #             logger.info(f"File {event.src_path} has been modified")
# #             self.analyzer.analyze_file()

# # def main():
# #     import argparse
    
# #     parser = argparse.ArgumentParser(description='News Trend Analyzer')
# #     parser.add_argument('--model', type=str, default='phi', choices=['phi', 'tinyllama'],
# #                         help='LLM model to use (phi or tinyllama)')
# #     parser.add_argument('--file', type=str, default='./Trends_detector-main/rss_output.json',
# #                         help='JSON file to watch for news data')
# #     parser.add_argument('--interval', type=int, default=10,
# #                         help='Interval in seconds to check for file updates')
# #     parser.add_argument('--watch', action='store_true',
# #                         help='Use watchdog to monitor file changes instead of polling')
    
# #     args = parser.parse_args()
    
# #     analyzer = NewsAnalyzer(
# #         model=args.model,
# #         watch_file=args.file,
# #         watch_interval=args.interval
# #     )
    
# #     if args.watch:
# #         # Use watchdog for file monitoring
# #         event_handler = FileChangeHandler(analyzer)
# #         observer = Observer()
# #         observer.schedule(event_handler, path=os.path.dirname(os.path.abspath(args.file)) or '.', 
# #                          recursive=False)
# #         observer.start()
        
# #         try:
# #             logger.info("Watching for file changes... Press Ctrl+C to stop")
# #             while True:
# #                 time.sleep(1)
# #         except KeyboardInterrupt:
# #             observer.stop()
# #         observer.join()
# #     else:
# #         # Use polling approach
# #         analyzer.watch_for_updates()

# # if __name__ == "__main__":
# #     main()

# import json
# import time
# import os
# import requests
# import numpy as np
# from watchdog.observers import Observer
# from watchdog.events import FileSystemEventHandler
# from datetime import datetime
# import logging
# from flask import Flask, jsonify, render_template_string, send_file
# import threading
# import uuid

# # Set up logging
# logging.basicConfig(level=logging.INFO, 
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#                     handlers=[logging.FileHandler("news_analyzer.log"),
#                               logging.StreamHandler()])
# logger = logging.getLogger(__name__)

# class NewsAnalyzer:
#     def __init__(self, model="phi", watch_file="./Trends_detector-main/rss_output.json", watch_interval=10):
#         """
#         Initialize the News Analyzer application
        
#         Args:
#             model (str): The model to use ('phi' or 'tinyllama')
#             watch_file (str): The JSON file to watch for changes
#             watch_interval (int): How often to check for file changes (seconds)
#         """
#         self.model = model
#         self.watch_file = watch_file
#         self.watch_interval = watch_interval
#         self.last_modified = 0
#         self.recent_keywords = []
#         self.recent_sentences = []
#         self.max_history = 50  # Keep track of last 50 sentences for trend analysis
        
#         # Store analysis results for API access
#         self.all_results = []
#         self.trend_data = {"trending_keywords": [], "anomalies": [], "timestamp": ""}
        
#         # Unique run ID for this instance
#         self.run_id = str(uuid.uuid4())[:8]
        
#         # Model selection parameters
#         if model == "phi":
#             self.model_name = "phi:latest"  # Fixed: Now using proper model name with tag
#         else:
#             self.model_name = "tinyllama:latest"  # Fixed: Now using proper model name with tag
            
#         logger.info(f"Initialized NewsAnalyzer with model: {self.model_name}")
        
#         # Verify Ollama is running
#         self._check_ollama()
    
#     def _check_ollama(self):
#         """Check if Ollama is running and the model is available"""
#         try:
#             response = requests.get("http://localhost:11434/api/tags")
#             if response.status_code == 200:
#                 models = response.json().get("models", [])
#                 model_names = [m["name"] for m in models]
                
#                 if self.model_name not in model_names:
#                     logger.warning(f"Model {self.model_name} not found in Ollama. Available models: {model_names}")
#                     logger.info(f"Please run: ollama pull {self.model_name.split(':')[0]}")
#                 else:
#                     logger.info(f"Connected to Ollama. Using model: {self.model_name}")
#             else:
#                 logger.error("Couldn't connect to Ollama API")
#         except requests.exceptions.ConnectionError:
#             logger.error("Failed to connect to Ollama. Is it running? Start with 'ollama serve'")

#     def analyze_text(self, sentence):
#         """
#         Send text to Ollama model and extract keywords and anomalies
        
#         Args:
#             sentence (str): The news sentence to analyze
            
#         Returns:
#             dict: Analysis results including keywords and anomalies
#         """
#         # Prepare prompt for keyword extraction
#         keyword_prompt = f"""
#         Extract the 3-5 most important keywords from this news sentence. 
#         Format as JSON with a 'keywords' array. Sentence: "{sentence}"
#         """
        
#         # Improved prompt for anomaly detection with clearer instructions
#         anomaly_prompt = f"""
#         Analyze this news sentence and identify any unusual, unexpected, or anomalous information.
#         Look for events, developments, or relationships that diverge from normal patterns.
#         Such that i can predict any future trends from the anamolies. Give me atleast 2-3 anomalies.
        
#         Current sentence: "{sentence}"
        
#         Context (recent keywords): {', '.join(self.recent_keywords[:30])}
#         Context (recent news): {' '.join(self.recent_sentences[-5:])}
        
#         Respond ONLY in this JSON format:
#         {{
#           "anomalies": ["specific anomaly 1", "specific anomaly 2"],
#           "confidence": 0.X  // value between 0-1 indicating confidence level of anomalies
#         }}
        
#         If no anomalies exist, return an empty array for anomalies but assign an appropriate confidence value.
#         """
        
#         try:
#             # Extract keywords
#             keyword_response = self._query_ollama(keyword_prompt)
#             keywords = self._extract_json_from_response(keyword_response)
            
#             # Extract anomalies with improved prompt
#             anomaly_response = self._query_ollama(anomaly_prompt)
#             anomalies = self._extract_json_from_response(anomaly_response)
            
#             # Default confidence
#             confidence = 0.0
            
#             # Calculate confidence if not provided explicitly
#             if anomalies:
#                 if 'confidence' in anomalies:
#                     confidence = anomalies.get('confidence', 0.0)
#                 elif 'anomalies' in anomalies and len(anomalies['anomalies']) > 0:
#                     # If anomalies exist but no confidence, assign default based on count
#                     confidence = min(0.7, 0.3 + (len(anomalies['anomalies']) * 0.2))
            
#             # Update recent keywords for context
#             if keywords and 'keywords' in keywords:
#                 self.recent_keywords.extend(keywords['keywords'])
#                 self.recent_keywords = self.recent_keywords[-100:]  # Keep last 100 keywords
            
#             # Update recent sentences
#             self.recent_sentences.append(sentence)
#             self.recent_sentences = self.recent_sentences[-self.max_history:]
            
#             # Combine results
#             result = {
#                 'sentence': sentence,
#                 'timestamp': datetime.now().isoformat(),
#                 'keywords': keywords.get('keywords', []) if keywords else [],
#                 'anomalies': anomalies.get('anomalies', []) if anomalies else [],
#                 'confidence': confidence
#             }
            
#             return result
            
#         except Exception as e:
#             logger.error(f"Error analyzing text: {e}")
#             return {
#                 'sentence': sentence,
#                 'timestamp': datetime.now().isoformat(),
#                 'keywords': [],
#                 'anomalies': [],
#                 'confidence': 0.0,
#                 'error': str(e)
#             }
    
#     def _query_ollama(self, prompt):
#         """Send a prompt to Ollama and get the response"""
#         try:
#             response = requests.post(
#                 "http://localhost:11434/api/generate",
#                 json={
#                     "model": self.model_name,
#                     "prompt": prompt,
#                     "stream": False
#                 }
#             )
            
#             if response.status_code == 200:
#                 return response.json().get('response', '')
#             else:
#                 logger.error(f"Ollama API error: {response.status_code} - {response.text}")
#                 return ""
                
#         except Exception as e:
#             logger.error(f"Error querying Ollama: {e}")
#             return ""
    
#     def _extract_json_from_response(self, text):
#         """Extract JSON from Ollama's text response"""
#         try:
#             # Find JSON-like content in the response
#             start_idx = text.find('{')
#             end_idx = text.rfind('}')
            
#             if start_idx >= 0 and end_idx > start_idx:
#                 json_str = text[start_idx:end_idx+1]
#                 return json.loads(json_str)
#             else:
#                 # Try to extract a list if no JSON object found
#                 start_idx = text.find('[')
#                 end_idx = text.rfind(']')
#                 if start_idx >= 0 and end_idx > start_idx:
#                     json_str = text[start_idx:end_idx+1]
#                     return {"keywords": json.loads(json_str)} 
#                 else:
#                     # If we can't find JSON, create keywords from text
#                     words = [w.strip() for w in text.split() if len(w.strip()) > 3]
#                     return {"keywords": words[:5]}
                    
#         except json.JSONDecodeError:
#             logger.warning(f"Could not parse JSON from response: {text[:100]}...")
#             # Fall back to simple word extraction
#             words = [w.strip() for w in text.split() if len(w.strip()) > 3]
#             return {"keywords": words[:5]}
    
#     def analyze_file(self):
#         """Analyze the news data file if it has been updated"""
#         try:
#             if not os.path.exists(self.watch_file):
#                 logger.warning(f"File {self.watch_file} not found")
#                 return None
                
#             # Check if file has been modified
#             current_modified = os.path.getmtime(self.watch_file)
#             if current_modified <= self.last_modified:
#                 return None
                
#             logger.info(f"Analyzing updated file: {self.watch_file}")
#             self.last_modified = current_modified
            
#             # Read and parse the JSON file
#             with open(self.watch_file, 'r') as f:
#                 data = json.load(f)
            
#             results = []
            
#             # Process each news sentence
#             if isinstance(data, list):
#                 for item in data:
#                     if isinstance(item, dict) and 'sentence' in item:
#                         result = self.analyze_text(item['sentence'])
#                         results.append(result)
#             elif isinstance(data, dict):
#                 # Handle case where JSON is a single object or has a different structure
#                 if 'sentences' in data and isinstance(data['sentences'], list):
#                     # Format with a sentences array
#                     for sentence in data['sentences']:
#                         if isinstance(sentence, str):
#                             result = self.analyze_text(sentence)
#                             results.append(result)
#                         elif isinstance(sentence, dict) and 'text' in sentence:
#                             result = self.analyze_text(sentence['text'])
#                             results.append(result)
#                 elif 'sentence' in data:
#                     # Single sentence entry
#                     result = self.analyze_text(data['sentence'])
#                     results.append(result)
            
#             # Save analysis results
#             output_file = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
#             with open(output_file, 'w') as f:
#                 json.dump(results, f, indent=2)
            
#             # Update the in-memory results for API access
#             self.all_results.extend(results)
#             # Keep only the most recent 100 results to avoid memory issues
#             self.all_results = self.all_results[-100:]
                
#             logger.info(f"Analysis complete. Results saved to {output_file}")
            
#             # Generate trending report
#             trend_report = self.generate_trending_report(results)
            
#             return results
            
#         except Exception as e:
#             logger.error(f"Error analyzing file: {e}")
#             return None
    
#     def generate_trending_report(self, results):
#         """Generate a report of trending keywords and anomalies"""
#         try:
#             all_keywords = []
#             all_anomalies = []
#             anomaly_confidence = {}
            
#             for result in results:
#                 all_keywords.extend(result.get('keywords', []))
                
#                 # Process anomalies and their confidence
#                 anomalies = result.get('anomalies', [])
#                 confidence = result.get('confidence', 0.0)
                
#                 for anomaly in anomalies:
#                     all_anomalies.append(anomaly)
#                     # Store highest confidence for each unique anomaly
#                     if anomaly in anomaly_confidence:
#                         anomaly_confidence[anomaly] = max(anomaly_confidence[anomaly], confidence)
#                     else:
#                         anomaly_confidence[anomaly] = confidence
            
#             # Count keyword occurrences
#             keyword_counts = {}
#             for keyword in all_keywords:
#                 if keyword in keyword_counts:
#                     keyword_counts[keyword] += 1
#                 else:
#                     keyword_counts[keyword] = 1
            
#             # Sort by count
#             sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
            
#             # Format anomalies with confidence
#             formatted_anomalies = [
#                 {"anomaly": anomaly, "confidence": confidence} 
#                 for anomaly, confidence in anomaly_confidence.items()
#             ]
            
#             # Sort anomalies by confidence
#             sorted_anomalies = sorted(formatted_anomalies, key=lambda x: x['confidence'], reverse=True)
            
#             # Create trend report
#             report = {
#                 "timestamp": datetime.now().isoformat(),
#                 "trending_keywords": [{"keyword": k, "count": c} for k, c in sorted_keywords[:10]],
#                 "anomalies": sorted_anomalies,
#                 "total_sentences_analyzed": len(results)
#             }
            
#             # Save trend report
#             trend_file = "trend_report.json"
#             with open(trend_file, 'w') as f:
#                 json.dump(report, f, indent=2)
            
#             # Update stored trend data for API
#             self.trend_data = report
                
#             logger.info(f"Trend report generated: {trend_file}")
            
#             return report
            
#         except Exception as e:
#             logger.error(f"Error generating trend report: {e}")
#             return None
    
#     def watch_for_updates(self):
#         """Watch for file updates and process them"""
#         logger.info(f"Watching for updates to {self.watch_file} every {self.watch_interval} seconds")
        
#         while True:
#             self.analyze_file()
#             time.sleep(self.watch_interval)

# class FileChangeHandler(FileSystemEventHandler):
#     """Handler for file system events"""
    
#     def __init__(self, analyzer):
#         self.analyzer = analyzer
        
#     def on_modified(self, event):
#         if not event.is_directory and event.src_path.endswith(self.analyzer.watch_file):
#             logger.info(f"File {event.src_path} has been modified")
#             self.analyzer.analyze_file()

# # Flask Web Server HTML template
# HTML_TEMPLATE = """
# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>News Trend Analyzer</title>
#     <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" rel="stylesheet">
#     <style>
#         .anomaly-high { background-color: rgba(255, 99, 71, 0.2); }
#         .anomaly-medium { background-color: rgba(255, 165, 0, 0.2); }
#         .anomaly-low { background-color: rgba(255, 255, 0, 0.2); }
#         .card { margin-bottom: 15px; }
#         .refresh-btn { margin-bottom: 20px; }
#         #last-update { font-size: 12px; color: #666; margin-bottom: 20px; }
#         .trending-item { display: flex; justify-content: space-between; margin-bottom: 5px; }
#         .trending-bar { 
#             height: 20px; 
#             background-color: #007bff; 
#             margin-top: 5px;
#         }
#         .confidence-indicator {
#             height: 10px;
#             border-radius: 5px;
#             margin-top: 5px;
#         }
#     </style>
# </head>
# <body>
#     <div class="container mt-4">
#         <h1>News Trend Analyzer</h1>
#         <p id="status">Status: <span class="badge bg-success">Running</span></p>
#         <p id="model-info">Model: <span id="model-name">Loading...</span></p>
#         <p id="last-update">Last updated: <span id="update-time">Never</span></p>
        
#         <button id="refresh" class="btn btn-primary refresh-btn">Refresh Data</button>
        
#         <div class="row">
#             <!-- Recent News -->
#             <div class="col-md-6">
#                 <div class="card">
#                     <div class="card-header">
#                         <h5>Recent News Analysis</h5>
#                     </div>
#                     <div class="card-body">
#                         <div id="recent-news">Loading...</div>
#                     </div>
#                 </div>
#             </div>
            
#             <!-- Trending -->
#             <div class="col-md-6">
#                 <div class="card">
#                     <div class="card-header">
#                         <h5>Trending Keywords</h5>
#                     </div>
#                     <div class="card-body">
#                         <div id="trending-keywords">Loading...</div>
#                     </div>
#                 </div>
                
#                 <div class="card mt-3">
#                     <div class="card-header">
#                         <h5>Detected Anomalies</h5>
#                         <small class="text-muted">Unusual or unexpected trends</small>
#                     </div>
#                     <div class="card-body">
#                         <div id="detected-anomalies">Loading...</div>
#                     </div>
#                 </div>
#             </div>
#         </div>
#     </div>
    
#     <script>
#         document.addEventListener('DOMContentLoaded', function() {
#             // Fetch initial data
#             fetchData();
            
#             // Setup refresh button
#             document.getElementById('refresh').addEventListener('click', fetchData);
            
#             // Auto-refresh every 10 seconds
#             setInterval(fetchData, 10000);
            
#             function fetchData() {
#                 // Fetch model info
#                 fetch('/api/info')
#                     .then(response => response.json())
#                     .then(data => {
#                         document.getElementById('model-name').textContent = data.model;
#                     });
                
#                 // Fetch recent news
#                 fetch('/api/recent')
#                     .then(response => response.json())
#                     .then(data => {
#                         const recentNews = document.getElementById('recent-news');
#                         recentNews.innerHTML = '';
                        
#                         if (data.length === 0) {
#                             recentNews.innerHTML = '<p>No data available yet</p>';
#                             return;
#                         }
                        
#                         data.forEach(item => {
#                             // Determine anomaly class based on confidence
#                             let anomalyClass = '';
#                             if (item.anomalies && item.anomalies.length > 0) {
#                                 if (item.confidence > 0.7) {
#                                     anomalyClass = 'anomaly-high';
#                                 } else if (item.confidence > 0.4) {
#                                     anomalyClass = 'anomaly-medium';
#                                 } else if (item.confidence > 0) {
#                                     anomalyClass = 'anomaly-low';
#                                 }
#                             }
                            
#                             const newsItem = document.createElement('div');
#                             newsItem.className = `mb-3 p-2 ${anomalyClass}`;
                            
#                             let anomalyHtml = '';
#                             if (item.anomalies && item.anomalies.length > 0) {
#                                 anomalyHtml = `
#                                     <div class="mt-2">
#                                         <strong>Anomalies (${(item.confidence * 100).toFixed(0)}% confidence):</strong>
#                                         <ul>${item.anomalies.map(a => `<li>${a}</li>`).join('')}</ul>
#                                     </div>
#                                 `;
#                             }
                            
#                             newsItem.innerHTML = `
#                                 <p><strong>${item.sentence}</strong></p>
#                                 <div><small>Keywords: ${item.keywords.join(', ')}</small></div>
#                                 ${anomalyHtml}
#                                 <div class="text-muted mt-1"><small>${new Date(item.timestamp).toLocaleString()}</small></div>
#                             `;
                            
#                             recentNews.appendChild(newsItem);
#                         });
                        
#                         // Update last update time
#                         document.getElementById('update-time').textContent = new Date().toLocaleString();
#                     });
                
#                 // Fetch trends
#                 fetch('/api/trends')
#                     .then(response => response.json())
#                     .then(data => {
#                         // Update trending keywords
#                         const trendingKeywords = document.getElementById('trending-keywords');
#                         trendingKeywords.innerHTML = '';
                        
#                         if (!data.trending_keywords || data.trending_keywords.length === 0) {
#                             trendingKeywords.innerHTML = '<p>No trending keywords available yet</p>';
#                         } else {
#                             const maxCount = Math.max(...data.trending_keywords.map(k => k.count));
                            
#                             data.trending_keywords.forEach(keyword => {
#                                 const percent = (keyword.count / maxCount) * 100;
#                                 const item = document.createElement('div');
#                                 item.className = 'trending-item';
#                                 item.innerHTML = `
#                                     <div>${keyword.keyword} <span class="badge bg-secondary">${keyword.count}</span></div>
#                                     <div class="trending-bar" style="width: ${percent}%"></div>
#                                 `;
#                                 trendingKeywords.appendChild(item);
#                             });
#                         }
                        
#                         // Update anomalies
#                         const detectedAnomalies = document.getElementById('detected-anomalies');
#                         detectedAnomalies.innerHTML = '';
                        
#                         if (!data.anomalies || data.anomalies.length === 0) {
#                             detectedAnomalies.innerHTML = '<p>No anomalies detected yet</p>';
#                         } else {
#                             data.anomalies.forEach(anomaly => {
#                                 const item = document.createElement('div');
#                                 item.className = 'mb-3';
                                
#                                 // Determine color based on confidence
#                                 let color = '#28a745'; // Low - green
#                                 if (anomaly.confidence > 0.7) {
#                                     color = '#dc3545'; // High - red
#                                 } else if (anomaly.confidence > 0.4) {
#                                     color = '#fd7e14'; // Medium - orange
#                                 }
                                
#                                 item.innerHTML = `
#                                     <div>${anomaly.anomaly}</div>
#                                     <div>Confidence: ${(anomaly.confidence * 100).toFixed(0)}%</div>
#                                     <div class="confidence-indicator" style="width: ${anomaly.confidence * 100}%; background-color: ${color}"></div>
#                                 `;
#                                 detectedAnomalies.appendChild(item);
#                             });
#                         }
#                     });
#             }
#         });
#     </script>
# </body>
# </html>
# """

# class WebServer:
#     def __init__(self, analyzer, port=8000):
#         self.analyzer = analyzer
#         self.port = port
#         self.app = Flask(__name__)
#         self.setup_routes()
        
#     def setup_routes(self):
#         @self.app.route('/')
#         def home():
#             return render_template_string(HTML_TEMPLATE)
        
#         @self.app.route('/api/info')
#         def get_info():
#             return jsonify({
#                 'model': self.analyzer.model_name,
#                 'file': self.analyzer.watch_file,
#                 'interval': self.analyzer.watch_interval,
#                 'run_id': self.analyzer.run_id
#             })
        
#         @self.app.route('/api/recent')
#         def get_recent():
#             return jsonify(self.analyzer.all_results)
        
#         @self.app.route('/api/trends')
#         def get_trends():
#             return jsonify(self.analyzer.trend_data)
        
#         @self.app.route('/api/report')
#         def get_report():
#             try:
#                 with open('trend_report.json', 'r') as f:
#                     return jsonify(json.load(f))
#             except:
#                 return jsonify({"error": "Report not found"}), 404
    
#     def start(self):
#         threading.Thread(target=self._run_server, daemon=True).start()
#         logger.info(f"Web server started at http://localhost:{self.port}")
    
#     def _run_server(self):
#         self.app.run(host='0.0.0.0', port=self.port)

# def main():
#     import argparse
    
#     parser = argparse.ArgumentParser(description='News Trend Analyzer')
#     parser.add_argument('--model', type=str, default='phi', choices=['phi', 'tinyllama'],
#                         help='LLM model to use (phi or tinyllama)')
#     parser.add_argument('--file', type=str, default='./Trends_detector-main/rss_output.json',
#                         help='JSON file to watch for news data')
#     parser.add_argument('--interval', type=int, default=10,
#                         help='Interval in seconds to check for file updates')
#     parser.add_argument('--watch', action='store_true',
#                         help='Use watchdog to monitor file changes instead of polling')
#     parser.add_argument('--port', type=int, default=8000,
#                         help='Port for the web interface')
    
#     args = parser.parse_args()
    
#     analyzer = NewsAnalyzer(
#         model=args.model,
#         watch_file=args.file,
#         watch_interval=args.interval
#     )
    
#     # Start the web server
#     web_server = WebServer(analyzer, port=args.port)
#     web_server.start()  # Fixed: Actually start the web server
    
#     # Start the file watcher
#     if args.watch:
#         # Use watchdog for file monitoring
#         event_handler = FileChangeHandler(analyzer)
#         observer = Observer()
#         observer.schedule(event_handler, path=os.path.dirname(args.file) or '.', recursive=False)
#         observer.start()
#         logger.info(f"Started file watcher for {args.file}")
        
#         try:
#             while True:
#                 time.sleep(1)
#         except KeyboardInterrupt:
#             observer.stop()
#         observer.join()
#     else:
#         # Use polling method
#         analyzer.watch_for_updates()

# if __name__ == "__main__":
#     main()


# import json
# import time
# import os
# import requests
# import numpy as np
# from watchdog.observers import Observer
# from watchdog.events import FileSystemEventHandler
# from datetime import datetime
# import logging
# from flask import Flask, jsonify, render_template_string
# import threading
# import uuid

# # Set up logging
# logging.basicConfig(level=logging.INFO, 
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#                     handlers=[logging.FileHandler("news_analyzer.log"),
#                               logging.StreamHandler()])
# logger = logging.getLogger(__name__)

# class NewsAnalyzer:
#     def __init__(self, model="llama3", watch_file="./Trends_detector-main/rss_output.json", watch_interval=10):
#         """
#         Initialize the News Analyzer application
        
#         Args:
#             model (str): The model to use ('llama3' or 'tinyllama')
#             watch_file (str): The JSON file to watch for changes
#             watch_interval (int): How often to check for file changes (seconds)
#         """
#         self.model = model
#         self.watch_file = watch_file
#         self.watch_interval = watch_interval
#         self.last_modified = 0
#         self.recent_keywords = []
#         self.recent_sentences = []
#         self.max_history = 50  # Keep track of last 50 sentences for trend analysis
        
#         # Store analysis results for API access
#         self.all_results = []
#         self.trend_data = {"trending_keywords": [], "anomalies": [], "timestamp": ""}
        
#         # Unique run ID for this instance
#         self.run_id = str(uuid.uuid4())[:8]
        
#         # Model selection parameters - Updated to use llama3 as default
#         if model == "llama3":
#             self.model_name = "llama3:latest"
#         else:
#             self.model_name = "tinyllama:latest"
            
#         logger.info(f"Initialized NewsAnalyzer with model: {self.model_name}")
        
#         # Verify Ollama is running
#         self._check_ollama()
    
#     def _check_ollama(self):
#         """Check if Ollama is running and the model is available"""
#         try:
#             response = requests.get("http://localhost:11434/api/tags")
#             if response.status_code == 200:
#                 models = response.json().get("models", [])
#                 model_names = [m["name"] for m in models]
                
#                 if self.model_name not in model_names:
#                     logger.warning(f"Model {self.model_name} not found in Ollama. Available models: {model_names}")
#                     logger.info(f"Please run: ollama pull {self.model_name.split(':')[0]}")
#                 else:
#                     logger.info(f"Connected to Ollama. Using model: {self.model_name}")
#             else:
#                 logger.error("Couldn't connect to Ollama API")
#         except requests.exceptions.ConnectionError:
#             logger.error("Failed to connect to Ollama. Is it running? Start with 'ollama serve'")

#     def analyze_text(self, sentence):
#         """
#         Send text to Ollama model and extract keywords and anomalies
        
#         Args:
#             sentence (str): The news sentence to analyze
            
#         Returns:
#             dict: Analysis results including keywords and anomalies
#         """
#         # IMPROVED PROMPT: More structured prompt for keyword extraction 
#         keyword_prompt = f"""
#         Task: Extract 3-5 of the most significant, specific, and informative keywords from the news sentence below.
        
#         Guidelines:
#         - Focus on entities, events, topics, and concepts that are central to the news item
#         - Prefer specific terms over generic ones (e.g., "Tesla" over "company")
#         - Include proper nouns, technical terms, and domain-specific vocabulary
#         - Avoid common words, articles, and generic terms
#         - Select keywords that would be useful for categorization and trend analysis
        
#         News sentence: "{sentence}"

#         Respond ONLY in this JSON format:
#         {{
#           "keywords": ["keyword1", "keyword2", "keyword3"]
#         }}
#         """
        
#         # IMPROVED PROMPT: Much more detailed prompt for anomaly detection
#         anomaly_prompt = f"""
#         Task: Analyze the news sentence for anomalies - unusual, unexpected, or significant information that deviates from normal patterns or expectations.
        
#         News sentence: "{sentence}"
        
#         Context:
#         - Recent keywords: {', '.join(self.recent_keywords[-30:])}
#         - Recent news trends: {' | '.join(self.recent_sentences[-5:])}
        
#         Analysis guidelines:
#         1. Identify information that contradicts established patterns or expectations
#         2. Look for unexpected connections between entities or events
#         3. Detect unusual shifts in sentiment, policy, or behavior
#         4. Find statistical outliers or surprising numerical data
#         5. Spot emerging trends that could signal important shifts
#         6. Consider geopolitical, economic, technological, or social pattern breaks
        
#         For each anomaly:
#         - Be specific and descriptive
#         - Explain why it's anomalous (contradicts expectations, breaks patterns, etc.)
#         - Consider its potential significance for future trends
        
#         Respond ONLY in this strict JSON format:
#         {{
#           "anomalies": [
#             "Specific detailed anomaly 1", 
#             "Specific detailed anomaly 2"
#           ],
#           "confidence": X.X  // Value between 0.0-1.0 representing confidence in anomaly detection
#                         // 0.8-1.0: High confidence - clear pattern break
#                         // 0.5-0.7: Medium confidence - possible anomaly
#                         // 0.1-0.4: Low confidence - subtle deviation
#                         // 0.0: No anomalies detected
#         }}
#         """
        
#         try:
#             # Extract keywords with improved prompt
#             keyword_response = self._query_ollama(keyword_prompt)
#             keywords = self._extract_json_from_response(keyword_response)
            
#             # Extract anomalies with improved prompt
#             anomaly_response = self._query_ollama(anomaly_prompt)
#             anomalies = self._extract_json_from_response(anomaly_response)
            
#             # Default confidence
#             confidence = 0.0
            
#             # Calculate confidence if not provided explicitly
#             if anomalies:
#                 if 'confidence' in anomalies:
#                     confidence = anomalies.get('confidence', 0.0)
#                 elif 'anomalies' in anomalies and len(anomalies['anomalies']) > 0:
#                     # If anomalies exist but no confidence, assign default based on count
#                     confidence = min(0.7, 0.3 + (len(anomalies['anomalies']) * 0.2))
            
#             # Update recent keywords for context
#             if keywords and 'keywords' in keywords:
#                 self.recent_keywords.extend(keywords['keywords'])
#                 self.recent_keywords = self.recent_keywords[-100:]  # Keep last 100 keywords
            
#             # Update recent sentences
#             self.recent_sentences.append(sentence)
#             self.recent_sentences = self.recent_sentences[-self.max_history:]
            
#             # Combine results
#             result = {
#                 'sentence': sentence,
#                 'timestamp': datetime.now().isoformat(),
#                 'keywords': keywords.get('keywords', []) if keywords else [],
#                 'anomalies': anomalies.get('anomalies', []) if anomalies else [],
#                 'confidence': confidence
#             }
            
#             return result
            
#         except Exception as e:
#             logger.error(f"Error analyzing text: {e}")
#             return {
#                 'sentence': sentence,
#                 'timestamp': datetime.now().isoformat(),
#                 'keywords': [],
#                 'anomalies': [],
#                 'confidence': 0.0,
#                 'error': str(e)
#             }
    
#     def _query_ollama(self, prompt):
#         """Send a prompt to Ollama and get the response"""
#         try:
#             # Added temperature and max_tokens parameters for more consistent outputs
#             response = requests.post(
#                 "http://localhost:11434/api/generate",
#                 json={
#                     "model": self.model_name,
#                     "prompt": prompt,
#                     "stream": False,
#                     "temperature": 0.1,  # Lower temperature for more deterministic responses
#                     "num_predict": 512   # Limit token generation
#                 }
#             )
            
#             if response.status_code == 200:
#                 return response.json().get('response', '')
#             else:
#                 logger.error(f"Ollama API error: {response.status_code} - {response.text}")
#                 return ""
                
#         except Exception as e:
#             logger.error(f"Error querying Ollama: {e}")
#             return ""
    
#     def _extract_json_from_response(self, text):
#         """Extract JSON from Ollama's text response with improved robustness"""
#         try:
#             # First try to find JSON-like content in the response
#             start_idx = text.find('{')
#             end_idx = text.rfind('}')
            
#             if start_idx >= 0 and end_idx > start_idx:
#                 json_str = text[start_idx:end_idx+1]
#                 try:
#                     return json.loads(json_str)
#                 except json.JSONDecodeError:
#                     # If direct extraction fails, try cleaning the string
#                     cleaned_json = self._clean_json_string(json_str)
#                     return json.loads(cleaned_json)
#             else:
#                 # Try to extract a list if no JSON object found
#                 start_idx = text.find('[')
#                 end_idx = text.rfind(']')
#                 if start_idx >= 0 and end_idx > start_idx:
#                     json_str = text[start_idx:end_idx+1]
#                     return {"keywords": json.loads(json_str)} 
#                 else:
#                     # If we can't find JSON, create keywords from text
#                     words = [w.strip().lower() for w in text.split() if len(w.strip()) > 3]
#                     # Remove duplicates
#                     words = list(dict.fromkeys(words))
#                     return {"keywords": words[:5]}
                    
#         except json.JSONDecodeError:
#             logger.warning(f"Could not parse JSON from response: {text[:100]}...")
#             # Fall back to simple word extraction
#             words = [w.strip().lower() for w in text.split() if len(w.strip()) > 3]
#             words = list(dict.fromkeys(words))
#             return {"keywords": words[:5]}

#     def _clean_json_string(self, json_str):
#         """Clean and fix common JSON formatting issues"""
#         # Replace single quotes with double quotes (common LLM mistake)
#         json_str = json_str.replace("'", '"')
        
#         # Fix trailing commas in arrays/objects (another common issue)
#         json_str = json_str.replace(",]", "]").replace(",}", "}")
        
#         # Fix missing quotes around keys
#         import re
#         json_str = re.sub(r'([{,])\s*([a-zA-Z0-9_]+):', r'\1"\2":', json_str)
        
#         return json_str
    
#     def analyze_file(self):
#         """Analyze the news data file if it has been updated"""
#         try:
#             if not os.path.exists(self.watch_file):
#                 logger.warning(f"File {self.watch_file} not found")
#                 return None
                
#             # Check if file has been modified
#             current_modified = os.path.getmtime(self.watch_file)
#             if current_modified <= self.last_modified:
#                 return None
                
#             logger.info(f"Analyzing updated file: {self.watch_file}")
#             self.last_modified = current_modified
            
#             # Read and parse the JSON file
#             with open(self.watch_file, 'r') as f:
#                 data = json.load(f)
            
#             results = []
            
#             # Process each news sentence
#             if isinstance(data, list):
#                 for item in data:
#                     if isinstance(item, dict) and 'sentence' in item:
#                         result = self.analyze_text(item['sentence'])
#                         results.append(result)
#             elif isinstance(data, dict):
#                 # Handle case where JSON is a single object or has a different structure
#                 if 'sentences' in data and isinstance(data['sentences'], list):
#                     # Format with a sentences array
#                     for sentence in data['sentences']:
#                         if isinstance(sentence, str):
#                             result = self.analyze_text(sentence)
#                             results.append(result)
#                         elif isinstance(sentence, dict) and 'text' in sentence:
#                             result = self.analyze_text(sentence['text'])
#                             results.append(result)
#                 elif 'sentence' in data:
#                     # Single sentence entry
#                     result = self.analyze_text(data['sentence'])
#                     results.append(result)
            
#             # Save analysis results
#             output_file = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
#             with open(output_file, 'w') as f:
#                 json.dump(results, f, indent=2)
            
#             # Update the in-memory results for API access
#             self.all_results.extend(results)
#             # Keep only the most recent 100 results to avoid memory issues
#             self.all_results = self.all_results[-100:]
                
#             logger.info(f"Analysis complete. Results saved to {output_file}")
            
#             # Generate trending report
#             trend_report = self.generate_trending_report(results)
            
#             return results
            
#         except Exception as e:
#             logger.error(f"Error analyzing file: {e}")
#             return None
    
#     def generate_trending_report(self, results):
#         """Generate a report of trending keywords and anomalies with improved analytics"""
#         try:
#             all_keywords = []
#             all_anomalies = []
#             anomaly_confidence = {}
            
#             # Improved tracking - also track source sentence for each anomaly
#             anomaly_sources = {}
            
#             for result in results:
#                 # Extract and normalize keywords
#                 keywords = [k.lower().strip() for k in result.get('keywords', [])]
#                 all_keywords.extend(keywords)
                
#                 # Process anomalies and their confidence
#                 anomalies = result.get('anomalies', [])
#                 confidence = result.get('confidence', 0.0)
#                 sentence = result.get('sentence', '')
                
#                 for anomaly in anomalies:
#                     all_anomalies.append(anomaly)
#                     # Store highest confidence for each unique anomaly
#                     if anomaly in anomaly_confidence:
#                         anomaly_confidence[anomaly] = max(anomaly_confidence[anomaly], confidence)
#                     else:
#                         anomaly_confidence[anomaly] = confidence
                        
#                     # Store source sentence
#                     if anomaly in anomaly_sources:
#                         anomaly_sources[anomaly].append(sentence)
#                     else:
#                         anomaly_sources[anomaly] = [sentence]
            
#             # Count keyword occurrences with improved normalization
#             keyword_counts = {}
#             for keyword in all_keywords:
#                 # Normalize keywords
#                 keyword = keyword.lower().strip()
#                 if keyword in keyword_counts:
#                     keyword_counts[keyword] += 1
#                 else:
#                     keyword_counts[keyword] = 1
            
#             # Sort by count
#             sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
            
#             # Format anomalies with confidence and sources
#             formatted_anomalies = [
#                 {
#                     "anomaly": anomaly, 
#                     "confidence": confidence,
#                     "sources": anomaly_sources.get(anomaly, [])[:2]  # Include up to 2 source sentences
#                 } 
#                 for anomaly, confidence in anomaly_confidence.items()
#             ]
            
#             # Sort anomalies by confidence
#             sorted_anomalies = sorted(formatted_anomalies, key=lambda x: x['confidence'], reverse=True)
            
#             # Create trend report
#             report = {
#                 "timestamp": datetime.now().isoformat(),
#                 "trending_keywords": [{"keyword": k, "count": c} for k, c in sorted_keywords[:10]],
#                 "anomalies": sorted_anomalies,
#                 "total_sentences_analyzed": len(results)
#             }
            
#             # Save trend report
#             trend_file = "trend_report.json"
#             with open(trend_file, 'w') as f:
#                 json.dump(report, f, indent=2)
            
#             # Update stored trend data for API
#             self.trend_data = report
                
#             logger.info(f"Trend report generated: {trend_file}")
            
#             return report
            
#         except Exception as e:
#             logger.error(f"Error generating trend report: {e}")
#             return None
    
#     def watch_for_updates(self):
#         """Watch for file updates and process them"""
#         logger.info(f"Watching for updates to {self.watch_file} every {self.watch_interval} seconds")
        
#         while True:
#             self.analyze_file()
#             time.sleep(self.watch_interval)

# class FileChangeHandler(FileSystemEventHandler):
#     """Handler for file system events"""
    
#     def __init__(self, analyzer):
#         self.analyzer = analyzer
        
#     def on_modified(self, event):
#         if not event.is_directory and event.src_path.endswith(self.analyzer.watch_file):
#             logger.info(f"File {event.src_path} has been modified")
#             self.analyzer.analyze_file()

# # Flask Web Server HTML template
# HTML_TEMPLATE = """
# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>News Trend Analyzer</title>
#     <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" rel="stylesheet">
#     <style>
#         .anomaly-high { background-color: rgba(255, 99, 71, 0.2); }
#         .anomaly-medium { background-color: rgba(255, 165, 0, 0.2); }
#         .anomaly-low { background-color: rgba(255, 255, 0, 0.2); }
#         .card { margin-bottom: 15px; }
#         .refresh-btn { margin-bottom: 20px; }
#         #last-update { font-size: 12px; color: #666; margin-bottom: 20px; }
#         .trending-item { display: flex; justify-content: space-between; margin-bottom: 5px; }
#         .trending-bar { 
#             height: 20px; 
#             background-color: #007bff; 
#             margin-top: 5px;
#         }
#         .confidence-indicator {
#             height: 10px;
#             border-radius: 5px;
#             margin-top: 5px;
#         }
#         .source-text {
#             font-size: 0.8rem;
#             color: #666;
#             font-style: italic;
#             margin-top: 5px;
#             border-left: 3px solid #ccc;
#             padding-left: 10px;
#         }
#     </style>
# </head>
# <body>
#     <div class="container mt-4">
#         <h1>News Trend Analyzer</h1>
#         <p id="status">Status: <span class="badge bg-success">Running</span></p>
#         <p id="model-info">Model: <span id="model-name">Loading...</span></p>
#         <p id="last-update">Last updated: <span id="update-time">Never</span></p>
        
#         <button id="refresh" class="btn btn-primary refresh-btn">Refresh Data</button>
        
#         <div class="row">
#             <!-- Recent News -->
#             <div class="col-md-6">
#                 <div class="card">
#                     <div class="card-header bg-primary text-white">
#                         <h5 class="mb-0">Recent News Analysis</h5>
#                     </div>
#                     <div class="card-body">
#                         <div id="recent-news">Loading...</div>
#                     </div>
#                 </div>
#             </div>
            
#             <!-- Trending -->
#             <div class="col-md-6">
#                 <div class="card">
#                     <div class="card-header bg-success text-white">
#                         <h5 class="mb-0">Trending Keywords</h5>
#                     </div>
#                     <div class="card-body">
#                         <div id="trending-keywords">Loading...</div>
#                     </div>
#                 </div>
                
#                 <div class="card mt-3">
#                     <div class="card-header bg-warning">
#                         <h5 class="mb-0">Detected Anomalies</h5>
#                         <small class="text-muted">Unusual or unexpected trends</small>
#                     </div>
#                     <div class="card-body">
#                         <div id="detected-anomalies">Loading...</div>
#                     </div>
#                 </div>
#             </div>
#         </div>
#     </div>
    
#     <script>
#         document.addEventListener('DOMContentLoaded', function() {
#             // Fetch initial data
#             fetchData();
            
#             // Setup refresh button
#             document.getElementById('refresh').addEventListener('click', fetchData);
            
#             // Auto-refresh every 10 seconds
#             setInterval(fetchData, 10000);
            
#             function fetchData() {
#                 // Fetch model info
#                 fetch('/api/info')
#                     .then(response => response.json())
#                     .then(data => {
#                         document.getElementById('model-name').textContent = data.model;
#                     });
                
#                 // Fetch recent news
#                 fetch('/api/recent')
#                     .then(response => response.json())
#                     .then(data => {
#                         const recentNews = document.getElementById('recent-news');
#                         recentNews.innerHTML = '';
                        
#                         if (data.length === 0) {
#                             recentNews.innerHTML = '<p>No data available yet</p>';
#                             return;
#                         }
                        
#                         data.forEach(item => {
#                             // Determine anomaly class based on confidence
#                             let anomalyClass = '';
#                             if (item.anomalies && item.anomalies.length > 0) {
#                                 if (item.confidence > 0.7) {
#                                     anomalyClass = 'anomaly-high';
#                                 } else if (item.confidence > 0.4) {
#                                     anomalyClass = 'anomaly-medium';
#                                 } else if (item.confidence > 0) {
#                                     anomalyClass = 'anomaly-low';
#                                 }
#                             }
                            
#                             const newsItem = document.createElement('div');
#                             newsItem.className = `mb-3 p-2 ${anomalyClass}`;
                            
#                             let anomalyHtml = '';
#                             if (item.anomalies && item.anomalies.length > 0) {
#                                 anomalyHtml = `
#                                     <div class="mt-2">
#                                         <strong>Anomalies (${(item.confidence * 100).toFixed(0)}% confidence):</strong>
#                                         <ul>${item.anomalies.map(a => `<li>${a}</li>`).join('')}</ul>
#                                     </div>
#                                 `;
#                             }
                            
#                             newsItem.innerHTML = `
#                                 <p><strong>${item.sentence}</strong></p>
#                                 <div><small>Keywords: ${item.keywords.join(', ')}</small></div>
#                                 ${anomalyHtml}
#                                 <div class="text-muted mt-1"><small>${new Date(item.timestamp).toLocaleString()}</small></div>
#                             `;
                            
#                             recentNews.appendChild(newsItem);
#                         });
                        
#                         // Update last update time
#                         document.getElementById('update-time').textContent = new Date().toLocaleString();
#                     });
                
#                 // Fetch trends
#                 fetch('/api/trends')
#                     .then(response => response.json())
#                     .then(data => {
#                         // Update trending keywords
#                         const trendingKeywords = document.getElementById('trending-keywords');
#                         trendingKeywords.innerHTML = '';
                        
#                         if (!data.trending_keywords || data.trending_keywords.length === 0) {
#                             trendingKeywords.innerHTML = '<p>No trending keywords available yet</p>';
#                         } else {
#                             const maxCount = Math.max(...data.trending_keywords.map(k => k.count));
                            
#                             data.trending_keywords.forEach(keyword => {
#                                 const percent = (keyword.count / maxCount) * 100;
#                                 const item = document.createElement('div');
#                                 item.className = 'trending-item';
#                                 item.innerHTML = `
#                                     <div>${keyword.keyword} <span class="badge bg-secondary">${keyword.count}</span></div>
#                                     <div class="trending-bar" style="width: ${percent}%"></div>
#                                 `;
#                                 trendingKeywords.appendChild(item);
#                             });
#                         }
                        
#                         // Update anomalies with sources
#                         const detectedAnomalies = document.getElementById('detected-anomalies');
#                         detectedAnomalies.innerHTML = '';
                        
#                         if (!data.anomalies || data.anomalies.length === 0) {
#                             detectedAnomalies.innerHTML = '<p>No anomalies detected yet</p>';
#                         } else {
#                             data.anomalies.forEach(anomaly => {
#                                 const item = document.createElement('div');
#                                 item.className = 'mb-3';
                                
#                                 // Determine color based on confidence
#                                 let color = '#28a745'; // Low - green
#                                 if (anomaly.confidence > 0.7) {
#                                     color = '#dc3545'; // High - red
#                                 } else if (anomaly.confidence > 0.4) {
#                                     color = '#fd7e14'; // Medium - orange
#                                 }
                                
#                                 // Add source examples if available
#                                 let sourceHtml = '';
#                                 if (anomaly.sources && anomaly.sources.length > 0) {
#                                     sourceHtml = `
#                                         <div class="source-text">
#                                             Example: "${anomaly.sources[0]}"
#                                         </div>
#                                     `;
#                                 }
                                
#                                 item.innerHTML = `
#                                     <div><strong>${anomaly.anomaly}</strong></div>
#                                     <div>Confidence: ${(anomaly.confidence * 100).toFixed(0)}%</div>
#                                     <div class="confidence-indicator" style="width: ${anomaly.confidence * 100}%; background-color: ${color}"></div>
#                                     ${sourceHtml}
#                                 `;
#                                 detectedAnomalies.appendChild(item);
#                             });
#                         }
#                     });
#             }
#         });
#     </script>
# </body>
# </html>
# """

# class WebServer:
#     def __init__(self, analyzer, port=8000):
#         self.analyzer = analyzer
#         self.port = port
#         self.app = Flask(__name__)
#         self.setup_routes()
        
#     def setup_routes(self):
#         @self.app.route('/')
#         def home():
#             return render_template_string(HTML_TEMPLATE)
        
#         @self.app.route('/api/info')
#         def get_info():
#             return jsonify({
#                 'model': self.analyzer.model_name,
#                 'file': self.analyzer.watch_file,
#                 'interval': self.analyzer.watch_interval,
#                 'run_id': self.analyzer.run_id
#             })
        
#         @self.app.route('/api/recent')
#         def get_recent():
#             return jsonify(self.analyzer.all_results)
        
#         @self.app.route('/api/trends')
#         def get_trends():
#             return jsonify(self.analyzer.trend_data)
        
#         @self.app.route('/api/report')
#         def get_report():
#             try:
#                 with open('trend_report.json', 'r') as f:
#                     return jsonify(json.load(f))
#             except:
#                 return jsonify({"error": "Report not found"}), 404
    
#     def start(self):
#         threading.Thread(target=self._run_server, daemon=True).start()
#         logger.info(f"Web server started at http://localhost:{self.port}")
    
#     def _run_server(self):
#         self.app.run(host='0.0.0.0', port=self.port)

# def main():
#     import argparse
    
#     parser = argparse.ArgumentParser(description='News Trend Analyzer')
#     parser.add_argument('--model', type=str, default='llama3', choices=['llama3', 'tinyllama'],
#                         help='LLM model to use (llama3 or tinyllama)')
#     parser.add_argument('--file', type=str, default='./Trends_detector-main/rss_output.json',
#                         help='JSON file to watch for news data')
#     parser.add_argument('--interval', type=int, default=10,
#                         help='Interval in seconds to check for file updates')
#     parser.add_argument('--watch', action='store_true',
#                         help='Use watchdog to monitor file changes instead of polling')
#     parser.add_argument('--port', type=int, default=8000,
#                         help='Port for the web interface')
    
#     args = parser.parse_args()
    
#     analyzer = NewsAnalyzer(
#         model=args.model,
#         watch_file=args.file,
#         watch_interval=args.interval
#     )
    
#     # Start the web server
#     web_server = WebServer(analyzer, port=args.port)
#     web_server.start()
    
#     # Start the file watcher
#     if args.watch:
#         # Use watchdog for file monitoring
#         event_handler = FileChangeHandler(analyzer)
#         observer = Observer()
#         path = os.path.dirname(os.path.abspath(args.file)) or '.'
#         observer.schedule(event_handler, path=path, recursive=False)
#         observer.start()
#         logger.info(f"Started file watcher for {args.file}")
        
#         try:
#             while True:
#                 time.sleep(1)
#         except KeyboardInterrupt:
#             observer.stop()
#         observer.join()
#     else:
#         # Use polling method
#         analyzer.watch_for_updates()

# if __name__ == "__main__":
#     main()


import json
import time
import os
import requests
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from datetime import datetime
import logging
from flask import Flask, jsonify, render_template_string
import threading
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("news_analyzer.log"),
                              logging.StreamHandler()])
logger = logging.getLogger(__name__)

class NewsAnalyzer:
    def __init__(self, model="llama3", watch_file="./Trends_detector-main/rss_output.json", watch_interval=10):
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
            with open(self.watch_file, 'r') as f:
                data = json.load(f)
            
            results = []
            
            # Process each news item based on the new format
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        if 'title' in item:  # Process new format with 'title' field
                            source = item.get('source', '')
                            result = self.analyze_text(item['title'], source)
                            results.append(result)
                        elif 'sentence' in item:  # Maintain backward compatibility
                            result = self.analyze_text(item['sentence'])
                            results.append(result)
            elif isinstance(data, dict):
                # Handle case where JSON is a single object
                if 'title' in data:
                    source = data.get('source', '')
                    result = self.analyze_text(data['title'], source)
                    results.append(result)
                # Maintain backward compatibility with old format
                elif 'sentences' in data and isinstance(data['sentences'], list):
                    for sentence in data['sentences']:
                        if isinstance(sentence, str):
                            result = self.analyze_text(sentence)
                            results.append(result)
                        elif isinstance(sentence, dict) and 'text' in sentence:
                            result = self.analyze_text(sentence['text'])
                            results.append(result)
                elif 'sentence' in data:
                    result = self.analyze_text(data['sentence'])
                    results.append(result)
            
            # Save analysis results
            output_file = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Update the in-memory results for API access
            self.all_results.extend(results)
            # Keep only the most recent 100 results to avoid memory issues
            self.all_results = self.all_results[-100:]
                
            logger.info(f"Analysis complete. Results saved to {output_file}")
            
            # Generate trending report
            trend_report = self.generate_trending_report(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing file: {e}")
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

# Flask Web Server HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>News Trend Analyzer</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .anomaly-high { background-color: rgba(255, 99, 71, 0.2); }
        .anomaly-medium { background-color: rgba(255, 165, 0, 0.2); }
        .anomaly-low { background-color: rgba(255, 255, 0, 0.2); }
        .card { margin-bottom: 15px; }
        .refresh-btn { margin-bottom: 20px; }
        #last-update { font-size: 12px; color: #666; margin-bottom: 20px; }
        .trending-item { display: flex; justify-content: space-between; margin-bottom: 5px; }
        .trending-bar { 
            height: 20px; 
            background-color: #007bff; 
            margin-top: 5px;
        }
        .confidence-indicator {
            height: 10px;
            border-radius: 5px;
            margin-top: 5px;
        }
        .source-text {
            font-size: 0.8rem;
            color: #666;
            font-style: italic;
            margin-top: 5px;
            border-left: 3px solid #ccc;
            padding-left: 10px;
        }
        .news-source {
            font-size: 0.8rem;
            color: #007bff;
            margin-top: 2px;
        }
    </style>
</head>
<body>
    <div class="container mt-4">
        <h1>News Trend Analyzer</h1>
        <p id="status">Status: <span class="badge bg-success">Running</span></p>
        <p id="model-info">Model: <span id="model-name">Loading...</span></p>
        <p id="last-update">Last updated: <span id="update-time">Never</span></p>
        
        <button id="refresh" class="btn btn-primary refresh-btn">Refresh Data</button>
        
        <div class="row">
            <!-- Recent News -->
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header bg-primary text-white">
                        <h5 class="mb-0">Recent News Analysis</h5>
                    </div>
                    <div class="card-body">
                        <div id="recent-news">Loading...</div>
                    </div>
                </div>
            </div>
            
            <!-- Trending -->
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header bg-success text-white">
                        <h5 class="mb-0">Trending Keywords</h5>
                    </div>
                    <div class="card-body">
                        <div id="trending-keywords">Loading...</div>
                    </div>
                </div>
                
                <div class="card mt-3">
                    <div class="card-header bg-warning">
                        <h5 class="mb-0">Detected Anomalies</h5>
                        <small class="text-muted">Unusual or unexpected trends</small>
                    </div>
                    <div class="card-body">
                        <div id="detected-anomalies">Loading...</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Fetch initial data
            fetchData();
            
            // Setup refresh button
            document.getElementById('refresh').addEventListener('click', fetchData);
            
            // Auto-refresh every 10 seconds
            setInterval(fetchData, 10000);
            
            function fetchData() {
                // Fetch model info
                fetch('/api/info')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('model-name').textContent = data.model;
                    });
                
                // Fetch recent news
                fetch('/api/recent')
                    .then(response => response.json())
                    .then(data => {
                        const recentNews = document.getElementById('recent-news');
                        recentNews.innerHTML = '';
                        
                        if (data.length === 0) {
                            recentNews.innerHTML = '<p>No data available yet</p>';
                            return;
                        }
                        
                        data.forEach(item => {
                            // Determine anomaly class based on confidence
                            let anomalyClass = '';
                            if (item.anomalies && item.anomalies.length > 0) {
                                if (item.confidence > 0.7) {
                                    anomalyClass = 'anomaly-high';
                                } else if (item.confidence > 0.4) {
                                    anomalyClass = 'anomaly-medium';
                                } else if (item.confidence > 0) {
                                    anomalyClass = 'anomaly-low';
                                }
                            }
                            
                            const newsItem = document.createElement('div');
                            newsItem.className = `mb-3 p-2 ${anomalyClass}`;
                            
                            let anomalyHtml = '';
                            if (item.anomalies && item.anomalies.length > 0) {
                                anomalyHtml = `
                                    <div class="mt-2">
                                        <strong>Anomalies (${(item.confidence * 100).toFixed(0)}% confidence):</strong>
                                        <ul>${item.anomalies.map(a => `<li>${a}</li>`).join('')}</ul>
                                    </div>
                                `;
                            }
                            
                            // Display title instead of sentence and include source if available
                            const sourceHtml = item.source ? `<div class="news-source">Source: ${item.source}</div>` : '';
                            
                            newsItem.innerHTML = `
                                <p><strong>${item.title || item.sentence}</strong></p>
                                ${sourceHtml}
                                <div><small>Keywords: ${item.keywords.join(', ')}</small></div>
                                ${anomalyHtml}
                                <div class="text-muted mt-1"><small>${new Date(item.timestamp).toLocaleString()}</small></div>
                            `;
                            
                            recentNews.appendChild(newsItem);
                        });
                        
                        // Update last update time
                        document.getElementById('update-time').textContent = new Date().toLocaleString();
                    });
                
                // Fetch trends
                fetch('/api/trends')
                    .then(response => response.json())
                    .then(data => {
                        // Update trending keywords
                        const trendingKeywords = document.getElementById('trending-keywords');
                        trendingKeywords.innerHTML = '';
                        
                        if (!data.trending_keywords || data.trending_keywords.length === 0) {
                            trendingKeywords.innerHTML = '<p>No trending keywords available yet</p>';
                        } else {
                            const maxCount = Math.max(...data.trending_keywords.map(k => k.count));
                            
                            data.trending_keywords.forEach(keyword => {
                                const percent = (keyword.count / maxCount) * 100;
                                const item = document.createElement('div');
                                item.className = 'trending-item';
                                item.innerHTML = `
                                    <div>${keyword.keyword} <span class="badge bg-secondary">${keyword.count}</span></div>
                                    <div class="trending-bar" style="width: ${percent}%"></div>
                                `;
                                trendingKeywords.appendChild(item);
                            });
                        }
                        
                        // Update anomalies with sources
                        const detectedAnomalies = document.getElementById('detected-anomalies');
                        detectedAnomalies.innerHTML = '';
                        
                        if (!data.anomalies || data.anomalies.length === 0) {
                            detectedAnomalies.innerHTML = '<p>No anomalies detected yet</p>';
                        } else {
                            data.anomalies.forEach(anomaly => {
                                const item = document.createElement('div');
                                item.className = 'mb-3';
                                
                                // Determine color based on confidence
                                let color = '#28a745'; // Low - green
                                if (anomaly.confidence > 0.7) {
                                    color = '#dc3545'; // High - red
                                } else if (anomaly.confidence > 0.4) {
                                    color = '#fd7e14'; // Medium - orange
                                }
                                
                                // Add source examples if available
                                let sourceHtml = '';
                                if (anomaly.sources && anomaly.sources.length > 0) {
                                    sourceHtml = `
                                        <div class="source-text">
                                            Example: "${anomaly.sources[0]}"
                                        </div>
                                    `;
                                }
                                
                                item.innerHTML = `
                                    <div><strong>${anomaly.anomaly}</strong></div>
                                    <div>Confidence: ${(anomaly.confidence * 100).toFixed(0)}%</div>
                                    <div class="confidence-indicator" style="width: ${anomaly.confidence * 100}%; background-color: ${color}"></div>
                                    ${sourceHtml}
                                `;
                                detectedAnomalies.appendChild(item);
                            });
                        }
                    });
            }
        });
    </script>
</body>
</html>
"""

class WebServer:
    def __init__(self, analyzer, port=8000):
        self.analyzer = analyzer
        self.port = port
        self.app = Flask(__name__)
        self.setup_routes()
        
    def setup_routes(self):
        @self.app.route('/')
        def home():
            return render_template_string(HTML_TEMPLATE)
        
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
    
    def start(self):
        threading.Thread(target=self._run_server, daemon=True).start()
        logger.info(f"Web server started at http://localhost:{self.port}")
    
    def _run_server(self):
        self.app.run(host='0.0.0.0', port=self.port)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='News Trend Analyzer')
    parser.add_argument('--model', type=str, default='llama3', choices=['llama3', 'tinyllama'],
                        help='LLM model to use (llama3 or tinyllama)')
    parser.add_argument('--file', type=str, default='./Trends_detector-main/rss_output.json',
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
    
    # Start the web server
    web_server = WebServer(analyzer, port=args.port)
    web_server.start()
    
    # Start the file watcher
    if args.watch:
        # Use watchdog for file monitoring
        event_handler = FileChangeHandler(analyzer)
        observer = Observer()
        path = os.path.dirname(os.path.abspath(args.file)) or '.'
        observer.schedule(event_handler, path=path, recursive=False)
        observer.start()
        logger.info(f"Started file watcher for {args.file}")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()
    else:
        # Use polling method
        analyzer.watch_for_updates()

if __name__ == "__main__":
    main()