#!/usr/bin/env python3
# services/ingestion/common/base_connector.py

import time
import logging
import requests
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from kafka import KafkaProducer
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@dataclass
class ApiCredentials:
    """Container for API credentials and authentication details."""
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    bearer_token: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    app_id: Optional[str] = None
    app_secret: Optional[str] = None
    
    def to_dict(self) -> Dict[str, str]:
        """Convert non-None credentials to dictionary."""
        return {k: v for k, v in self.__dict__.items() if v is not None}

class RateLimiter:
    """Implements rate limiting for API requests."""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.minimum_interval = 60.0 / requests_per_minute
        self.last_request_time = 0
        
    def wait_if_needed(self):
        """Wait if necessary to respect the rate limit."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.minimum_interval:
            wait_time = self.minimum_interval - elapsed
            logging.debug(f"Rate limiting: waiting {wait_time:.2f} seconds")
            time.sleep(wait_time)
            
        self.last_request_time = time.time()

class RetryPolicy:
    """Implements retry logic with exponential backoff."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a specific retry attempt with exponential backoff."""
        delay = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
        jitter = delay * 0.1 * (2 * (0.5 - time.time() % 1))  # Add some randomness
        return delay + jitter

class BaseConnector(ABC):
    """Base class for all data source connectors."""
    
    def __init__(self, 
                 credentials: ApiCredentials,
                 base_url: str,
                 logger: Optional[logging.Logger] = None,
                 rate_limit: int = 60,
                 retry_policy: Optional[RetryPolicy] = None,
                 source_name: Optional[str] = None,
                 kafka_topic: Optional[str] = None,
                 max_retries: Optional[int] = 3,
                 retry_delay: Optional[int] = 5
                 ):
        self.credentials = credentials
        self.base_url = base_url
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.rate_limiter = RateLimiter(rate_limit)
        self.retry_policy = retry_policy or RetryPolicy()
        self.session = self._create_session()
        self.max_retries = max_retries
        self.retry_delay = retry_delay

          # ✅ Kafka config
        self.kafka_topic = kafka_topic
        if kafka_topic:
            self.producer = KafkaProducer(
                bootstrap_servers=['localhost:9092'],  # Change as needed
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
        else:
            self.producer = None
            
     # ✅ Kafka send method
    def send_to_kafka(self, data: Dict[str, Any]):
        """Send transformed data to Kafka."""
        if self.producer and self.kafka_topic:
            try:
                self.producer.send(self.kafka_topic, value=data)
                self.producer.flush()
                self.logger.info(f"Data sent to Kafka topic '{self.kafka_topic}'")
            except Exception as e:
                self.logger.error(f"Failed to send data to Kafka: {str(e)}")
        else:
            self.logger.warning("Kafka producer not configured or topic not set. Skipping Kafka send.")

    def _create_session(self) -> requests.Session:
        """Create and configure a requests session."""
        session = requests.Session()
        # Add default headers, etc.
        session.headers.update({
            'User-Agent': 'TrendIntelligencePlatform/1.0',
            'Accept': 'application/json',
        })
        return session
    
    def _make_request(self, 
                     method: str, 
                     endpoint: str, 
                     params: Optional[Dict[str, Any]] = None,
                     data: Optional[Dict[str, Any]] = None,
                     headers: Optional[Dict[str, str]] = None,
                     authenticate: bool = True) -> Dict[str, Any]:
        """
        Makes an HTTP request with retry logic and rate limiting.
        
        Args:
            method: HTTP method (GET, POST, etc)
            endpoint: API endpoint to call
            params: Query parameters
            data: Request body for POST/PUT
            headers: Additional headers
            authenticate: Whether to add authentication headers
            
        Returns:
            JSON response as dictionary
        """
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        request_headers = {}
        
        if headers:
            request_headers.update(headers)
            
        if authenticate:
            auth_headers = self._get_auth_headers()
            request_headers.update(auth_headers)
        
        self.logger.debug(f"Making {method} request to {url}")
        
        for attempt in range(1, self.retry_policy.max_retries + 1):
            try:
                # Apply rate limiting
                self.rate_limiter.wait_if_needed()
                
                # Make the request
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=data,
                    headers=request_headers,
                )
                
                # Check for rate limiting response
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    self.logger.warning(f"Rate limited. Waiting {retry_after} seconds.")
                    time.sleep(retry_after)
                    continue
                    
                # Handle other error responses
                response.raise_for_status()
                
                # Success! Return the data
                return response.json()
                
            except requests.exceptions.RequestException as e:
                if attempt == self.retry_policy.max_retries:
                    self.logger.error(f"Request failed after {attempt} attempts: {str(e)}")
                    raise
                
                delay = self.retry_policy.get_delay(attempt)
                self.logger.warning(f"Request failed (attempt {attempt}): {str(e)}. Retrying in {delay:.2f}s")
                time.sleep(delay)
        
        # This should never be reached due to the raise in the except block
        raise RuntimeError("Unexpected error in request retry logic")
    
    @abstractmethod
    def _get_auth_headers(self) -> Dict[str, str]:
        """
        Return authentication headers if needed for requests.
        """
        # Example for bearer token authorization
        return {
            'Authorization': f'Bearer {self.credentials.bearer_token}' if self.credentials.bearer_token else ''
        }
    
    @abstractmethod
    def fetch_data(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Fetch data from the specific API.
        """
        # For example, you could fetch data like posts from Reddit or another API
        return []

    @abstractmethod
    def transform_data(self, raw_data: Any) -> Any:
        """
        Transform raw data into a specific format.
        """
        # Example: raw_data could be transformed into a standardized dictionary format
        return raw_data  # or apply some transformation to the raw data here

    def ingest(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Full ingestion process: fetch and transform data.
        
        Returns:
            Standardized data ready for processing
        """
        raw_data = self.fetch_data(**kwargs)
        return self.transform_data(raw_data)
    
    def handle_pagination(self, 
                         endpoint: str, 
                         params: Dict[str, Any],
                         data_key: str,
                         next_token_key: str,
                         next_token_param: str,
                         max_pages: int = 5) -> List[Dict[str, Any]]:
        """
        Generic pagination handler.
        
        Args:
            endpoint: API endpoint
            params: Initial query parameters
            data_key: JSON key containing the data items
            next_token_key: JSON key containing the next page token
            next_token_param: Query parameter name for the next page token
            max_pages: Maximum number of pages to fetch
            
        Returns:
            Combined data from all pages
        """
        all_data = []
        page_count = 0
        next_token = None
        
        while page_count < max_pages:
            page_count += 1
            
            # Add next token to params if it exists
            if next_token:
                params[next_token_param] = next_token
            
            # Make the request
            response = self._make_request("GET", endpoint, params=params)
            
            # Extract data
            if data_key in response:
                page_data = response[data_key]
                all_data.extend(page_data)
                self.logger.info(f"Fetched page {page_count} with {len(page_data)} items")
            else:
                self.logger.warning(f"No '{data_key}' found in response")
                break
            
            # Check for next page token
            next_token = response.get(next_token_key)
            if not next_token:
                break
                
        return all_data
    
    def close(self):
        """Clean up resources."""
        self.session.close()
