import os
import time
import logging
import requests
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import praw

from services.ingestion.base_connector import BaseConnector, ApiCredentials, RetryPolicy

import logging

# Import the base connector class
from services.ingestion.base_connector import BaseConnector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class RedditConnector(BaseConnector):
    """
    Reddit API connector for fetching data from Reddit using PRAW.
    
    This connector handles authentication, rate limiting, and data transformation
    for Reddit data sources.
    """
    
    def __init__(
        self,
        credentials: ApiCredentials,
        base_url: str,
        client_id: str,
        client_secret: str,
        user_agent: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        kafka_topic: str = "reddit_data",
        max_retries: int = 3,
        retry_delay: int = 5,
        logger: Optional[logging.Logger] = None,
        rate_limit: int = 60
    ):
        """
        Initialize the Reddit connector with authentication credentials.
        
        Args:
            client_id: Reddit API client ID
            client_secret: Reddit API client secret
            user_agent: User agent string for Reddit API
            username: Reddit username (optional for read-only access)
            password: Reddit password (optional for read-only access)
            kafka_topic: Kafka topic to publish data to
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
        """
        # Pass the necessary parameters to the base class
        super().__init__(
            credentials=credentials,
            base_url=base_url,
            rate_limit=rate_limit,
            kafka_topic=kafka_topic,
            max_retries=max_retries,
            retry_delay=retry_delay,
            logger=logger 
        )
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize PRAW Reddit client
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            username=username,
            password=password
        )
        
        self.logger.info("Reddit connector initialized")
    
    def _get_auth_headers(self) -> dict:
        """
        Get the authentication headers needed for Reddit API requests.
        
        Returns:
            Authentication headers dictionary
        """
        return {
            'Authorization': f'Bearer {self.reddit.auth.access_token}'  # Ensure you replace with correct logic for token handling
        }

    def fetch_data(self, **kwargs) -> list:
        """
        Fetch data from Reddit based on provided parameters.
        
        Args:
            **kwargs: Parameters for data fetching (e.g., subreddit_name, category)
            
        Returns:
            List of fetched posts
        """
        subreddit_name = kwargs.get("subreddit_name")
        category = kwargs.get("category", "hot")
        limit = kwargs.get("limit", 100)
        time_filter = kwargs.get("time_filter", "day")
        
        # Call the fetch_subreddit_posts method to fetch posts
        return self.fetch_subreddit_posts(
            subreddit_name=subreddit_name, 
            limit=limit, 
            category=category, 
            time_filter=time_filter
        )
    
    def transform_data(self, raw_data: Any) -> Any:
        """
        Transform raw data into a specific format and send it to Kafka.
        
        Args:
            raw_data: Raw data fetched from Reddit (e.g., posts)
            
        Returns:
            Transformed data (e.g., a cleaned-up version of posts)
        """
        transformed_data = []
        for item in raw_data:
            post = self._transform_post(item)
            transformed_data.append(post)
            self.send_to_kafka(post)  # ✅ Send each transformed post to Kafka

        return transformed_data

    def _handle_rate_limit(self, response: requests.Response) -> bool:
        """
        Handle rate limiting by checking response headers and sleeping if necessary.
        
        Args:
            response: Response object from Reddit API
            
        Returns:
            True if rate limited, False otherwise
        """
        # Check if we're rate limited
        if response.status_code == 429:
            self.logger.warning("Rate limited by Reddit API")
            
            # Get rate limit info from headers
            reset_time = int(response.headers.get('X-Ratelimit-Reset', 0))
            if reset_time > 0:
                wait_time = max(reset_time - time.time(), 1)
                self.logger.info(f"Waiting {wait_time:.2f} seconds for rate limit reset")
                time.sleep(wait_time)
            else:
                # Default wait time if header info is not available
                self.logger.info(f"Waiting {self.retry_delay} seconds before retrying")
                time.sleep(self.retry_delay)
            
            return True
        
        return False
    
    def fetch_subreddit_posts(
        self, 
        subreddit_name: str, 
        limit: int = 100, 
        time_filter: str = "day",
        category: str = "hot"
    ) -> List[Dict[str, Any]]:
        """
        Fetch posts from a specific subreddit.
        
        Args:
            subreddit_name: Name of the subreddit to fetch posts from
            limit: Maximum number of posts to fetch
            time_filter: Time filter for posts (hour, day, week, month, year, all)
            category: Category of posts to fetch (hot, new, top, rising, controversial)
            
        Returns:
            List of post dictionaries
        """
        subreddit_name = "python"
        self.logger.info(f"Fetching {limit} {category} posts from r/{subreddit_name}")
        if not subreddit_name:
          raise ValueError("Subreddit name cannot be None")
        subreddit = self.reddit.subreddit(subreddit_name)
        posts = []
        
        try:
            # Get the posts based on the category
            if category == "hot":
                submission_generator = subreddit.hot(limit=limit)
            elif category == "new":
                submission_generator = subreddit.new(limit=limit)
            elif category == "top":
                submission_generator = subreddit.top(time_filter=time_filter, limit=limit)
            elif category == "rising":
                submission_generator = subreddit.rising(limit=limit)
            elif category == "controversial":
                submission_generator = subreddit.controversial(time_filter=time_filter, limit=limit)
            else:
                self.logger.error(f"Invalid category: {category}")
                return []
            
            # Process each submission
            for submission in submission_generator:
                post_data = self._transform_post(submission)
                posts.append(post_data)
                
            self.logger.info(f"Successfully fetched {len(posts)} posts from r/{subreddit_name}")
            return posts
            
        except Exception as e:
            self.logger.error(f"Error fetching posts from r/{subreddit_name}: {str(e)}")
            raise
    
    def fetch_post_comments(
        self, 
        post_id: str, 
        limit: Optional[int] = None,
        sort: str = "top"
    ) -> List[Dict[str, Any]]:
        """
        Fetch comments for a specific post.
        
        Args:
            post_id: ID of the post to fetch comments for
            limit: Maximum number of comments to fetch
            sort: Sort order for comments (top, new, controversial, old, qa)
            
        Returns:
            List of comment dictionaries
        """
        self.logger.info(f"Fetching comments for post {post_id}")
        
        try:
            submission = self.reddit.submission(id=post_id)
            
            # Set the comment sort
            submission.comment_sort = sort
            
            # Replace MoreComments objects with actual comments if needed
            if limit is not None:
                submission.comments.replace_more(limit=0)
            
            # Extract comments
            comments = []
            self._process_comments(submission.comments, comments)
            
            if limit is not None:
                comments = comments[:limit]
                
            self.logger.info(f"Successfully fetched {len(comments)} comments for post {post_id}")
            return comments
            
        except Exception as e:
            self.logger.error(f"Error fetching comments for post {post_id}: {str(e)}")
            raise
    
    def _process_comments(
        self, 
        comments_forest, 
        result: List[Dict[str, Any]], 
        parent_id: Optional[str] = None,
        depth: int = 0
    ) -> None:
        """
        Process comment forest recursively to extract comment data.
        
        Args:
            comments_forest: PRAW comment forest object
            result: List to store comment dictionaries
            parent_id: ID of the parent comment
            depth: Current depth in the comment tree
        """
        for comment in comments_forest:
            # Skip MoreComments objects
            if isinstance(comment, praw.models.MoreComments):
                continue
                
            comment_data = self._transform_comment(comment, parent_id, depth)
            result.append(comment_data)
            
            # Process replies recursively
            if comment.replies:
                self._process_comments(comment.replies, result, comment.id, depth + 1)
    
    def search_reddit(
        self, 
        query: str, 
        subreddit: Optional[str] = None,
        sort: str = "relevance",
        time_filter: str = "all",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search Reddit for posts matching a query.
        
        Args:
            query: Search query string
            subreddit: Optional subreddit to limit search to
            sort: Sort order (relevance, hot, new, top, comments)
            time_filter: Time filter (hour, day, week, month, year, all)
            limit: Maximum number of results to return
            
        Returns:
            List of post dictionaries matching the search criteria
        """
        self.logger.info(f"Searching Reddit for '{query}'")
        
        try:
            if subreddit:
                search_results = self.reddit.subreddit(subreddit).search(
                    query, sort=sort, time_filter=time_filter, limit=limit
                )
            else:
                search_results = self.reddit.subreddit("all").search(
                    query, sort=sort, time_filter=time_filter, limit=limit
                )
            
            results = []
            for post in search_results:
                post_data = self._transform_post(post)
                results.append(post_data)
                
            self.logger.info(f"Found {len(results)} results for query '{query}'")
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching Reddit for '{query}': {str(e)}")
            raise
    
    def stream_subreddit_posts(self, subreddit_name: str, duration: int = 60) -> None:
        """
        Stream new posts from a subreddit for a specified duration.
        
        Args:
            subreddit_name: Name of the subreddit to stream posts from
            duration: Duration to stream in seconds
        """
        if not subreddit_name:
         self.logger.error("Subreddit name is None or empty!")
         raise ValueError("Subreddit name cannot be None or empty.")
    
        self.logger.info(f"Streaming posts from r/{subreddit_name} for {duration} seconds")
        
        subreddit = self.reddit.subreddit(subreddit_name)
        end_time = datetime.now() + timedelta(seconds=duration)
        
        try:
            for submission in subreddit.stream.submissions():
                if datetime.now() > end_time:
                    break
                    
                post_data = self._transform_post(submission)
                self.send_to_kafka(post_data)
                
        except Exception as e:
            self.logger.error(f"Error streaming posts from r/{subreddit_name}: {str(e)}")
            raise
            
        self.logger.info(f"Finished streaming posts from r/{subreddit_name}")
    
    def _transform_post(self, submission) -> Dict[str, Any]:
        """
        Transform a PRAW submission object into a standardized dictionary.

        Args:
            submission: PRAW submission object or a dictionary containing post data

        Returns:
            Dictionary containing post data
        """
        # If it's a dict, fetch the actual submission
        if isinstance(submission, dict):
            self.logger.warning(f"Expected praw.models.Submission, got dict. Fetching using ID: {submission.get('id')}")
            submission = self.reddit.submission(id=submission["id"])

        if isinstance(submission, praw.models.Submission):
            return {
                "id": submission.id,
                "type": "post",
                "title": submission.title,
                "author": submission.author.name if submission.author else "[deleted]",
                "subreddit": submission.subreddit.display_name,
                "selftext": submission.selftext,
                "url": submission.url,
                "permalink": f"https://www.reddit.com{submission.permalink}",
                "created_utc": submission.created_utc,
                "score": submission.score,
                "upvote_ratio": submission.upvote_ratio,
                "num_comments": submission.num_comments,
                "is_video": submission.is_video,
                "media": submission.media,
            }

        self.logger.error(f"Unexpected submission type: {type(submission)}")
        raise ValueError(f"Expected praw.models.Submission, but got {type(submission)}")


    def convert_dicts_to_submissions(self, posts: List[Dict[str, Any]]) -> List[praw.models.Submission]:
        submissions = []
        for post in posts:
            try:
                submission = self.reddit.submission(id=post["id"])
                submissions.append(submission)
            except Exception as e:
                self.logger.error(f"Failed to convert post dict to Submission: {e}")
        return submissions
    
    def _transform_comment(self, comment, parent_id: Optional[str], depth: int) -> Dict[str, Any]:
        """
        Transform a PRAW comment object into a standardized dictionary.
        
        Args:
            comment: PRAW comment object
            parent_id: ID of the parent comment
            depth: Depth in the comment tree
            
        Returns:
            Dictionary containing comment data
        """
        return {
            "id": comment.id,
            "type": "comment",
            "author": comment.author.name if comment.author else "[deleted]",
            "parent_id": parent_id,
            "depth": depth,
            "body": comment.body,
            "score": comment.score,
            "created_utc": comment.created_utc,
            "permalink": f"https://www.reddit.com{comment.permalink}",
        }
