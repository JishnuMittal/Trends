from services.ingestion.social_connectors.reddit_connector import RedditConnector
from services.ingestion.base_connector import ApiCredentials
from services.ingestion.news_connectors.rss_connector import fetch_rss_titles

from kafka import KafkaProducer
import logging
import json

# --- Logger setup ---
logger = logging.getLogger("IngestionScript")
logger.setLevel(logging.INFO)

# --- Kafka Producer setup ---
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# --- Reddit Setup ---
rate_limit = 60
subreddit_name = "python"

credentials = ApiCredentials(
    app_id="m-lKdy4O5_cnERIMel6mtQ",
    app_secret="HA6P5zB53w4GMkWm1V9KJ7qhbolaOw",
    username="hamosepian"
)

reddit_connector = RedditConnector(
    credentials=credentials,
    base_url="https://oauth.reddit.com",
    client_id="m-lKdy4O5_cnERIMel6mtQ",
    client_secret="HA6P5zB53w4GMkWm1V9KJ7qhbolaOw",
    user_agent="hamosepian",
    logger=logger,
    rate_limit=rate_limit
)

# --- RSS Logic ---
# def send_rss_to_kafka():
#     logger.info("Fetching RSS feed titles...")
#     rss_data = fetch_rss_titles()
#     for item in rss_data:
#         producer.send("rss_data", item)
#         logger.info(f"Sent RSS item to Kafka: {item['title']}")
def send_rss_to_kafka():
    logger.info("Fetching RSS feed titles...")
    rss_data = fetch_rss_titles()

    with open("rss_output.json", "w", encoding="utf-8") as f:
        json.dump(rss_data, f, ensure_ascii=False, indent=4)

    for item in rss_data:
        producer.send("rss_data", item)
        logger.info(f"Sent RSS item to Kafka: {item['title']}")

    producer.flush()  # Ensure all messages are delivered


# --- Main Execution ---
if __name__ == "__main__":
    # Ingest Reddit data and send to Kafka
    # reddit_connector.fetch_subreddit_posts(subreddit_name=subreddit_name)
    # reddit_connector.ingest()

    # Ingest RSS feed data and send to Kafka
    send_rss_to_kafka()
