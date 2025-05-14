from kafka import KafkaConsumer
import json
import logging

# --- Logger setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RSSConsumer")

# --- Kafka Consumer setup ---
consumer = KafkaConsumer(
    'rss_data',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',  # or 'latest' depending on use case
    enable_auto_commit=True,
    group_id='rss-consumer-group',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

# --- Consume messages ---
logger.info("Starting RSS consumer...")

try:
    for message in consumer:
        data = message.value
        logger.info(f"Received: {data['title']}")
        # Optional: Add further processing logic here
        # e.g., keyword extraction, storage, etc.
except KeyboardInterrupt:
    logger.info("Consumer stopped by user.")
finally:
    consumer.close()
