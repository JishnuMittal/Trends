import feedparser
import yake

def fetch_rss_titles():
    rss_urls = [
        "http://feeds.bbci.co.uk/news/rss.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml"
    ]

    titles = []

    for url in rss_urls:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            title = entry.title
            titles.append({"title": title, "source": url})  # You can add more metadata if needed

    return titles
