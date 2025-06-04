import multiprocessing
import logging
from fetch_socket import NewsScraperBroadcaster, process_sitemaps
from news_analyzer import NewsAnalyzer, WebServer
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_scraper(port):
    # Initialize and run the scraper
    try:
        scraper = NewsScraperBroadcaster(port=port)
        
        # Your sitemap URLs
        sitemap_urls = [
            "https://www.shiksha.com/updates.xml",
            "https://www.shiksha.com/NewsIndex1.xml",
            "https://www.shiksha.com/sitemap_index.xml",
            "https://opendoorsdata.org/sitemap_index.xml",
            "https://opendoorsdata.org/fact_sheets-sitemap.xml",
            "https://opendoorsdata.org/data-sitemap.xml",
            "https://opendoorsdata.org/fast_facts-sitemap.xml",
            "https://www.educations.com/sitemap-index.xml",
            "https://thepienews.com/sitemap_index.xml"
        ]
        
        @scraper.app.before_first_request
        def start_background_task():
            import asyncio
            asyncio.run(process_sitemaps(scraper, sitemap_urls))
        
        scraper.start()
    except Exception as e:
        logger.error(f"Scraper process error: {e}")
        raise

def run_analyzer(port):
    # Initialize and run the analyzer
    try:
        analyzer = NewsAnalyzer(
            model="llama3",
            watch_file="rss_output.json",
            watch_interval=10
        )
        web_server = WebServer(analyzer, port=port)
        web_server.start()
    except Exception as e:
        logger.error(f"Analyzer process error: {e}")
        raise

def main():
    # Set environment variable to allow unsafe Werkzeug
    os.environ['WERKZEUG_RUN_MAIN'] = 'true'
    
    # Start scraper and analyzer in separate processes
    scraper_process = multiprocessing.Process(
        target=run_scraper,
        args=(8765,),
        name="ScrapeProcess"
    )
    analyzer_process = multiprocessing.Process(
        target=run_analyzer,
        args=(8000,),
        name="AnalyzeProcess"
    )
    
    # Start processes
    try:
        scraper_process.start()
        logger.info("Started scraper process")
        
        # Small delay between process starts
        import time
        time.sleep(2)
        
        analyzer_process.start()
        logger.info("Started analyzer process")
        
        # Monitor processes
        while True:
            if not scraper_process.is_alive():
                logger.error("Scraper process died")
                break
            if not analyzer_process.is_alive():
                logger.error("Analyzer process died")
                break
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Error in main process: {e}")
    finally:
        # Clean shutdown
        for p in [scraper_process, analyzer_process]:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    import signal
                    os.kill(p.pid, signal.SIGKILL)

if __name__ == "__main__":
    # Required for Windows compatibility
    multiprocessing.freeze_support()
    main() 