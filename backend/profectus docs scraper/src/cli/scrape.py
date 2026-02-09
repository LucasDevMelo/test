from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from src.lib.config import load_config
from src.lib.logging import get_logger
from src.models.db import Database, ExportPaths
from src.services.http import HttpClient
from src.services.robots import RobotsPolicy
from src.services.frontier import UrlFrontier
from src.services.parse import parse_html
from src.services.persist import upsert_page
from src.services.sitemap import fetch_sitemap_seeds
from src.services.export import export_pages_csv, export_pages_full_csv


logger = get_logger("cli")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profectus Docs Scraper")
    parser.add_argument("--config", default="config/scraper.yaml", help="Path to YAML config")
    parser.add_argument("--mode", choices=["scrape", "update"], required=True)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--all", action="store_true", help="Apply to all in-scope URLs")
    selector.add_argument("--filter", dest="filter_pattern", help="Substring/regex filter for URLs")
    selector.add_argument("--url", dest="single_url", help="Operate on a single absolute URL")
    parser.add_argument("--report", choices=["url-index","pages-csv","pages-full-csv"], help="Emit reports after run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    db = Database()
    http = HttpClient(cfg.user_agent, cfg.rate_limit_per_host_per_sec)
    robots = RobotsPolicy(cfg.user_agent)

    try:
        if args.mode == "scrape":
            if args.all:
                seeds = list(cfg.start_urls)
                # Try sitemap seeds for each start URL
                for su in cfg.start_urls:
                    for s in fetch_sitemap_seeds(su):
                        seeds.append(s)
                for u in seeds:
                    db.upsert_url(u, status="queued", notes="seed")
                logger.info(f"queued seeds",)
                frontier = UrlFrontier(db, allowed_patterns=cfg.allowed_patterns, deny_patterns=cfg.deny_patterns)
                frontier.seed(seeds)
                pages_seen = 0
                while frontier.has_next():
                    if cfg.max_pages and pages_seen >= cfg.max_pages:
                        break
                    url = frontier.next_url()
                    if not url:
                        break
                    chk = robots.check(url)
                    if not chk.allowed:
                        continue
                    if chk.crawl_delay:
                        # robots crawl-delay is per-request hint; HttpClient also rate-limits per host
                        import time
                        time.sleep(float(chk.crawl_delay))
                    resp = http.get(url)
                    page = parse_html(url, resp.text, content_type=resp.content_type, status_code=resp.status_code)
                    changed = upsert_page(db, url, page)
                    db.mark_scraped(url)
                    frontier.discover_links(url, page.links)
                    pages_seen += 1
            elif args.filter_pattern:
                frontier = UrlFrontier(
                    db,
                    allowed_patterns=cfg.allowed_patterns,
                    deny_patterns=cfg.deny_patterns,
                    filter_substring=args.filter_pattern,
                )
                frontier.seed(cfg.start_urls)
                pages_seen = 0
                while frontier.has_next():
                    if cfg.max_pages and pages_seen >= cfg.max_pages:
                        break
                    url = frontier.next_url()
                    if not url:
                        break
                    chk = robots.check(url)
                    if not chk.allowed:
                        continue
                    if chk.crawl_delay:
                        import time
                        time.sleep(float(chk.crawl_delay))
                    resp = http.get(url)
                    page = parse_html(url, resp.text, content_type=resp.content_type, status_code=resp.status_code)
                    changed = upsert_page(db, url, page)
                    db.mark_scraped(url)
                    frontier.discover_links(url, page.links)
                    pages_seen += 1
            elif args.single_url:
                db.upsert_url(args.single_url, status="queued", notes="single-url")
                frontier = UrlFrontier(db, allowed_patterns=cfg.allowed_patterns, deny_patterns=cfg.deny_patterns)
                frontier.enqueue(args.single_url)
                while frontier.has_next():
                    url = frontier.next_url()
                    if not url:
                        break
                    chk = robots.check(url)
                    if not chk.allowed:
                        continue
                    if chk.crawl_delay:
                        import time
                        time.sleep(float(chk.crawl_delay))
                    resp = http.get(url)
                    page = parse_html(url, resp.text, content_type=resp.content_type, status_code=resp.status_code)
                    changed = upsert_page(db, url, page)
                    db.mark_scraped(url)
                    frontier.discover_links(url, page.links)
        elif args.mode == "update":
            if args.all:
                for url in db.query_urls_by_filter(None):
                    resp = http.get(url)
                    page = parse_html(url, resp.text, content_type=resp.content_type, status_code=resp.status_code)
                    _ = upsert_page(db, url, page)
                    db.mark_scraped(url)
            elif args.filter_pattern:
                for url in db.query_urls_by_filter(args.filter_pattern):
                    resp = http.get(url)
                    page = parse_html(url, resp.text, content_type=resp.content_type, status_code=resp.status_code)
                    _ = upsert_page(db, url, page)
                    db.mark_scraped(url)
            elif args.single_url:
                resp = http.get(args.single_url)
                page = parse_html(args.single_url, resp.text, content_type=resp.content_type, status_code=resp.status_code)
                _ = upsert_page(db, args.single_url, page)
                db.mark_scraped(args.single_url)

        if args.report == "url-index":
            db.export_url_index(ExportPaths())
            logger.info("exported url index")
        elif args.report == "pages-csv":
            export_pages_csv(db)
            logger.info("exported pages csv")
        elif args.report == "pages-full-csv":
            export_pages_full_csv(db)
            logger.info("exported pages full csv")
    finally:
        try:
            http.close()
        finally:
            db.close()


if __name__ == "__main__":
    main()
