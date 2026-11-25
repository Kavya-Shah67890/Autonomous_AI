"""Playwright-based Google Search and Webpage Exploration tools."""

from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import quote_plus, urlparse

from bs4 import BeautifulSoup
from playwright.sync_api import Browser, Playwright, sync_playwright

from src.permissions import PermissionManager


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


@dataclass
class PageContent:
    url: str
    title: str
    text: str
    links: List[str]
    error: Optional[str] = None


class WebBrowserTool:
    """Manages Playwright browser sessions for searching and reading webpages."""

    def __init__(
        self,
        headless: bool = True,
        permission_manager: Optional[PermissionManager] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.headless = headless
        self.permission_manager = permission_manager
        self.logger = logger or logging.getLogger("research_system.browser")
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None

    def __enter__(self) -> "WebBrowserTool":
        self._ensure_browser()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()

    def _ensure_browser(self) -> Browser:
        if self._browser is None:
            self._playwright = sync_playwright().start()
            try:
                self._browser = self._playwright.chromium.launch(headless=self.headless)
            except Exception as exc:
                if "Executable doesn't exist" in str(exc):
                    self.logger.info("Playwright Chromium not found. Installing automatically...")
                    subprocess.run(
                        [sys.executable, "-m", "playwright", "install", "chromium"],
                        check=True,
                    )
                    self._browser = self._playwright.chromium.launch(headless=self.headless)
                else:
                    raise
        return self._browser

    def search_google(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Search Google (with DuckDuckGo fallback if Google blocks) using Playwright."""
        self.logger.info("Searching web for: %s", query)
        results: List[SearchResult] = []
        browser = self._ensure_browser()
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
            )
        )
        page = context.new_page()
        try:
            google_url = f"https://www.google.com/search?q={quote_plus(query)}&hl=en"
            page.goto(google_url, timeout=15000, wait_until="domcontentloaded")
            page.wait_for_timeout(1200)
            html = page.content()
            results = self._parse_google_html(html, max_results)

            if not results:
                self.logger.info("Google returned 0 parsed results; falling back to DuckDuckGo HTML.")
                ddg_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
                page.goto(ddg_url, timeout=15000, wait_until="domcontentloaded")
                results = self._parse_ddg_html(page.content(), max_results)
        except Exception as exc:
            self.logger.error("Web search failed for '%s': %s", query, exc)
        finally:
            context.close()

        self.logger.info("Found %d search results for '%s'", len(results), query)
        return results

    @staticmethod
    def _parse_google_html(html: str, max_results: int) -> List[SearchResult]:
        soup = BeautifulSoup(html, "html.parser")
        results: List[SearchResult] = []
        for g in soup.select("div.g"):
            link_el = g.select_one("a[href]")
            title_el = g.select_one("h3")
            if not link_el or not title_el:
                continue
            href = link_el.get("href", "")
            if not href.startswith("http"):
                continue
            snippet_el = g.select_one("div.VwiC3b, span.aCOpRe")
            snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""
            results.append(SearchResult(title=title_el.get_text(strip=True), url=href, snippet=snippet))
            if len(results) >= max_results:
                break
        return results

    @staticmethod
    def _parse_ddg_html(html: str, max_results: int) -> List[SearchResult]:
        soup = BeautifulSoup(html, "html.parser")
        results: List[SearchResult] = []
        for item in soup.select("div.result"):
            link_el = item.select_one("a.result__a")
            if not link_el:
                continue
            href = link_el.get("href", "")
            if not href.startswith("http"):
                continue
            snippet_el = item.select_one("a.result__snippet")
            snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""
            results.append(SearchResult(title=link_el.get_text(strip=True), url=href, snippet=snippet))
            if len(results) >= max_results:
                break
        return results

    def fetch_page(self, url: str, max_chars: int = 8000) -> PageContent:
        """Navigate to a webpage, check permission if needed, and extract clean text and links."""
        domain = urlparse(url).netloc
        if self.permission_manager is not None:
            allowed = self.permission_manager.request_permission(
                action="visit_webpage",
                target=url,
                reason=f"Investigation Agent wants to read content from external domain '{domain}'.",
            )
            if not allowed:
                return PageContent(
                    url=url,
                    title="",
                    text="",
                    links=[],
                    error="Permission denied by user before visiting webpage.",
                )

        self.logger.info("Exploring webpage: %s", url)
        browser = self._ensure_browser()
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
            )
        )
        page = context.new_page()
        try:
            page.goto(url, timeout=20000, wait_until="domcontentloaded")
            page.wait_for_timeout(1000)
            title = page.title() or url
            html = page.content()
            soup = BeautifulSoup(html, "html.parser")

            for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
                tag.decompose()

            text = " ".join(soup.get_text(" ", strip=True).split())[:max_chars]
            links: List[str] = []
            for a in soup.select("a[href]"):
                href = a.get("href", "")
                if href.startswith("http") and href not in links:
                    links.append(href)
                if len(links) >= 15:
                    break

            return PageContent(url=url, title=title, text=text, links=links)
        except Exception as exc:
            self.logger.error("Failed to fetch %s: %s", url, exc)
            return PageContent(url=url, title="", text="", links=[], error=str(exc))
        finally:
            context.close()
