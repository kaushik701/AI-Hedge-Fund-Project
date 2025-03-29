class Cache:
    """In-memory cache for API responses."""
    
    def __init__(self):
        self._prices_cache = {}
        self._financial_metrics_cache = {}
        self._line_items_cache = {}
        self._insider_trades_cache = {}
        self._company_news_cache = {}
    
    def _merge_data(self, existing, new_data, key_field):
        """Merge existing and new data, avoiding duplicates based on a key field."""
        if not existing:
            return new_data
        
        existing_keys = {item[key_field] for item in existing}
        merged = existing.copy()
        merged.extend([item for item in new_data if item[key_field] not in existing_keys])
        return merged
    
    def get_prices(self, ticker):
        """Get cached price data if available."""
        return self._prices_cache.get(ticker)
    
    def set_prices(self, ticker, data):
        """Append new price data to cache."""
        self._prices_cache[ticker] = self._merge_data(self._prices_cache.get(ticker), data, key_field="time")
    
    def get_financial_metrics(self, ticker):
        """Get cached financial metrics if available."""
        return self._financial_metrics_cache.get(ticker)
    
    def set_financial_metrics(self, ticker, data):
        """Append new financial metrics to cache."""
        self._financial_metrics_cache[ticker] = self._merge_data(self._financial_metrics_cache.get(ticker), data, key_field="report_period")
    
    def get_line_items(self, ticker):
        """Get cached line items if available."""
        return self._line_items_cache.get(ticker)
    
    def set_line_items(self, ticker, data):
        """Append new line items to cache."""
        self._line_items_cache[ticker] = self._merge_data(self._line_items_cache.get(ticker), data, key_field="report_period")
    
    def get_insider_trades(self, ticker):
        """Get cached insider trades if available."""
        return self._insider_trades_cache.get(ticker)
    
    def set_insider_trades(self, ticker, data):
        """Append new insider trades to cache."""
        self._insider_trades_cache[ticker] = self._merge_data(self._insider_trades_cache.get(ticker), data, key_field="filing_date")
    
    def get_company_news(self, ticker):
        """Get cached company news if available."""
        return self._company_news_cache.get(ticker)
    
    def set_company_news(self, ticker, data):
        """Append new company news to cache."""
        self._company_news_cache[ticker] = self._merge_data(self._company_news_cache.get(ticker), data, key_field="date")


# Global cache instance
_cache = Cache()

def get_cache():
    """Get the global cache instance."""
    return _cache