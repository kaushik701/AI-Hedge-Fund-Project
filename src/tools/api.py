import os
import pandas as pd
import requests
from data.cache import get_cache
from data.models import (
    CompanyNews, CompanyNewsResponse, FinancialMetrics, FinancialMetricsResponse,
    Price, PriceResponse, LineItem, LineItemResponse,
    InsiderTrade, InsiderTradeResponse
)

# Get global cache
_cache = get_cache()


def get_prices(ticker, start_date, end_date):
    """Fetch price data from cache or API."""
    # Check cache first
    cached_data = _cache.get_prices(ticker)
    if cached_data:
        filtered_data = [Price(**price) for price in cached_data 
                        if start_date <= price["time"] <= end_date]
        if filtered_data:
            return filtered_data

    # Fetch from API
    headers = {}
    api_key = os.environ.get("FINANCIAL_DATASETS_API_KEY")
    if api_key:
        headers["X-API-KEY"] = api_key

    url = f"https://api.financialdatasets.ai/prices/?ticker={ticker}&interval=day&interval_multiplier=1&start_date={start_date}&end_date={end_date}"
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")

    price_response = PriceResponse(**response.json())
    prices = price_response.prices
    
    if prices:
        _cache.set_prices(ticker, [p.model_dump() for p in prices])
    
    return prices


def get_financial_metrics(ticker, end_date, period="ttm", limit=10):
    """Fetch financial metrics from cache or API."""
    # Check cache first
    cached_data = _cache.get_financial_metrics(ticker)
    if cached_data:
        filtered_data = [FinancialMetrics(**metric) for metric in cached_data 
                        if metric["report_period"] <= end_date]
        filtered_data.sort(key=lambda x: x.report_period, reverse=True)
        if filtered_data:
            return filtered_data[:limit]

    # Fetch from API
    headers = {}
    api_key = os.environ.get("FINANCIAL_DATASETS_API_KEY")
    if api_key:
        headers["X-API-KEY"] = api_key

    url = f"https://api.financialdatasets.ai/financial-metrics/?ticker={ticker}&report_period_lte={end_date}&limit={limit}&period={period}"
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")

    metrics_response = FinancialMetricsResponse(**response.json())
    financial_metrics = metrics_response.financial_metrics
    
    if financial_metrics:
        _cache.set_financial_metrics(ticker, [m.model_dump() for m in financial_metrics])
    
    return financial_metrics


def search_line_items(ticker, line_items, end_date, period="ttm", limit=10):
    """Fetch line items from API."""
    headers = {}
    api_key = os.environ.get("FINANCIAL_DATASETS_API_KEY")
    if api_key:
        headers["X-API-KEY"] = api_key

    url = "https://api.financialdatasets.ai/financials/search/line-items"
    body = {
        "tickers": [ticker],
        "line_items": line_items,
        "end_date": end_date,
        "period": period,
        "limit": limit,
    }
    
    response = requests.post(url, headers=headers, json=body)
    
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")
    
    response_model = LineItemResponse(**response.json())
    search_results = response_model.search_results
    
    return search_results[:limit] if search_results else []


def get_insider_trades(ticker, end_date, start_date=None, limit=1000):
    """Fetch insider trades from cache or API."""
    # Check cache first
    cached_data = _cache.get_insider_trades(ticker)
    if cached_data:
        filtered_data = [InsiderTrade(**trade) for trade in cached_data 
                        if (start_date is None or (trade.get("transaction_date") or trade["filing_date"]) >= start_date)
                        and (trade.get("transaction_date") or trade["filing_date"]) <= end_date]
        filtered_data.sort(key=lambda x: x.transaction_date or x.filing_date, reverse=True)
        if filtered_data:
            return filtered_data

    # Fetch from API
    headers = {}
    api_key = os.environ.get("FINANCIAL_DATASETS_API_KEY")
    if api_key:
        headers["X-API-KEY"] = api_key

    all_trades = []
    current_end_date = end_date
    
    while True:
        url = f"https://api.financialdatasets.ai/insider-trades/?ticker={ticker}&filing_date_lte={current_end_date}"
        if start_date:
            url += f"&filing_date_gte={start_date}"
        url += f"&limit={limit}"
        
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")
        
        insider_trades = InsiderTradeResponse(**response.json()).insider_trades
        
        if not insider_trades:
            break
            
        all_trades.extend(insider_trades)
        
        # Stop pagination if we don't need more data
        if not start_date or len(insider_trades) < limit:
            break
            
        # Update for next page
        current_end_date = min(trade.filing_date for trade in insider_trades).split('T')[0]
        if current_end_date <= start_date:
            break

    if all_trades:
        _cache.set_insider_trades(ticker, [trade.model_dump() for trade in all_trades])
    
    return all_trades


def get_company_news(ticker, end_date, start_date=None, limit=1000):
    """Fetch company news from cache or API."""
    # Check cache first
    cached_data = _cache.get_company_news(ticker)
    if cached_data:
        filtered_data = [CompanyNews(**news) for news in cached_data 
                        if (start_date is None or news["date"] >= start_date)
                        and news["date"] <= end_date]
        filtered_data.sort(key=lambda x: x.date, reverse=True)
        if filtered_data:
            return filtered_data

    # Fetch from API
    headers = {}
    api_key = os.environ.get("FINANCIAL_DATASETS_API_KEY")
    if api_key:
        headers["X-API-KEY"] = api_key

    all_news = []
    current_end_date = end_date
    
    while True:
        url = f"https://api.financialdatasets.ai/news/?ticker={ticker}&end_date={current_end_date}"
        if start_date:
            url += f"&start_date={start_date}"
        url += f"&limit={limit}"
        
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")
        
        company_news = CompanyNewsResponse(**response.json()).news
        
        if not company_news:
            break
            
        all_news.extend(company_news)
        
        # Stop pagination if we don't need more data
        if not start_date or len(company_news) < limit:
            break
            
        # Update for next page
        current_end_date = min(news.date for news in company_news).split('T')[0]
        if current_end_date <= start_date:
            break

    if all_news:
        _cache.set_company_news(ticker, [news.model_dump() for news in all_news])
    
    return all_news


def get_market_cap(ticker, end_date):
    """Get market cap for a ticker."""
    financial_metrics = get_financial_metrics(ticker, end_date)
    if not financial_metrics:
        return None
    return financial_metrics[0].market_cap


def prices_to_df(prices):
    """Convert price list to DataFrame."""
    if not prices:
        return pd.DataFrame()
        
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    
    # Convert numeric columns
    for col in ["open", "close", "high", "low", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
    df.sort_index(inplace=True)
    return df


def get_price_data(ticker, start_date, end_date):
    """Get price data as DataFrame."""
    prices = get_prices(ticker, start_date, end_date)
    return prices_to_df(prices)