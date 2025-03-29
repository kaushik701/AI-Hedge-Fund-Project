from langchain_core.messages import HumanMessage
from graph.state import AgentState, show_agent_reasoning
from util.progress import progress
from tools.api import get_insider_trades, get_company_news
import pandas as pd
import numpy as np
import json


def sentiment_agent(state: AgentState):
    """Analyzes market sentiment from insider trades and news."""
    data = state["data"]
    tickers = data["tickers"]
    end_date = data["end_date"]
    
    # Initialize results dictionary
    sentiment_analysis = {}
    
    for ticker in tickers:
        # Get insider trading data
        progress.update_status("sentiment_agent", ticker, "Fetching insider trades")
        insider_trades = get_insider_trades(ticker, end_date, limit=1000)
        
        # Get news data
        progress.update_status("sentiment_agent", ticker, "Fetching company news")
        company_news = get_company_news(ticker, end_date, limit=100)
        
        # Analyze both data sources
        progress.update_status("sentiment_agent", ticker, "Analyzing sentiment")
        insider_signal = analyze_insider_trades(insider_trades)
        news_signal = analyze_news_sentiment(company_news)
        
        # Combine signals with weights
        progress.update_status("sentiment_agent", ticker, "Combining signals")
        signal, confidence, reasoning = combine_signals(insider_signal, news_signal)
        
        # Store results
        sentiment_analysis[ticker] = {
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning
        }
        
        progress.update_status("sentiment_agent", ticker, "Done")
    
    # Create message with analysis results
    message = HumanMessage(content=json.dumps(sentiment_analysis), name="sentiment_agent")
    
    # Show reasoning if requested
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(sentiment_analysis, "Sentiment Analysis Agent")
    
    # Store analysis in state
    state["data"]["analyst_signals"]["sentiment_agent"] = sentiment_analysis
    
    return {
        "messages": [message],
        "data": data
    }


def analyze_insider_trades(insider_trades):
    """Analyze insider trading patterns."""
    if not insider_trades:
        return {"bullish": 0, "bearish": 0, "count": 0}
    
    # Extract transaction shares
    transaction_shares = [t.transaction_shares for t in insider_trades if hasattr(t, 'transaction_shares') and t.transaction_shares is not None]
    
    # Count bullish and bearish signals
    bullish = sum(1 for shares in transaction_shares if shares > 0)
    bearish = sum(1 for shares in transaction_shares if shares < 0)
    
    return {
        "bullish": bullish,
        "bearish": bearish,
        "count": len(transaction_shares)
    }


def analyze_news_sentiment(news_items):
    """Analyze sentiment from news articles."""
    if not news_items:
        return {"bullish": 0, "bearish": 0, "neutral": 0, "count": 0}
    
    # Extract sentiment
    sentiments = [n.sentiment for n in news_items if hasattr(n, 'sentiment') and n.sentiment is not None]
    
    # Count different sentiment types
    bullish = sum(1 for s in sentiments if s == "positive")
    bearish = sum(1 for s in sentiments if s == "negative")
    neutral = sum(1 for s in sentiments if s != "positive" and s != "negative")
    
    return {
        "bullish": bullish,
        "bearish": bearish,
        "neutral": neutral,
        "count": len(sentiments)
    }


def combine_signals(insider_signal, news_signal):
    """Combine insider and news signals with weights."""
    # Weights for each data source
    insider_weight = 0.3
    news_weight = 0.7
    
    # Calculate weighted signals
    weighted_bullish = (
        insider_signal["bullish"] * insider_weight + 
        news_signal["bullish"] * news_weight
    )
    
    weighted_bearish = (
        insider_signal["bearish"] * insider_weight + 
        news_signal["bearish"] * news_weight
    )
    
    # Determine overall signal
    if weighted_bullish > weighted_bearish:
        signal = "bullish"
        max_signal = weighted_bullish
    elif weighted_bearish > weighted_bullish:
        signal = "bearish"
        max_signal = weighted_bearish
    else:
        signal = "neutral"
        max_signal = weighted_bullish  # Both are equal in this case
    
    # Calculate confidence
    total_weighted_count = (
        insider_signal["count"] * insider_weight + 
        news_signal["count"] * news_weight
    )
    
    confidence = 0
    if total_weighted_count > 0:
        confidence = round((max_signal / total_weighted_count) * 100, 2)
    
    # Create reasoning string
    reasoning = f"Weighted Bullish signals: {weighted_bullish:.1f}, Weighted Bearish signals: {weighted_bearish:.1f}"
    
    return signal, confidence, reasoning