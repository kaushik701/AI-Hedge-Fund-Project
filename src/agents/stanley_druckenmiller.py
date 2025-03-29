from graph.state import AgentState, show_agent_reasoning
from tools.api import (
    get_financial_metrics,
    get_market_cap,
    search_line_items,
    get_insider_trades,
    get_company_news,
    get_prices,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from dataclasses import dataclass, field
import json
from typing_extensions import Literal
from typing import Dict, List, Any, Optional
from util.progress import progress
from util.llm import call_llm
import statistics


class StanleyDruckenmillerSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


@dataclass
class AnalysisResult:
    score: float
    details: str
    extra_data: Dict = field(default_factory=dict)


def get_safe_value(item, attr, default=None):
    """Safely get attribute value with fallback."""
    if hasattr(item, attr) and getattr(item, attr) is not None:
        return getattr(item, attr)
    return default


def calc_growth(current, previous):
    """Calculate growth rate between two values."""
    if not previous or abs(previous) < 1e-9:
        return None
    return (current - previous) / abs(previous)


def scale_score(raw_score, max_raw, max_final=10):
    """Scale a raw score to a final score."""
    return min(max_final, (raw_score / max_raw) * max_final)


def stanley_druckenmiller_agent(state):
    """Analyze stocks using Stanley Druckenmiller's momentum-focused approach."""
    data = state["data"]
    tickers = data["tickers"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    metadata = state["metadata"]
    
    analysis_data = {}
    druck_analysis = {}
    
    for ticker in tickers:
        # Fetch all required data
        progress.update_status("stanley_druckenmiller_agent", ticker, "Gathering financial data")
        metrics = get_financial_metrics(ticker, end_date, period="annual", limit=5)
        financials = search_line_items(
            ticker,
            ["revenue", "earnings_per_share", "net_income", "operating_income", 
            "gross_margin", "operating_margin", "free_cash_flow", "capital_expenditure",
            "cash_and_equivalents", "total_debt", "shareholders_equity",
            "outstanding_shares", "ebit", "ebitda"],
            end_date, period="annual", limit=5
        )
        market_cap = get_market_cap(ticker, end_date)
        insider_trades = get_insider_trades(ticker, end_date, start_date=None, limit=50)
        company_news = get_company_news(ticker, end_date, start_date=None, limit=50)
        prices = get_prices(ticker, start_date=start_date, end_date=end_date)
        
        # Run analysis functions
        growth = analyze_growth_momentum(financials, prices)
        sentiment = analyze_sentiment(company_news)
        insider = analyze_insider_activity(insider_trades)
        risk_reward = analyze_risk_reward(financials, market_cap, prices)
        valuation = analyze_valuation(financials, market_cap)
        
        # Calculate weighted score
        total_score = (
            growth.score * 0.35 +
            risk_reward.score * 0.20 +
            valuation.score * 0.20 +
            sentiment.score * 0.15 +
            insider.score * 0.10
        )
        
        # Generate signal
        if total_score >= 7.5:
            signal = "bullish"
        elif total_score <= 4.5:
            signal = "bearish"
        else:
            signal = "neutral"
        
        # Store analysis data
        analysis_data[ticker] = {
            "signal": signal,
            "score": total_score,
            "max_score": 10,
            "growth_momentum_analysis": {"score": growth.score, "details": growth.details},
            "sentiment_analysis": {"score": sentiment.score, "details": sentiment.details},
            "insider_activity": {"score": insider.score, "details": insider.details},
            "risk_reward_analysis": {"score": risk_reward.score, "details": risk_reward.details},
            "valuation_analysis": {"score": valuation.score, "details": valuation.details}
        }
        
        # Generate final recommendation
        progress.update_status("stanley_druckenmiller_agent", ticker, "Creating investment recommendation")
        recommendation = generate_druckenmiller_output(
            ticker=ticker,
            analysis_data=analysis_data,
            model_name=metadata["model_name"],
            model_provider=metadata["model_provider"],
        )
        
        druck_analysis[ticker] = {
            "signal": recommendation.signal,
            "confidence": recommendation.confidence,
            "reasoning": recommendation.reasoning
        }
        
        progress.update_status("stanley_druckenmiller_agent", ticker, "Analysis complete")
    
    # Create message and update state
    message = HumanMessage(content=json.dumps(druck_analysis), name="stanley_druckenmiller_agent")
    
    if metadata.get("show_reasoning"):
        show_agent_reasoning(druck_analysis, "Stanley Druckenmiller Agent")
    
    state["data"]["analyst_signals"]["stanley_druckenmiller_agent"] = druck_analysis
    
    return {"messages": [message], "data": state["data"]}


def analyze_growth_momentum(financials, prices):
    """Evaluate growth metrics and price momentum."""
    if not financials or len(financials) < 2:
        return AnalysisResult(0, "Insufficient data for growth analysis")
    
    score = 0
    details = []
    
    # 1. Revenue Growth
    revenues = [get_safe_value(item, 'revenue') for item in financials]
    revenues = [r for r in revenues if r is not None]
    
    if len(revenues) >= 2:
        latest, oldest = revenues[0], revenues[-1]
        growth_rate = calc_growth(latest, oldest)
        
        if growth_rate:
            if growth_rate > 0.30:  # >30% growth
                score += 3
                details.append(f"Strong revenue growth: {growth_rate:.1%}")
            elif growth_rate > 0.15:  # >15% growth
                score += 2
                details.append(f"Moderate revenue growth: {growth_rate:.1%}")
            elif growth_rate > 0.05:  # >5% growth
                score += 1
                details.append(f"Slight revenue growth: {growth_rate:.1%}")
            else:
                details.append(f"Minimal/negative revenue growth: {growth_rate:.1%}")
    
    # 2. EPS Growth
    eps_values = [get_safe_value(item, 'earnings_per_share') for item in financials]
    eps_values = [e for e in eps_values if e is not None]
    
    if len(eps_values) >= 2:
        latest, oldest = eps_values[0], eps_values[-1]
        growth_rate = calc_growth(latest, oldest)
        
        if growth_rate:
            if growth_rate > 0.30:  # >30% growth
                score += 3
                details.append(f"Strong EPS growth: {growth_rate:.1%}")
            elif growth_rate > 0.15:  # >15% growth
                score += 2
                details.append(f"Moderate EPS growth: {growth_rate:.1%}")
            elif growth_rate > 0.05:  # >5% growth
                score += 1
                details.append(f"Slight EPS growth: {growth_rate:.1%}")
            else:
                details.append(f"Minimal/negative EPS growth: {growth_rate:.1%}")
    
    # 3. Price Momentum
    if prices and len(prices) > 30:
        sorted_prices = sorted(prices, key=lambda p: p.time)
        close_prices = [p.close for p in sorted_prices if p.close is not None]
        
        if len(close_prices) >= 2:
            start_price = close_prices[0]
            end_price = close_prices[-1]
            
            if start_price > 0:
                pct_change = (end_price - start_price) / start_price
                
                if pct_change > 0.50:  # >50% price increase
                    score += 3
                    details.append(f"Very strong price momentum: {pct_change:.1%}")
                elif pct_change > 0.20:  # >20% price increase
                    score += 2
                    details.append(f"Moderate price momentum: {pct_change:.1%}")
                elif pct_change > 0:  # Positive price movement
                    score += 1
                    details.append(f"Slight positive momentum: {pct_change:.1%}")
                else:
                    details.append(f"Negative price momentum: {pct_change:.1%}")
    
    # Scale score (max raw score is 9)
    final_score = scale_score(score, 9)
    
    return AnalysisResult(final_score, "; ".join(details))


def analyze_insider_activity(insider_trades):
    """Analyze insider buying and selling patterns."""
    # Default neutral score
    score = 5
    details = []
    
    if not insider_trades:
        return AnalysisResult(score, "No insider trading data available")
    
    buys = 0
    sells = 0
    
    for trade in insider_trades:
        shares = get_safe_value(trade, 'transaction_shares', 0)
        if shares > 0:
            buys += 1
        elif shares < 0:
            sells += 1
    
    total = buys + sells
    if total == 0:
        return AnalysisResult(score, "No meaningful insider transactions found")
    
    buy_ratio = buys / total
    
    if buy_ratio > 0.7:  # Strong buying
        score = 8
        details.append(f"Heavy insider buying: {buys} buys vs. {sells} sells")
    elif buy_ratio > 0.4:  # Balanced activity
        score = 6
        details.append(f"Moderate insider buying: {buys} buys vs. {sells} sells")
    else:  # Mostly selling
        score = 4
        details.append(f"Mostly insider selling: {buys} buys vs. {sells} sells")
    
    return AnalysisResult(score, "; ".join(details))


def analyze_sentiment(news_items):
    """Analyze news sentiment based on headlines."""
    # Default neutral score
    score = 5
    
    if not news_items:
        return AnalysisResult(score, "No news data available")
    
    negative_keywords = ["lawsuit", "fraud", "negative", "downturn", "decline", "investigation", "recall"]
    negative_count = 0
    
    for news in news_items:
        title = get_safe_value(news, 'title', '').lower()
        if any(keyword in title for keyword in negative_keywords):
            negative_count += 1
    
    negative_ratio = negative_count / len(news_items) if news_items else 0
    
    if negative_ratio > 0.3:  # >30% negative news
        score = 3
        details = f"High proportion of negative headlines: {negative_count}/{len(news_items)}"
    elif negative_count > 0:  # Some negative news
        score = 6
        details = f"Some negative headlines: {negative_count}/{len(news_items)}"
    else:  # No negative news
        score = 8
        details = "Mostly positive/neutral headlines"
    
    return AnalysisResult(score, details)


def analyze_risk_reward(financials, market_cap, prices):
    """Evaluate risk-reward profile based on debt and volatility."""
    if not financials or not prices:
        return AnalysisResult(0, "Insufficient data for risk-reward analysis")
    
    score = 0
    details = []
    
    # 1. Debt-to-Equity Analysis
    debt_values = [get_safe_value(item, 'total_debt') for item in financials]
    equity_values = [get_safe_value(item, 'shareholders_equity') for item in financials]
    
    if debt_values and equity_values and debt_values[0] is not None and equity_values[0] is not None and equity_values[0] > 0:
        de_ratio = debt_values[0] / equity_values[0]
        
        if de_ratio < 0.3:  # Very low debt
            score += 3
            details.append(f"Low debt-to-equity: {de_ratio:.2f}")
        elif de_ratio < 0.7:  # Moderate debt
            score += 2
            details.append(f"Moderate debt-to-equity: {de_ratio:.2f}")
        elif de_ratio < 1.5:  # Higher but manageable debt
            score += 1
            details.append(f"Somewhat high debt-to-equity: {de_ratio:.2f}")
        else:
            details.append(f"High debt-to-equity: {de_ratio:.2f}")
    
    # 2. Price Volatility Analysis
    if len(prices) > 10:
        sorted_prices = sorted(prices, key=lambda p: p.time)
        close_prices = [p.close for p in sorted_prices if p.close is not None]
        
        if len(close_prices) > 10:
            # Calculate daily returns
            daily_returns = []
            for i in range(1, len(close_prices)):
                prev_close = close_prices[i - 1]
                if prev_close > 0:
                    daily_returns.append((close_prices[i] - prev_close) / prev_close)
            
            if daily_returns:
                stdev = statistics.pstdev(daily_returns)
                
                if stdev < 0.01:  # Very low volatility
                    score += 3
                    details.append(f"Low volatility: daily returns stdev {stdev:.2%}")
                elif stdev < 0.02:  # Moderate volatility
                    score += 2
                    details.append(f"Moderate volatility: daily returns stdev {stdev:.2%}")
                elif stdev < 0.04:  # Higher volatility
                    score += 1
                    details.append(f"High volatility: daily returns stdev {stdev:.2%}")
                else:
                    details.append(f"Very high volatility: daily returns stdev {stdev:.2%}")
    
    # Scale score (max raw score is 6)
    final_score = scale_score(score, 6)
    
    return AnalysisResult(final_score, "; ".join(details))


def analyze_valuation(financials, market_cap):
    """Evaluate valuation metrics with Druckenmiller's approach."""
    if not financials or not market_cap or market_cap <= 0:
        return AnalysisResult(0, "Insufficient data for valuation analysis")
    
    score = 0
    details = []
    
    # Get relevant financial metrics
    net_income = get_safe_value(financials[0], 'net_income') if financials else None
    fcf = get_safe_value(financials[0], 'free_cash_flow') if financials else None
    ebit = get_safe_value(financials[0], 'ebit') if financials else None
    ebitda = get_safe_value(financials[0], 'ebitda') if financials else None
    
    # Get debt and cash for enterprise value calculation
    debt = get_safe_value(financials[0], 'total_debt', 0) if financials else 0
    cash = get_safe_value(financials[0], 'cash_and_equivalents', 0) if financials else 0
    
    # Calculate enterprise value
    enterprise_value = market_cap + debt - cash
    
    # 1. P/E Ratio
    if net_income and net_income > 0:
        pe = market_cap / net_income
        
        if pe < 15:  # Low P/E
            score += 2
            details.append(f"Attractive P/E: {pe:.2f}")
        elif pe < 25:  # Moderate P/E
            score += 1
            details.append(f"Fair P/E: {pe:.2f}")
        else:
            details.append(f"High P/E: {pe:.2f}")
    
    # 2. P/FCF Ratio
    if fcf and fcf > 0:
        pfcf = market_cap / fcf
        
        if pfcf < 15:  # Low P/FCF
            score += 2
            details.append(f"Attractive P/FCF: {pfcf:.2f}")
        elif pfcf < 25:  # Moderate P/FCF
            score += 1
            details.append(f"Fair P/FCF: {pfcf:.2f}")
        else:
            details.append(f"High P/FCF: {pfcf:.2f}")
    
    # 3. EV/EBIT Ratio
    if enterprise_value > 0 and ebit and ebit > 0:
        ev_ebit = enterprise_value / ebit
        
        if ev_ebit < 15:  # Low EV/EBIT
            score += 2
            details.append(f"Attractive EV/EBIT: {ev_ebit:.2f}")
        elif ev_ebit < 25:  # Moderate EV/EBIT
            score += 1
            details.append(f"Fair EV/EBIT: {ev_ebit:.2f}")
        else:
            details.append(f"High EV/EBIT: {ev_ebit:.2f}")
    
    # 4. EV/EBITDA Ratio
    if enterprise_value > 0 and ebitda and ebitda > 0:
        ev_ebitda = enterprise_value / ebitda
        
        if ev_ebitda < 10:  # Low EV/EBITDA
            score += 2
            details.append(f"Attractive EV/EBITDA: {ev_ebitda:.2f}")
        elif ev_ebitda < 18:  # Moderate EV/EBITDA
            score += 1
            details.append(f"Fair EV/EBITDA: {ev_ebitda:.2f}")
        else:
            details.append(f"High EV/EBITDA: {ev_ebitda:.2f}")
    
    # Scale score (max raw score is 8)
    final_score = scale_score(score, 8)
    
    return AnalysisResult(final_score, "; ".join(details))


def generate_druckenmiller_output(
    ticker: str,
    analysis_data: dict,
    model_name: str,
    model_provider: str,
) -> StanleyDruckenmillerSignal:
    """Generate investment recommendation using Druckenmiller's principles."""
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a Stanley Druckenmiller AI agent evaluating investments using these principles:
            
            1. Seek asymmetric risk-reward opportunities
            2. Focus on growth, momentum, and market sentiment
            3. Preserve capital by avoiding major drawdowns
            4. Pay for quality growth but remain valuation-conscious
            5. Be aggressive when conviction is high
            
            In your analysis:
            - Emphasize growth and momentum metrics
            - Evaluate risk-reward with specific data points
            - Consider sentiment and potential catalysts
            - Balance upside potential against downside risks
            - Use Druckenmiller's decisive, conviction-driven voice
            """
        ),
        (
            "human",
            """Based on this analysis for {ticker}:
            {analysis_data}

            Return exactly this JSON format:
            {{
            "signal": "bullish" or "bearish" or "neutral",
            "confidence": float (0-100),
            "reasoning": "string"
            }}
            """
        )
    ])

    prompt = template.invoke({
        "analysis_data": json.dumps(analysis_data, indent=2),
        "ticker": ticker
    })

    def default_signal():
        return StanleyDruckenmillerSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Error in analysis, defaulting to neutral stance."
        )

    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=StanleyDruckenmillerSignal,
        agent_name="stanley_druckenmiller_agent",
        default_factory=default_signal,
    )