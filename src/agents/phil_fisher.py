from graph.state import AgentState, show_agent_reasoning
from tools.api import (
    get_financial_metrics,
    get_market_cap,
    search_line_items,
    get_insider_trades,
    get_company_news,
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


class PhilFisherSignal(BaseModel):
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


def phil_fisher_agent(state: AgentState):
    """Analyze stocks using Phil Fisher's growth-focused investment principles."""
    data = state["data"]
    tickers = data["tickers"]
    end_date = data["end_date"]
    start_date = data["start_date"]
    metadata = state["metadata"]
    
    analysis_data = {}
    fisher_analysis = {}
    
    for ticker in tickers:
        # Fetch all required data
        progress.update_status("phil_fisher_agent", ticker, "Gathering financial data")
        metrics = get_financial_metrics(ticker, end_date, period="annual", limit=5)
        financials = search_line_items(
            ticker,
            ["revenue", "net_income", "earnings_per_share", "free_cash_flow",
            "research_and_development", "operating_income", "operating_margin",
            "gross_margin", "total_debt", "shareholders_equity",
            "cash_and_equivalents", "ebit", "ebitda"],
            end_date, period="annual", limit=5
        )
        market_cap = get_market_cap(ticker, end_date)
        insider_trades = get_insider_trades(ticker, end_date, start_date=None, limit=50)
        company_news = get_company_news(ticker, end_date, start_date=None, limit=50)
        
        # Run analysis functions
        growth = analyze_growth_quality(financials)
        margins = analyze_margins_stability(financials)
        management = analyze_management_efficiency(financials)
        valuation = analyze_fisher_valuation(financials, market_cap)
        insider = analyze_insider_activity(insider_trades)
        sentiment = analyze_sentiment(company_news)
        
        # Calculate weighted score
        total_score = (
            growth.score * 0.30 +
            margins.score * 0.25 +
            management.score * 0.20 +
            valuation.score * 0.15 +
            insider.score * 0.05 +
            sentiment.score * 0.05
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
            "growth_quality": {"score": growth.score, "details": growth.details},
            "margins_stability": {"score": margins.score, "details": margins.details},
            "management_efficiency": {"score": management.score, "details": management.details},
            "valuation_analysis": {"score": valuation.score, "details": valuation.details},
            "insider_activity": {"score": insider.score, "details": insider.details},
            "sentiment_analysis": {"score": sentiment.score, "details": sentiment.details}
        }
        
        # Generate final recommendation
        progress.update_status("phil_fisher_agent", ticker, "Creating investment recommendation")
        recommendation = generate_fisher_output(
            ticker=ticker,
            analysis_data=analysis_data,
            model_name=metadata["model_name"],
            model_provider=metadata["model_provider"],
        )
        
        fisher_analysis[ticker] = {
            "signal": recommendation.signal,
            "confidence": recommendation.confidence,
            "reasoning": recommendation.reasoning
        }
        
        progress.update_status("phil_fisher_agent", ticker, "Analysis complete")
    
    # Create message and update state
    message = HumanMessage(content=json.dumps(fisher_analysis), name="phil_fisher_agent")
    
    if metadata.get("show_reasoning"):
        show_agent_reasoning(fisher_analysis, "Phil Fisher Agent")
    
    state["data"]["analyst_signals"]["phil_fisher_agent"] = fisher_analysis
    
    return {"messages": [message], "data": state["data"]}


def analyze_growth_quality(financials: list) -> AnalysisResult:
    """Evaluate growth quality using Fisher's criteria."""
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
            if growth_rate > 0.80:  # >80% growth
                score += 3
                details.append(f"Exceptional revenue growth: {growth_rate:.1%}")
            elif growth_rate > 0.40:  # >40% growth
                score += 2
                details.append(f"Strong revenue growth: {growth_rate:.1%}")
            elif growth_rate > 0.10:  # >10% growth
                score += 1
                details.append(f"Moderate revenue growth: {growth_rate:.1%}")
            else:
                details.append(f"Limited revenue growth: {growth_rate:.1%}")
    
    # 2. EPS Growth
    eps_values = [get_safe_value(item, 'earnings_per_share') for item in financials]
    eps_values = [e for e in eps_values if e is not None]
    
    if len(eps_values) >= 2:
        latest, oldest = eps_values[0], eps_values[-1]
        growth_rate = calc_growth(latest, oldest)
        
        if growth_rate:
            if growth_rate > 0.80:  # >80% growth
                score += 3
                details.append(f"Exceptional EPS growth: {growth_rate:.1%}")
            elif growth_rate > 0.40:  # >40% growth
                score += 2
                details.append(f"Strong EPS growth: {growth_rate:.1%}")
            elif growth_rate > 0.10:  # >10% growth
                score += 1
                details.append(f"Moderate EPS growth: {growth_rate:.1%}")
            else:
                details.append(f"Limited EPS growth: {growth_rate:.1%}")
    
    # 3. R&D Investment
    rnd = [get_safe_value(item, 'research_and_development') for item in financials]
    
    if revenues and rnd and revenues[0] > 0 and rnd[0] is not None:
        rnd_ratio = rnd[0] / revenues[0]
        
        if 0.03 <= rnd_ratio <= 0.15:  # 3-15% of revenue
            score += 3
            details.append(f"Ideal R&D investment: {rnd_ratio:.1%} of revenue")
        elif rnd_ratio > 0.15:  # >15% of revenue
            score += 2
            details.append(f"Heavy R&D investment: {rnd_ratio:.1%} of revenue")
        elif rnd_ratio > 0:  # Some R&D
            score += 1
            details.append(f"Modest R&D investment: {rnd_ratio:.1%} of revenue")
        else:
            details.append("No significant R&D investment")
    
    # Scale score (max raw score is 9)
    final_score = scale_score(score, 9)
    
    return AnalysisResult(final_score, "; ".join(details))


def analyze_margins_stability(financials: list) -> AnalysisResult:
    """Evaluate margin consistency and stability."""
    if not financials or len(financials) < 2:
        return AnalysisResult(0, "Insufficient data for margins analysis")
    
    score = 0
    details = []
    
    # 1. Operating Margin Trend
    op_margins = [get_safe_value(item, 'operating_margin') for item in financials]
    op_margins = [m for m in op_margins if m is not None]
    
    if len(op_margins) >= 2:
        oldest, newest = op_margins[-1], op_margins[0]
        
        if newest >= oldest > 0:  # Stable or improving
            score += 2
            details.append(f"Improving margins: {oldest:.1%} → {newest:.1%}")
        elif newest > 0:  # Positive but declining
            score += 1
            details.append(f"Positive but declining margins: {oldest:.1%} → {newest:.1%}")
        else:
            details.append("Negative or severely declining margins")
    
    # 2. Gross Margin Level
    gm_values = [get_safe_value(item, 'gross_margin') for item in financials]
    gm_values = [m for m in gm_values if m is not None]
    
    if gm_values:
        recent_gm = gm_values[0]
        
        if recent_gm > 0.5:  # >50% gross margin
            score += 2
            details.append(f"Excellent gross margin: {recent_gm:.1%}")
        elif recent_gm > 0.3:  # >30% gross margin
            score += 1
            details.append(f"Good gross margin: {recent_gm:.1%}")
        else:
            details.append(f"Low gross margin: {recent_gm:.1%}")
    
    # 3. Margin Consistency
    if len(op_margins) >= 3:
        stdev = statistics.pstdev(op_margins)
        
        if stdev < 0.02:  # Very stable
            score += 2
            details.append(f"Highly stable margins (σ={stdev:.3f})")
        elif stdev < 0.05:  # Moderately stable
            score += 1
            details.append(f"Reasonably stable margins (σ={stdev:.3f})")
        else:
            details.append(f"Volatile margins (σ={stdev:.3f})")
    
    # Scale score (max raw score is 6)
    final_score = scale_score(score, 6)
    
    return AnalysisResult(final_score, "; ".join(details))


def analyze_management_efficiency(financials: list) -> AnalysisResult:
    """Evaluate management efficiency and financial leverage."""
    if not financials:
        return AnalysisResult(0, "Insufficient data for management analysis")
    
    score = 0
    details = []
    
    # 1. Return on Equity
    ni_values = [get_safe_value(item, 'net_income') for item in financials]
    eq_values = [get_safe_value(item, 'shareholders_equity') for item in financials]
    
    if ni_values and eq_values and ni_values[0] is not None and eq_values[0] is not None and eq_values[0] > 0:
        roe = ni_values[0] / eq_values[0]
        
        if roe > 0.2:  # >20% ROE
            score += 3
            details.append(f"Excellent ROE: {roe:.1%}")
        elif roe > 0.1:  # >10% ROE
            score += 2
            details.append(f"Good ROE: {roe:.1%}")
        elif roe > 0:  # Positive ROE
            score += 1
            details.append(f"Positive ROE: {roe:.1%}")
        else:
            details.append(f"Negative ROE: {roe:.1%}")
    
    # 2. Debt-to-Equity
    debt_values = [get_safe_value(item, 'total_debt') for item in financials]
    
    if debt_values and eq_values and debt_values[0] is not None and eq_values[0] is not None and eq_values[0] > 0:
        dte = debt_values[0] / eq_values[0]
        
        if dte < 0.3:  # Very low debt
            score += 2
            details.append(f"Low debt-to-equity: {dte:.2f}")
        elif dte < 1.0:  # Moderate debt
            score += 1
            details.append(f"Manageable debt-to-equity: {dte:.2f}")
        else:
            details.append(f"High debt-to-equity: {dte:.2f}")
    
    # 3. Free Cash Flow Consistency
    fcf_values = [get_safe_value(item, 'free_cash_flow') for item in financials]
    fcf_values = [f for f in fcf_values if f is not None]
    
    if fcf_values and len(fcf_values) >= 2:
        positive_count = sum(1 for f in fcf_values if f > 0)
        positive_ratio = positive_count / len(fcf_values)
        
        if positive_ratio > 0.8:  # >80% positive periods
            score += 1
            details.append(f"Consistent FCF: {positive_count}/{len(fcf_values)} positive periods")
        else:
            details.append(f"Inconsistent FCF: {positive_count}/{len(fcf_values)} positive periods")
    
    # Scale score (max raw score is 6)
    final_score = scale_score(score, 6)
    
    return AnalysisResult(final_score, "; ".join(details))


def analyze_fisher_valuation(financials: list, market_cap: float) -> AnalysisResult:
    """Evaluate valuation using Fisher's approach."""
    if not financials or not market_cap or market_cap <= 0:
        return AnalysisResult(0, "Insufficient data for valuation analysis")
    
    score = 0
    details = []
    
    # 1. Price-to-Earnings
    ni_values = [get_safe_value(item, 'net_income') for item in financials]
    
    if ni_values and ni_values[0] is not None and ni_values[0] > 0:
        pe = market_cap / ni_values[0]
        
        if pe < 20:  # Low P/E
            score += 2
            details.append(f"Attractive P/E: {pe:.1f}")
        elif pe < 30:  # Moderate P/E
            score += 1
            details.append(f"Reasonable P/E: {pe:.1f}")
        else:
            details.append(f"High P/E: {pe:.1f}")
    
    # 2. Price-to-FCF
    fcf_values = [get_safe_value(item, 'free_cash_flow') for item in financials]
    
    if fcf_values and fcf_values[0] is not None and fcf_values[0] > 0:
        pfcf = market_cap / fcf_values[0]
        
        if pfcf < 20:  # Low P/FCF
            score += 2
            details.append(f"Attractive P/FCF: {pfcf:.1f}")
        elif pfcf < 30:  # Moderate P/FCF
            score += 1
            details.append(f"Reasonable P/FCF: {pfcf:.1f}")
        else:
            details.append(f"High P/FCF: {pfcf:.1f}")
    
    # Scale score (max raw score is 4)
    final_score = scale_score(score, 4)
    
    return AnalysisResult(final_score, "; ".join(details))


def analyze_insider_activity(insider_trades: list) -> AnalysisResult:
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
        details.append(f"Strong insider buying: {buys}/{total} transactions")
    elif buy_ratio > 0.4:  # Balanced activity
        score = 6
        details.append(f"Moderate insider buying: {buys}/{total} transactions")
    else:  # Mostly selling
        score = 4
        details.append(f"Significant insider selling: {sells}/{total} transactions")
    
    return AnalysisResult(score, "; ".join(details))


def analyze_sentiment(news_items: list) -> AnalysisResult:
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
    
    negative_ratio = negative_count / len(news_items)
    
    if negative_ratio > 0.3:  # >30% negative news
        score = 3
        details = f"High negative sentiment: {negative_count}/{len(news_items)} negative headlines"
    elif negative_count > 0:  # Some negative news
        score = 6
        details = f"Mixed sentiment: {negative_count}/{len(news_items)} negative headlines"
    else:  # No negative news
        score = 8
        details = "Positive sentiment: no negative headlines"
    
    return AnalysisResult(score, details)


def generate_fisher_output(
    ticker: str,
    analysis_data: dict,
    model_name: str,
    model_provider: str,
) -> PhilFisherSignal:
    """Generate investment recommendation using Phil Fisher's principles."""
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a Phil Fisher AI agent evaluating investments using these principles:
            
            1. Focus on long-term growth potential and quality management
            2. Value companies investing heavily in R&D for future products
            3. Look for consistent profitability and strong margins
            4. Willing to pay premium for quality but still value-conscious
            5. Focus on thorough research and fundamental analysis
            
            In your evaluation:
            - Emphasize specific growth metrics and trends
            - Analyze management quality and capital allocation
            - Assess competitive advantages that can sustain growth
            - Use Fisher's methodical, growth-focused, long-term voice
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
        return PhilFisherSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Error in analysis, defaulting to neutral stance."
        )

    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=PhilFisherSignal,
        agent_name="phil_fisher_agent",
        default_factory=default_signal,
    )