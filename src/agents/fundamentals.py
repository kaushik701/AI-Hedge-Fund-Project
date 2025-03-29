from langchain_core.messages import HumanMessage
from graph.state import AgentState, show_agent_reasoning
from util.progress import progress
from tools.api import get_financial_metrics
import json


def fundamentals_agent(state):
    """Analyzes fundamental data for multiple stocks and generates trading signals."""
    data = state["data"]
    tickers = data["tickers"]
    end_date = data["end_date"]
    
    # Initialize results dictionary
    fundamental_analysis = {}
    
    for ticker in tickers:
        progress.update_status("fundamentals_agent", ticker, "Fetching financial metrics")
        
        # Get financial metrics
        metrics_list = get_financial_metrics(ticker, end_date, period="ttm", limit=10)
        
        if not metrics_list:
            progress.update_status("fundamentals_agent", ticker, "Failed: No data found")
            continue
        
        # Use most recent metrics
        metrics = metrics_list[0]
        
        # Analyze different fundamental aspects
        progress.update_status("fundamentals_agent", ticker, "Analyzing fundamentals")
        
        profitability = analyze_profitability(metrics)
        growth = analyze_growth(metrics)
        financial_health = analyze_financial_health(metrics)
        valuation = analyze_valuation(metrics)
        
        # Collect all signals
        signals = [
            profitability["signal"],
            growth["signal"],
            financial_health["signal"],
            valuation["signal"]
        ]
        
        # Calculate overall signal
        bullish_count = signals.count("bullish")
        bearish_count = signals.count("bearish")
        
        if bullish_count > bearish_count:
            overall_signal = "bullish"
        elif bearish_count > bullish_count:
            overall_signal = "bearish"
        else:
            overall_signal = "neutral"
        
        # Calculate confidence
        confidence = round(max(bullish_count, bearish_count) / len(signals), 2) * 100
        
        # Store analysis results
        fundamental_analysis[ticker] = {
            "signal": overall_signal,
            "confidence": confidence,
            "reasoning": {
                "profitability_signal": profitability,
                "growth_signal": growth,
                "financial_health_signal": financial_health,
                "price_ratios_signal": valuation
            }
        }
        
        progress.update_status("fundamentals_agent", ticker, "Done")
    
    # Create message with analysis results
    message = HumanMessage(content=json.dumps(fundamental_analysis), name="fundamentals_agent")
    
    # Show reasoning if requested
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(fundamental_analysis, "Fundamental Analysis Agent")
    
    # Store signals in state
    state["data"]["analyst_signals"]["fundamentals_agent"] = fundamental_analysis
    
    return {"messages": [message], "data": data}


def analyze_profitability(metrics):
    """Analyze company profitability metrics."""
    # Extract relevant metrics
    roe = getattr(metrics, "return_on_equity", None)
    net_margin = getattr(metrics, "net_margin", None)
    op_margin = getattr(metrics, "operating_margin", None)
    
    # Score based on thresholds
    score = 0
    if roe and roe > 0.15:  # ROE > 15%
        score += 1
    if net_margin and net_margin > 0.20:  # Net margin > 20%
        score += 1
    if op_margin and op_margin > 0.15:  # Operating margin > 15%
        score += 1
    
    # Determine signal based on score
    if score >= 2:
        signal = "bullish"
    elif score == 0:
        signal = "bearish"
    else:
        signal = "neutral"
    
    # Format details
    details = []
    if roe is not None:
        details.append(f"ROE: {roe:.2%}")
    else:
        details.append("ROE: N/A")
        
    if net_margin is not None:
        details.append(f"Net Margin: {net_margin:.2%}")
    else:
        details.append("Net Margin: N/A")
        
    if op_margin is not None:
        details.append(f"Op Margin: {op_margin:.2%}")
    else:
        details.append("Op Margin: N/A")
    
    return {
        "signal": signal,
        "details": ", ".join(details)
    }


def analyze_growth(metrics):
    """Analyze company growth metrics."""
    # Extract relevant metrics
    rev_growth = getattr(metrics, "revenue_growth", None)
    earnings_growth = getattr(metrics, "earnings_growth", None)
    book_growth = getattr(metrics, "book_value_growth", None)
    
    # Score based on thresholds
    score = 0
    if rev_growth and rev_growth > 0.10:  # Revenue growth > 10%
        score += 1
    if earnings_growth and earnings_growth > 0.10:  # Earnings growth > 10%
        score += 1
    if book_growth and book_growth > 0.10:  # Book value growth > 10%
        score += 1
    
    # Determine signal based on score
    if score >= 2:
        signal = "bullish"
    elif score == 0:
        signal = "bearish"
    else:
        signal = "neutral"
    
    # Format details
    details = []
    if rev_growth is not None:
        details.append(f"Revenue Growth: {rev_growth:.2%}")
    else:
        details.append("Revenue Growth: N/A")
        
    if earnings_growth is not None:
        details.append(f"Earnings Growth: {earnings_growth:.2%}")
    else:
        details.append("Earnings Growth: N/A")
        
    if book_growth is not None:
        details.append(f"Book Value Growth: {book_growth:.2%}")
    else:
        details.append("Book Value Growth: N/A")
    
    return {
        "signal": signal,
        "details": ", ".join(details)
    }


def analyze_financial_health(metrics):
    """Analyze company financial health metrics."""
    # Extract relevant metrics
    current_ratio = getattr(metrics, "current_ratio", None)
    debt_to_equity = getattr(metrics, "debt_to_equity", None)
    fcf_per_share = getattr(metrics, "free_cash_flow_per_share", None)
    eps = getattr(metrics, "earnings_per_share", None)
    
    # Score based on thresholds
    score = 0
    if current_ratio and current_ratio > 1.5:  # Current ratio > 1.5
        score += 1
    if debt_to_equity and debt_to_equity < 0.5:  # D/E < 0.5
        score += 1
    if fcf_per_share and eps and fcf_per_share > eps * 0.8:  # FCF > 80% of EPS
        score += 1
    
    # Determine signal based on score
    if score >= 2:
        signal = "bullish"
    elif score == 0:
        signal = "bearish"
    else:
        signal = "neutral"
    
    # Format details
    details = []
    if current_ratio is not None:
        details.append(f"Current Ratio: {current_ratio:.2f}")
    else:
        details.append("Current Ratio: N/A")
        
    if debt_to_equity is not None:
        details.append(f"D/E: {debt_to_equity:.2f}")
    else:
        details.append("D/E: N/A")
        
    if fcf_per_share is not None and eps is not None:
        details.append(f"FCF/EPS: {fcf_per_share/eps:.2f}")
    else:
        details.append("FCF/EPS: N/A")
    
    return {
        "signal": signal,
        "details": ", ".join(details)
    }


def analyze_valuation(metrics):
    """Analyze company valuation metrics."""
    # Extract relevant metrics
    pe_ratio = getattr(metrics, "price_to_earnings_ratio", None)
    pb_ratio = getattr(metrics, "price_to_book_ratio", None)
    ps_ratio = getattr(metrics, "price_to_sales_ratio", None)
    
    # Score based on thresholds (higher score means more overvalued)
    score = 0
    if pe_ratio and pe_ratio > 25:  # P/E > 25
        score += 1
    if pb_ratio and pb_ratio > 3:  # P/B > 3
        score += 1
    if ps_ratio and ps_ratio > 5:  # P/S > 5
        score += 1
    
    # Determine signal based on score (inverted for valuation)
    if score >= 2:
        signal = "bearish"  # High valuation metrics are bearish
    elif score == 0:
        signal = "bullish"  # Low valuation metrics are bullish
    else:
        signal = "neutral"
    
    # Format details
    details = []
    if pe_ratio is not None:
        details.append(f"P/E: {pe_ratio:.2f}")
    else:
        details.append("P/E: N/A")
        
    if pb_ratio is not None:
        details.append(f"P/B: {pb_ratio:.2f}")
    else:
        details.append("P/B: N/A")
        
    if ps_ratio is not None:
        details.append(f"P/S: {ps_ratio:.2f}")
    else:
        details.append("P/S: N/A")
    
    return {
        "signal": signal,
        "details": ", ".join(details)
    }