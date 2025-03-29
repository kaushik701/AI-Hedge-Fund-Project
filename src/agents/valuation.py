from langchain_core.messages import HumanMessage
from graph.state import AgentState, show_agent_reasoning
from util.progress import progress
from tools.api import get_financial_metrics, get_market_cap, search_line_items
import json


def valuation_agent(state):
    """Performs valuation analysis using multiple methods."""
    data = state["data"]
    tickers = data["tickers"]
    end_date = data["end_date"]
    
    # Initialize results dictionary
    valuation_analysis = {}
    
    for ticker in tickers:
        progress.update_status("valuation_agent", ticker, "Gathering financial data")
        
        # Get financial metrics
        metrics = get_financial_metrics(ticker, end_date, period="ttm")
        if not metrics:
            progress.update_status("valuation_agent", ticker, "Failed: No financial metrics")
            continue
        
        # Get specific financial line items
        financials = search_line_items(
            ticker,
            ["free_cash_flow", "net_income", "depreciation_and_amortization", 
            "capital_expenditure", "working_capital"],
            end_date, period="ttm", limit=2
        )
        
        if len(financials) < 2:
            progress.update_status("valuation_agent", ticker, "Failed: Insufficient financial data")
            continue
        
        # Get current and previous period data
        current = financials[0]
        previous = financials[1]
        
        # Calculate working capital change
        wc_change = current.working_capital - previous.working_capital
        
        # Perform valuations
        progress.update_status("valuation_agent", ticker, "Calculating valuations")
        
        owner_earnings = calculate_owner_earnings(
            net_income=current.net_income,
            depreciation=current.depreciation_and_amortization,
            capex=current.capital_expenditure,
            wc_change=wc_change,
            growth_rate=metrics[0].earnings_growth
        )
        
        dcf_value = calculate_dcf(
            fcf=current.free_cash_flow,
            growth_rate=metrics[0].earnings_growth
        )
        
        # Get market cap for comparison
        market_cap = get_market_cap(ticker, end_date)
        
        # Calculate valuation gaps
        progress.update_status("valuation_agent", ticker, "Analyzing results")
        
        dcf_gap = (dcf_value - market_cap) / market_cap if market_cap else 0
        oe_gap = (owner_earnings - market_cap) / market_cap if market_cap else 0
        avg_gap = (dcf_gap + oe_gap) / 2
        
        # Determine signal based on valuation gap
        if avg_gap > 0.15:  # Undervalued by >15%
            signal = "bullish"
        elif avg_gap < -0.15:  # Overvalued by >15%
            signal = "bearish"
        else:
            signal = "neutral"
        
        # Calculate confidence based on gap size
        confidence = round(abs(avg_gap) * 100, 2)
        
        # Create reasoning details
        reasoning = {
            "dcf_analysis": {
                "signal": get_signal_from_gap(dcf_gap),
                "details": f"Intrinsic Value: ${dcf_value:,.2f}, Market Cap: ${market_cap:,.2f}, Gap: {dcf_gap:.1%}"
            },
            "owner_earnings_analysis": {
                "signal": get_signal_from_gap(oe_gap),
                "details": f"Owner Earnings Value: ${owner_earnings:,.2f}, Market Cap: ${market_cap:,.2f}, Gap: {oe_gap:.1%}"
            }
        }
        
        # Store analysis results
        valuation_analysis[ticker] = {
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning
        }
        
        progress.update_status("valuation_agent", ticker, "Done")
    
    # Create message with results
    message = HumanMessage(content=json.dumps(valuation_analysis), name="valuation_agent")
    
    # Show reasoning if requested
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(valuation_analysis, "Valuation Analysis Agent")
    
    # Store analysis in state
    state["data"]["analyst_signals"]["valuation_agent"] = valuation_analysis
    
    return {"messages": [message], "data": data}


def calculate_owner_earnings(
    net_income: float,
    depreciation: float,
    capex: float,
    wc_change: float,
    growth_rate: float = 0.05,
    required_return: float = 0.15,
    margin_of_safety: float = 0.25,
    years: int = 5
):
    """Calculate intrinsic value using Buffett's Owner Earnings method."""
    # Check for valid inputs
    if not all(isinstance(x, (int, float)) for x in [net_income, depreciation, capex, wc_change]):
        return 0
    
    # Calculate base owner earnings
    owner_earnings = net_income + depreciation - capex - wc_change
    
    if owner_earnings <= 0:
        return 0
        
    # Project future cash flows
    present_values = []
    for year in range(1, years + 1):
        future_value = owner_earnings * (1 + growth_rate) ** year
        present_value = future_value / (1 + required_return) ** year
        present_values.append(present_value)
    
    # Calculate terminal value
    terminal_growth = min(growth_rate, 0.03)  # Cap terminal growth at 3%
    terminal_value = (present_values[-1] * (1 + terminal_growth)) / (required_return - terminal_growth)
    discounted_terminal = terminal_value / (1 + required_return) ** years
    
    # Apply margin of safety
    total_value = sum(present_values) + discounted_terminal
    safe_value = total_value * (1 - margin_of_safety)
    
    return safe_value


def calculate_dcf(
    fcf: float,
    growth_rate: float = 0.05,
    discount_rate: float = 0.10,
    terminal_rate: float = 0.03,
    years: int = 5
):
    """Calculate intrinsic value using Discounted Cash Flow method."""
    if not isinstance(fcf, (int, float)) or fcf <= 0:
        return 0
    
    # Project cash flows
    cash_flows = [fcf * (1 + growth_rate) ** i for i in range(years)]
    
    # Calculate present value of each cash flow
    present_values = [cf / (1 + discount_rate) ** (i + 1) for i, cf in enumerate(cash_flows)]
    
    # Calculate terminal value
    terminal_value = cash_flows[-1] * (1 + terminal_rate) / (discount_rate - terminal_rate)
    discounted_terminal = terminal_value / (1 + discount_rate) ** years
    
    # Calculate total DCF value
    total_value = sum(present_values) + discounted_terminal
    
    return total_value


def get_signal_from_gap(gap):
    """Convert valuation gap to trading signal."""
    if gap > 0.15:
        return "bullish"
    elif gap < -0.15:
        return "bearish"
    else:
        return "neutral"