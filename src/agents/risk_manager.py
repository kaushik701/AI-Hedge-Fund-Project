from langchain_core.messages import HumanMessage
from graph.state import AgentState, show_agent_reasoning
from util.progress import progress
from tools.api import get_prices, prices_to_df
import json


def risk_management_agent(state: AgentState):
    """Controls position sizing based on portfolio risk constraints."""
    data = state["data"]
    portfolio = data["portfolio"]
    tickers = data["tickers"]
    
    # Initialize risk analysis for each ticker
    risk_analysis = {}
    
    for ticker in tickers:
        progress.update_status("risk_management_agent", ticker, "Fetching price data")
        
        # Get price data
        prices = get_prices(ticker, data["start_date"], data["end_date"])
        
        if not prices:
            progress.update_status("risk_management_agent", ticker, "Failed: No price data found")
            continue
        
        # Convert to DataFrame and get current price
        prices_df = prices_to_df(prices)
        current_price = float(prices_df["close"].iloc[-1])
        
        progress.update_status("risk_management_agent", ticker, "Calculating position limits")
        
        # Calculate portfolio metrics
        position_metrics = calculate_position_metrics(ticker, current_price, portfolio)
        
        # Store analysis results
        risk_analysis[ticker] = {
            "remaining_position_limit": float(position_metrics["max_position_size"]),
            "current_price": current_price,
            "reasoning": {
                "portfolio_value": float(position_metrics["portfolio_value"]),
                "current_position": float(position_metrics["current_position"]),
                "position_limit": float(position_metrics["position_limit"]),
                "remaining_limit": float(position_metrics["remaining_limit"]),
                "available_cash": float(position_metrics["available_cash"]),
            }
        }
        
        progress.update_status("risk_management_agent", ticker, "Done")
    
    # Create message with analysis results
    message = HumanMessage(content=json.dumps(risk_analysis), name="risk_management_agent")
    
    # Show reasoning if requested
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(risk_analysis, "Risk Management Agent")
    
    # Store analysis in state
    state["data"]["analyst_signals"]["risk_management_agent"] = risk_analysis
    
    return {
        "messages": state["messages"] + [message],
        "data": data,
    }


def calculate_position_metrics(ticker, current_price, portfolio):
    """Calculate position limits and related metrics."""
    # Get current position value
    cost_basis = portfolio.get("cost_basis", {})
    current_position = cost_basis.get(ticker, 0)
    
    # Calculate total portfolio value
    portfolio_value = portfolio.get("cash", 0)
    for position_value in cost_basis.values():
        portfolio_value += position_value
    
    # Position limit is 20% of portfolio value
    position_limit = portfolio_value * 0.20
    
    # Remaining limit is position limit minus current position
    remaining_limit = position_limit - current_position
    
    # Available cash
    available_cash = portfolio.get("cash", 0)
    
    # Maximum position size is the minimum of remaining limit and available cash
    max_position_size = min(remaining_limit, available_cash)
    
    return {
        "portfolio_value": portfolio_value,
        "current_position": current_position,
        "position_limit": position_limit,
        "remaining_limit": remaining_limit,
        "available_cash": available_cash,
        "max_position_size": max_position_size
    }