import json
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from graph.state import AgentState, show_agent_reasoning
from pydantic import BaseModel, Field
from typing_extensions import Literal
from util.progress import progress
from util.llm import call_llm


class PortfolioDecision(BaseModel):
    action: Literal["buy", "sell", "short", "cover", "hold"]
    quantity: int = Field(description="Number of shares to trade")
    confidence: float = Field(description="Confidence level (0-100)")
    reasoning: str = Field(description="Explanation for the decision")


class PortfolioManagerOutput(BaseModel):
    decisions: dict[str, PortfolioDecision] = Field(description="Trading decisions by ticker")


def portfolio_management_agent(state: AgentState):
    """Makes trading decisions based on analyst signals and portfolio constraints."""
    data = state["data"]
    tickers = data["tickers"]
    portfolio = data["portfolio"]
    analyst_signals = data["analyst_signals"]
    metadata = state["metadata"]
    
    progress.update_status("portfolio_management_agent", None, "Analyzing signals")
    
    # Organize data needed for trading decisions
    ticker_data = {}
    for ticker in tickers:
        progress.update_status("portfolio_management_agent", ticker, "Processing signals")
        
        # Get risk limits and price data
        risk_data = analyst_signals.get("risk_management_agent", {}).get(ticker, {})
        position_limit = risk_data.get("remaining_position_limit", 0)
        current_price = risk_data.get("current_price", 0)
        
        # Calculate max shares based on position limit
        max_shares = int(position_limit / current_price) if current_price > 0 else 0
        
        # Collect all analyst signals for this ticker
        ticker_signals = {}
        for agent, signals in analyst_signals.items():
            if agent != "risk_management_agent" and ticker in signals:
                ticker_signals[agent] = {
                    "signal": signals[ticker]["signal"],
                    "confidence": signals[ticker]["confidence"]
                }
        
        # Store organized data
        ticker_data[ticker] = {
            "signals": ticker_signals,
            "current_price": current_price,
            "max_shares": max_shares
        }
    
    progress.update_status("portfolio_management_agent", None, "Making trading decisions")
    
    # Generate trading decisions
    signals_by_ticker = {ticker: data["signals"] for ticker, data in ticker_data.items()}
    current_prices = {ticker: data["current_price"] for ticker, data in ticker_data.items()}
    max_shares = {ticker: data["max_shares"] for ticker, data in ticker_data.items()}
    
    trading_decisions = generate_trading_decision(
        tickers=tickers,
        signals_by_ticker=signals_by_ticker,
        current_prices=current_prices,
        max_shares=max_shares,
        portfolio=portfolio,
        model_name=metadata["model_name"],
        model_provider=metadata["model_provider"]
    )
    
    # Create message with decisions
    decision_dict = {ticker: decision.model_dump() for ticker, decision in trading_decisions.decisions.items()}
    message = HumanMessage(content=json.dumps(decision_dict), name="portfolio_management")
    
    # Show reasoning if requested
    if metadata["show_reasoning"]:
        show_agent_reasoning(decision_dict, "Portfolio Management Agent")
    
    progress.update_status("portfolio_management_agent", None, "Done")
    
    return {
        "messages": state["messages"] + [message],
        "data": data
    }


def generate_trading_decision(
    tickers: list[str],
    signals_by_ticker: dict,
    current_prices: dict,
    max_shares: dict,
    portfolio: dict,
    model_name: str,
    model_provider: str
) -> PortfolioManagerOutput:
    """Generate final trading decisions using an LLM."""
    # Create the prompt template
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a portfolio manager making trading decisions based on analyst signals.

            Trading Rules:
            - For long positions:
              * Buy only if cash is available
              * Sell only if you have existing long positions
              * Buy quantity must be ≤ max_shares for that ticker
              * Sell quantity must be ≤ current position size
            
            - For short positions:
              * Short only if margin is available (50% of position value required)
              * Cover only if you have existing short positions
              * Short quantity must respect margin limits
              * Cover quantity must be ≤ current short position size
            
            - Follow position limits provided in max_shares
            - Consider both long and short opportunities
            - Balance risk across the portfolio

            Available Actions:
            - "buy": Open or add to long position
            - "sell": Reduce or close long position
            - "short": Open or add to short position
            - "cover": Reduce or close short position
            - "hold": No action
            """
        ),
        (
            "human",
            """Make trading decisions based on the following information:

            Analyst Signals:
            {signals_by_ticker}

            Current Prices:
            {current_prices}

            Maximum Shares Allowed:
            {max_shares}

            Portfolio Cash: {portfolio_cash}
            Current Positions: {portfolio_positions}
            Margin Requirement: {margin_requirement}

            Output in this JSON format:
            {
            "decisions": {
                "TICKER1": {
                "action": "buy/sell/short/cover/hold",
                "quantity": number_of_shares,
                "confidence": 0-100,
                "reasoning": "explanation"
                },
                "TICKER2": { ... },
                ...
            }
            }
            """
        )
    ])
    
    # Format input data
    prompt = template.invoke({
        "signals_by_ticker": json.dumps(signals_by_ticker, indent=2),
        "current_prices": json.dumps(current_prices, indent=2),
        "max_shares": json.dumps(max_shares, indent=2),
        "portfolio_cash": f"{portfolio.get('cash', 0):.2f}",
        "portfolio_positions": json.dumps(portfolio.get('positions', {}), indent=2),
        "margin_requirement": f"{portfolio.get('margin_requirement', 0):.2f}"
    })
    
    # Default output if LLM call fails
    def default_output():
        return PortfolioManagerOutput(decisions={
            ticker: PortfolioDecision(
                action="hold",
                quantity=0,
                confidence=0.0,
                reasoning="Error in portfolio management, defaulting to hold"
            ) for ticker in tickers
        })
    
    # Call LLM and return decisions
    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=PortfolioManagerOutput,
        agent_name="portfolio_management_agent",
        default_factory=default_output
    )