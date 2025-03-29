from graph.state import AgentState, show_agent_reasoning
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from typing_extensions import Literal
from tools.api import get_financial_metrics, get_market_cap, search_line_items
from util.llm import call_llm
from util.progress import progress
import json


class WarrenBuffettSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def warren_buffett_agent(state):
    """Analyzes stocks using Warren Buffett's investment principles."""
    data = state["data"]
    tickers = data["tickers"]
    end_date = data["end_date"]
    metadata = state["metadata"]
    
    analysis_data = {}
    buffett_analysis = {}
    
    for ticker in tickers:
        # Fetch financial data
        progress.update_status("warren_buffett_agent", ticker, "Gathering financial data")
        metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=5)
        financials = search_line_items(
            ticker,
            ["capital_expenditure", "depreciation_and_amortization", "net_income", 
            "outstanding_shares", "total_assets", "total_liabilities", 
            "dividends_and_other_cash_distributions", "issuance_or_purchase_of_equity_shares"],
            end_date
        )
        market_cap = get_market_cap(ticker, end_date)
        
        # Run analysis functions
        progress.update_status("warren_buffett_agent", ticker, "Analyzing company")
        fundamental = analyze_fundamentals(metrics)
        consistency = analyze_consistency(financials)
        moat = analyze_moat(metrics)
        management = analyze_management(financials)
        intrinsic = calculate_intrinsic_value(financials)
        
        # Calculate total score
        total_score = (
            fundamental["score"] + 
            consistency["score"] + 
            moat["score"] + 
            management["score"]
        )
        max_score = 10 + moat["max_score"] + management["max_score"]
        
        # Calculate margin of safety
        margin_of_safety = None
        if intrinsic["intrinsic_value"] and market_cap:
            margin_of_safety = (intrinsic["intrinsic_value"] - market_cap) / market_cap
        
        # Determine signal
        if (total_score >= 0.7 * max_score) and margin_of_safety and margin_of_safety >= 0.3:
            signal = "bullish"
        elif total_score <= 0.3 * max_score or (margin_of_safety and margin_of_safety < -0.3):
            signal = "bearish"
        else:
            signal = "neutral"
        
        # Store analysis data
        analysis_data[ticker] = {
            "signal": signal,
            "score": total_score,
            "max_score": max_score,
            "fundamental_analysis": fundamental,
            "consistency_analysis": consistency,
            "moat_analysis": moat,
            "management_analysis": management,
            "intrinsic_value_analysis": intrinsic,
            "market_cap": market_cap,
            "margin_of_safety": margin_of_safety
        }
        
        # Generate LLM analysis
        progress.update_status("warren_buffett_agent", ticker, "Creating Buffett-style analysis")
        buffett_output = generate_buffett_output(
            ticker=ticker,
            analysis_data=analysis_data,
            model_name=metadata["model_name"],
            model_provider=metadata["model_provider"]
        )
        
        buffett_analysis[ticker] = {
            "signal": buffett_output.signal,
            "confidence": buffett_output.confidence,
            "reasoning": buffett_output.reasoning
        }
        
        progress.update_status("warren_buffett_agent", ticker, "Analysis complete")
    
    # Create message with results
    message = HumanMessage(content=json.dumps(buffett_analysis), name="warren_buffett_agent")
    
    # Show reasoning if requested
    if metadata["show_reasoning"]:
        show_agent_reasoning(buffett_analysis, "Warren Buffett Agent")
    
    # Store analysis in state
    state["data"]["analyst_signals"]["warren_buffett_agent"] = buffett_analysis
    
    return {"messages": [message], "data": data}


def analyze_fundamentals(metrics):
    """Analyze company fundamentals using Buffett's criteria."""
    if not metrics:
        return {"score": 0, "details": "Insufficient fundamental data"}
    
    latest = metrics[0]
    score = 0
    reasons = []
    
    # Check ROE
    roe = getattr(latest, 'return_on_equity', None)
    if roe and roe > 0.15:  # >15% ROE
        score += 2
        reasons.append(f"Strong ROE of {roe:.1%}")
    elif roe:
        reasons.append(f"Weak ROE of {roe:.1%}")
    else:
        reasons.append("ROE data not available")
    
    # Check Debt-to-Equity
    debt_equity = getattr(latest, 'debt_to_equity', None)
    if debt_equity and debt_equity < 0.5:  # Low debt
        score += 2
        reasons.append("Conservative debt levels")
    elif debt_equity:
        reasons.append(f"High debt-to-equity ratio of {debt_equity:.1f}")
    else:
        reasons.append("Debt-to-equity data not available")
    
    # Check Operating Margin
    op_margin = getattr(latest, 'operating_margin', None)
    if op_margin and op_margin > 0.15:  # >15% margin
        score += 2
        reasons.append("Strong operating margins")
    elif op_margin:
        reasons.append(f"Weak operating margin of {op_margin:.1%}")
    else:
        reasons.append("Operating margin data not available")
    
    # Check Current Ratio
    curr_ratio = getattr(latest, 'current_ratio', None)
    if curr_ratio and curr_ratio > 1.5:  # Good liquidity
        score += 1
        reasons.append("Good liquidity position")
    elif curr_ratio:
        reasons.append(f"Weak liquidity with current ratio of {curr_ratio:.1f}")
    else:
        reasons.append("Current ratio data not available")
    
    return {
        "score": score,
        "details": "; ".join(reasons),
        "metrics": latest.model_dump() if hasattr(latest, 'model_dump') else {}
    }


def analyze_consistency(financials):
    """Analyze earnings consistency and growth."""
    if len(financials) < 4:
        return {"score": 0, "details": "Insufficient historical data"}
    
    score = 0
    reasons = []
    
    # Check earnings growth trend
    earnings = [item.net_income for item in financials if getattr(item, 'net_income', None)]
    
    if len(earnings) >= 4:
        # Check if earnings are growing
        consistent_growth = all(earnings[i] > earnings[i+1] for i in range(len(earnings)-1))
        
        if consistent_growth:
            score += 3
            reasons.append("Consistent earnings growth over past periods")
        else:
            reasons.append("Inconsistent earnings growth pattern")
        
        # Calculate total growth rate
        if len(earnings) >= 2 and earnings[-1] != 0:
            growth_rate = (earnings[0] - earnings[-1]) / abs(earnings[-1])
            reasons.append(f"Total earnings growth of {growth_rate:.1%} over past {len(earnings)} periods")
    else:
        reasons.append("Insufficient earnings data for trend analysis")
    
    return {
        "score": score,
        "details": "; ".join(reasons)
    }


def analyze_moat(metrics):
    """Evaluate competitive advantage (moat) based on consistent returns."""
    if not metrics or len(metrics) < 3:
        return {"score": 0, "max_score": 3, "details": "Insufficient data for moat analysis"}
    
    score = 0
    reasons = []
    
    # Extract ROE and margins
    roes = [m.return_on_equity for m in metrics if getattr(m, 'return_on_equity', None) is not None]
    margins = [m.operating_margin for m in metrics if getattr(m, 'operating_margin', None) is not None]
    
    # Check for consistent high ROE
    if len(roes) >= 3:
        if all(r > 0.15 for r in roes):  # All ROEs >15%
            score += 1
            reasons.append("Stable ROE above 15% across periods (suggests moat)")
        else:
            reasons.append("ROE not consistently above 15%")
    
    # Check for consistent high margins
    if len(margins) >= 3:
        if all(m > 0.15 for m in margins):  # All margins >15%
            score += 1
            reasons.append("Stable operating margins above 15% (moat indicator)")
        else:
            reasons.append("Operating margin not consistently above 15%")
    
    # Bonus point if both are strong
    if score == 2:
        score += 1
        reasons.append("Both ROE and margin stability indicate a solid moat")
    
    return {
        "score": score,
        "max_score": 3,
        "details": "; ".join(reasons)
    }


def analyze_management(financials):
    """Analyze management quality through capital allocation decisions."""
    if not financials:
        return {"score": 0, "max_score": 2, "details": "Insufficient data for management analysis"}
    
    score = 0
    reasons = []
    latest = financials[0]
    
    # Check share repurchases
    share_activity = getattr(latest, 'issuance_or_purchase_of_equity_shares', None)
    if share_activity and share_activity < 0:  # Negative = buybacks
        score += 1
        reasons.append("Company has been repurchasing shares (shareholder-friendly)")
    elif share_activity and share_activity > 0:  # Positive = dilution
        reasons.append("Recent common stock issuance (potential dilution)")
    else:
        reasons.append("No significant share activity detected")
    
    # Check dividends
    dividends = getattr(latest, 'dividends_and_other_cash_distributions', None)
    if dividends and dividends < 0:  # Negative = paying dividends
        score += 1
        reasons.append("Company has a track record of paying dividends")
    else:
        reasons.append("No or minimal dividends paid")
    
    return {
        "score": score,
        "max_score": 2,
        "details": "; ".join(reasons)
    }


def calculate_owner_earnings(financials):
    """Calculate owner earnings (Buffett's preferred earnings measure)."""
    if not financials:
        return {"owner_earnings": None, "details": ["Insufficient data for owner earnings calculation"]}
    
    latest = financials[0]
    
    # Extract required components
    net_income = getattr(latest, 'net_income', None)
    depreciation = getattr(latest, 'depreciation_and_amortization', None)
    capex = getattr(latest, 'capital_expenditure', None)
    
    if not all([net_income, depreciation, capex]):
        return {"owner_earnings": None, "details": ["Missing components for owner earnings calculation"]}
    
    # Estimate maintenance capex (75% of total capex)
    maintenance_capex = capex * 0.75
    
    # Calculate owner earnings
    owner_earnings = net_income + depreciation - maintenance_capex
    
    return {
        "owner_earnings": owner_earnings,
        "components": {
            "net_income": net_income,
            "depreciation": depreciation,
            "maintenance_capex": maintenance_capex
        },
        "details": ["Owner earnings calculated successfully"]
    }


def calculate_intrinsic_value(financials):
    """Calculate intrinsic value using DCF with owner earnings."""
    if not financials:
        return {"intrinsic_value": None, "details": ["Insufficient data for valuation"]}
    
    # Calculate owner earnings
    earnings_data = calculate_owner_earnings(financials)
    if not earnings_data["owner_earnings"]:
        return {"intrinsic_value": None, "details": earnings_data["details"]}
    
    owner_earnings = earnings_data["owner_earnings"]
    shares = getattr(financials[0], 'outstanding_shares', None)
    
    if not shares:
        return {"intrinsic_value": None, "details": ["Missing shares outstanding data"]}
    
    # DCF parameters
    growth_rate = 0.05  # 5% growth
    discount_rate = 0.09  # 9% discount rate
    terminal_multiple = 12
    years = 10
    
    # Calculate present value of future earnings
    present_value = 0
    for year in range(1, years + 1):
        future_earnings = owner_earnings * (1 + growth_rate) ** year
        present_value += future_earnings / (1 + discount_rate) ** year
    
    # Calculate terminal value
    terminal_value = (
        owner_earnings * (1 + growth_rate) ** years * terminal_multiple
    ) / (1 + discount_rate) ** years
    
    # Total intrinsic value
    intrinsic_value = present_value + terminal_value
    
    return {
        "intrinsic_value": intrinsic_value,
        "owner_earnings": owner_earnings,
        "assumptions": {
            "growth_rate": growth_rate,
            "discount_rate": discount_rate,
            "terminal_multiple": terminal_multiple,
            "projection_years": years
        },
        "details": ["Intrinsic value calculated using DCF model with owner earnings"]
    }


def generate_buffett_output(ticker, analysis_data, model_name, model_provider):
    """Generate investment recommendation using Buffett's principles."""
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a Warren Buffett AI agent evaluating investments using these principles:
            
            1. Margin of Safety: Buy at a significant discount to intrinsic value
            2. Economic Moat: Look for durable competitive advantages
            3. Quality Management: Favor shareholder-oriented teams
            4. Financial Strength: Prefer low debt, strong returns on equity
            5. Long-term perspective: Focus on business fundamentals, not market sentiment
            
            In your analysis:
            - Explain key factors influencing your decision
            - Highlight alignment with Buffett principles
            - Provide specific financial evidence
            - Use Warren Buffett's conversational style
            """
        ),
        (
            "human",
            """Based on this analysis for {ticker}:
            {analysis_data}

            Return exactly this JSON format:
            {{
            "signal": "bullish" or "bearish" or "neutral",
            "confidence": float between 0 and 100,
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
        return WarrenBuffettSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Error in analysis, defaulting to neutral stance."
        )

    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=WarrenBuffettSignal,
        agent_name="warren_buffett_agent",
        default_factory=default_signal,
    )