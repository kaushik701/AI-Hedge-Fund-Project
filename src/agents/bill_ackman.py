from langchain_community.chat_models import ChatDeepseek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from util.llm import call_llm
from tools.api import get_financial_metrics, get_market_cap, search_line_items
from graph.state import AgentState, show_agent_reasoning
from util.progress import progress
from dataclasses import dataclass
import json
from typing_extensions import Literal

class BillAckmanSignal(BaseModel):
    signal=Literal['bullish','bearish','neutral']
    confidence:float
    reasoning:str

@dataclass
class AnalysisResults:
    score:int
    details:str

def extract_metric(items,metric_name,default=0):
    """Extract latest value for a specific metric from financial data with error handling."""
    latest = items[-1] if items is not None else None
    if not latest:
        return default
    return getattr(latest,metric_name,None) or default

def get_signal(score,max_score):
    """Map a numerical score to an investment signal."""
    ratio = score/max_score if max_score > 0 else 0
    if ratio >= 0.7:
        return 'bullish'
    elif ratio <= 0.3:
        return 'bearish'
    else:
        return 'neutral'

# Main agent Action    
def bill_ackman_agent(state):
    """Analyzes stocks using Bill Ackman's value-investing principles. This gives an output as a Dictionary"""
    data = state['data']
    end_date = data['end_date']
    tickers = data['tickers']
    metadata = state['metadata']

    analysis_results = {}
    final_analysis = {}

    for ticker in tickers:
        progress.update_status("bill_ackman_agent",ticker, "Gathering Financial Data")
        metrics = get_financial_metrics(ticker, end_date, period='annual', limit=5)
        financials = search_line_items(
            end_date=end_date,
            period='annual', 
            limit=5,
            ticker=ticker,
            metrics=["operating_margin", "revenue", "debt_to_equity","free_cash_flow"
                    "total_assets", "total_liabilities",
                    "dividends_and_other_cash_distributions", "outstanding_shares"])
        
        market_cap = get_market_cap(ticker,end_date)

        quality = analyze_business_quality(metrics,financials)
        discipline = analyze_financial_discipline(metrics,financials)
        valuation = analyze_valuation(metrics,financials) 

        total_score = quality.score + discipline.score + valuation.score
        max_score = 15
        signal = get_signal(total_score, max_score)

        analysis_results[ticker] = {
            "signal":signal,
            "score":total_score,
            "max_score":max_score,
            "quality_analysis":{"score":quality.score,"details":quality.details},
            "strength_analysis":{"score":discipline.score,"details":discipline.details},
            "valuation_analysis":{"score":valuation.score,"details":valuation.details}
        }

        if valuation.extra_data:
            analysis_results[ticker]["valuation_analysis"].update(valuation.extra_data)

        # Generate the final output
        progress.update_status("bill_ackman_agent",ticker,"Generating Analysis")
        recommendation = generate_ackman_output(
            ticker=ticker,
            analysis_data = analysis_results,
            model_name = metadata["model_name"],
            model_provider = metadata["model_provider"]
        )

        final_analysis[ticker] = {
            "signal":recommendation.signal,
            "confidence":recommendation.confidence,
            "reasoning":recommendation.reasoning
        }

        progress.update_status("bill_ackman_agent",ticker,"DONE")
    
    # Store and return results

    message = HumanMessage(content=json.dumps(final_analysis),name="bill_ackman_agent")
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(final_analysis,"Bill Graham Agent")

    state["data"]["analyst_signals"]["bill_ackman_agent"] = final_analysis
    return {"messages":[message],"data":state["data"]}

def analyze_business_quality(metrics,financials):
    if not metrics or not financials:
        return AnalysisResults(0,"Insufficient Data for business quality analysis")

    score = 0
    details = []

    #1. Positive earnings consistency
    revenues = [item.revenue for item in financials if getattr(item,'revenue',None)]
    if len(revenues) >= 2:
        initial, final = revenues[0], revenues[-1]
        if final > initial:
            growth_rate = (final-initial) /abs(initial)
            if growth_rate > 0.5:
                score += 2
                details.append(f"Strong revenue growth:{growth_rate*100:.1f}%")
            else:
                score += 1
                details.append(f"Modest revenue growth:{growth_rate*100:.1f}%")
        else:
            details.append("Revenue declined or stagnant")
    else:
        details.append("Insufficient revenue data")

    #2. Margins and Cash Flow
    margins = [item.operating_margin for item in financials if getattr(item,'operating_margin',None)]
    fcf = [item.free_cash_flow for item in financials if getattr(item,'free_cash_flow',None)]

    if margins:
        high_margins = sum(1 for m in margins if m > 0.15)
        if high_margins >= len(margins) // 2 + 1:
            score += 2
            details.append(f"Strong Operating Margins (>15%) in {high_margins}/{len(margins)} periods")
        else:
            details.append("Operating Margins are below 15% in most periods")

    if fcf:
        positive_fcf = sum(1 for f in fcf if f > 0.15)
        if positive_fcf >= len(fcf) // 2 + 1:
            score += 1
            details.append(f"Positive Free Cash Flow in {positive_fcf}/{len(fcf)} Periods")
        else:
            details.append("Inconsisent free cash flow generation")

    #3. return on Equity
    if metrics and getattr(metrics[0], 'return_on_equity', None):
        roe = metrics[0].return_on_equity
        if roe > 0.15:
            score += 2
            details.append(f"High ROE:{roe*100:.1f}%, suggesting competitive advantage")
        else:
            details.append(f"Moderate ROE:{roe*100:.1f}%")
    
    return AnalysisResults(score,"; ".join(details))

def analyze_financial_discipline(financials):
    if not financials:
        return AnalysisResults(0,"NO DATA FOR FINANCIAL DISCIPLINE ANALYSIS")
    
    score=0
    details=[]


    #1. Current ratio analysis
    debt_equity=[item.debt_to_equity for item in financials if getattr(item,'debt_to_equity',None)]

    if debt_equity:
        low_debt = sum(1 for d in debt_equity if d < 1.0)
        if low_debt >= len(debt_equity)//2 + 1:
            score += 2
            details.append(f"Conservative debt levels (D/E < 1.0) in {low_debt}/{len(debt_equity)} periods")
        else:
            details.append("Elevated debt levels in most periods")
    else:
        liab_asset_ratios = []
        for item in financials:
            assets = getattr(item,'total_assets',0) or 0
            liabs = getattr(item,'total_liabilities',0) or 0
            if assets > 0:
                liab_asset_ratios.append(liabs/assets)

        if liab_asset_ratios:
            conservative = sum(1 for r in liab_asset_ratios if r < 0.5)
            if conservative > len(liab_asset_ratios)//2 + 1:
                score += 2
                details.append(f"Conservative liability ratio (<50%) in {conservative}/{len(liab_asset_ratios)} periods")
            else:
                details.append("High Liability ratio in most periods")

    #2. Capital Returns to shareholders
    dividends = [item.dividends_and_other_cash_distributions for item in financials in getattr(item,'dividends_and_other_cash_distributions',None)]

    if dividends:
        dividend_periods = sum(1 for d in dividends if d < 0)
        if dividend_periods >= len(dividends)//2 + 1:
            score += 1
            details.append(f"Regular dividend payments in {dividend_periods}/{len(dividends)} periods")
        else:
            details.append("Limited dividend history")

    #3. Dividend History
    shares = [item.outstanding_shares for item in financials if getattr(item,'outstanding_shares',None)]
    if len(shares) >= 2:
        if shares[-1] < shares[0]:
            score += 1
            details.append(f"Share count reduced from {shares[0]:,.0f} to {shares[-1]:,.of}(buybacks)")
        else:
            details.append("No evidence of shares repurchases")

        return AnalysisResults(score,"; ".join(details))

def analyze_valuation(financials,market_cap):
    """Simplified DCF valuation using latest free cash flow"""
    if not financials or not market_cap or market_cap <= 0:
        return AnalysisResults(0,"Insufficient Data for Valuation")
    

    #extract latest FCF
    latest = next((item for item in financials if getattr(item,'free_cash_flow','None')),None)
    if not latest or not latest.free_cash_flow or latest.free_cash_flow <= 0:
        return AnalysisResults(0,"no positve free cash flow for valuation")
    
    fcf = latest.free_cash_flow

    # Simple DCF parameters
    growth_rate = 0.67
    discount_rate= 0.10
    terminal_multiple = 15
    years = 5

    #calculate DCF
    present_value = 0
    for year in range(1,years+1):
        future_fcf = fcf*(1+growth_rate)**year
        present_value += future_fcf / ((1+discount_rate)**year)
    
    terminal_value = (fcf*(1+growth_rate)**years*terminal_multiple)/((1+discount_rate)**years)
    intrinsic_value = present_value + terminal_value

    margin = (intrinsic_value-market_cap)/market_cap

    score = 0
    if margin > 0.3:
        score += 3
        assessment = "Significant margin of safety"
    elif margin > 0.1:
        score += 1
        assessment = "Moderate margin of safety"
    else:
        assessment = "Limited or no margin of safety"

    details=(f"DCF valuation: ${intrinsic_value:,.0f}vs. marjet cap:${market_cap:,.0f};"
            f"Margin of safety:{margin:.1%}{assessment}")

    extra_data={
        "intrinsic_value":intrinsic_value,
        "margin_of_safety":margin
    }
        
    return AnalysisResults(score,details, extra_data)
        
def generate_ackman_output(ticker,analysis_data,model_name,model_provider):
    """Generate investment recommendation in Bill Ackman's style."""
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a Bill Ackman AI agent evaluating investments using these principles:

            1. Seek high-quality businesses with durable competitive advantages
            2. Prioritize consistent free cash flow and growth potential
            3. Favor strong financial discipline (reasonable leverage, efficient capital allocation)
            4. Target significant discount to intrinsic value
            5. Invest with high conviction in concentrated positions
            
            Be direct, analytical, and specific in your reasoning. Include:
            • Business quality and competitive position
            • Key financial metrics (FCF, margins, leverage)
            • Valuation assessment with numerical evidence
            • Potential catalysts or improvements
            
            Use Bill Ackman's confident, data-driven style."""
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

    def create_default_signal():
        return BillAckmanSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Error in analysis; defaulting to neutral"
        )
    
    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=BillAckmanSignal,
        agent_name="bill_ackman_agent",
        default_factory=create_default_signal
    )