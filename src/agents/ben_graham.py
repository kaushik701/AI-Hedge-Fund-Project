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
import math

class BenGrahamSignal(BaseModel):
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

def map_signal(score,max_score):
    """Map a numerical score to an investment signal."""
    ratio = score/max_score if max_score > 0 else 0
    if ratio >= 0.7:
        return 'bullish'
    elif ratio <= 0.3:
        return 'bearish'
    else:
        return 'neutral'

# Main agent Action    
def ben_graham_agent(state):
    """Analyzes stocks using Benjamin Graham's value-investing principles. This gives an output as a Dictionary"""
    data = state['data']
    end_date = data['end_date']
    tickers = data['tickers']
    metadata = state['metadata']

    analysis_data = {}
    graham_analysis = {}

    for ticker in tickers:
        progress.update_status("ben_graham_agent",ticker, "Gathering Financial Data")
        metrics = get_financial_metrics(ticker, end_date, period='annual', limit=10)
        line_items = search_line_items(
            end_date=end_date,
            period='annual', 
            limit=10,
            ticker=ticker,
            metrics=["earnings_per_share", "revenue", "net_income", "book_value_per_share",
                    "total_assets", "total_liabilities", "current_assets", "current_liabilities",
                    "dividends_and_other_cash_distributions", "outstanding_shares"])
        
        market_cap = get_market_cap(ticker,end_date)

        earnings = analyze_earning_stability(line_items)
        strength = analyze_financial_strength(line_items)
        valuation = analyze_valuation_graham(line_items) 

        total_score = earnings.score + strength.score + valuation.score
        max_score = 15
        signal = map_signal(total_score, max_score)

        analysis_data[ticker] = {
            "signal":signal,
            "score":total_score,
            "max_score":max_score,
            "earning_analysis":{"score":earnings.score,"details":earnings.details},
            "strength_analysis":{"score":strength.score,"details":strength.details},
            "valuation_analysis":{"score":valuation.score,"details":valuation.details}
        }

        # Generate the final output
        progress.update_status("ben_graham_agent",ticker,"Generating Analysis")
        graham_output = generate_graham_output(
            ticker=ticker,
            analysis_data = analysis_data,
            model_name = metadata["model_name"],
            model_provider = metadata["model_provider"]
        )

        graham_analysis[ticker] = {
            "signal":graham_output.signal,
            "confidence":graham_output.confidence,
            "reasoning":graham_output.reasoning
        }

        progress.update_status("ben_graham_agent",ticker,"DONE")
    
    # Store and return results

    message = HumanMessage(content=json.dumps(graham_analysis),name="ben_graham_agent")
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(graham_analysis,"Ben Graham Agent")

    state["data"]["analyst_signals"]["ben_graham_agent"] = graham_analysis
    return {"messages":[message],"data":state["data"]}

def analyze_earning_stability(financial_items):
    if not financial_items or len(financial_items) < 2:
        return AnalysisResults(0,"Insufficient Data for earnings stability analysis")
    
    # Extract EPS values
    eps_vals = [item.earnings_per_share for item in financial_items if item.earnings_per_share is not None]
    if len(eps_vals) < 2:
        return AnalysisResults(0,"Not enough multi year EPS data")
    
    # Score based on EPS history
    score = 0
    details = []

    #1. Positive earnings consistency
    positive_years = sum(1 for e in eps_vals if e > 0)
    total_years = len(eps_vals)

    if positive_years == total_years:
        score += 3
        details.append("EPS positive in all periods")
    elif positive_years >= (total_years*0.8):
        score += 2
        details.append(f"EPS positive in {positive_years}/{total_years} periods")
    else:
        details.append(f"EPS negative in {total_years-positive_years}/{total_years} periods")

    #2. Growth from earliest to latest
    if eps_vals[-1] > eps_vals[0]:
        score += 1
        details.append(f"EPS grew from {eps_vals[0]:.2f} to {eps_vals[-1]:.2f}")
    else:
        details.append(f"NO EPS GROWTH:{eps_vals[0]:.2f} to {eps_vals[-1]:.2f}")

    return AnalysisResults(score,"; " .join(details))

def analyze_financial_strength(financial_items):
    if not financial_items:
        return AnalysisResults(0,"NO DATA FOR FINANCIAL STRENGTH ANALYSIS")
    
    score=0
    details=[]

    #extract latest metrics
    latest=financial_items[-1]
    total_assets=getattr(latest,'total_assets',0) or 0
    total_liabs=getattr(latest,'total_liabilities',0) or 0
    current_assets=getattr(latest,'current_assets',0) or 0
    current_liabs=getattr(latest,'current_liabilities',) or 0

    #1. Current ratio analysis
    if current_liabs > 0:
        current_ratio=current_assets/current_liabs
        if current_ratio >= 2.0:
            score+=2
            details.append(f"Current Ratio:{current_ratio:.2f}(Excellent)")
        elif current_ratio >= 1.5:
            score+=1
            details.append(f"Current Ratio:{current_ratio:.2f}(Good)")
        else:
            details.append(f"Current Ratio:{current_ratio:.2f}(Weak)")
    else:
        details.append("Cannot compute Current Ratio")

    #2. Debt Ratio analysis
    if total_assets > 0:
        debt_ratio = total_liabs/total_assets
        if debt_ratio < 0.5:
            score += 2
            details.append(f"Debt Ratio{debt_ratio:.2f}(Conservative)")
        elif debt_ratio < 0.8:
            score += 1
            details.append(f"Debt Ratio{debt_ratio:.2f}(Acceptable)")
        else:
            details.append(f"Debt Ratio{debt_ratio:.2f}(High)")
    else:
        details.append("Cannot compute Debt Ratio")

    #3. Dividend History
    div_values = [item.dividends_and_other_cash_distributions for item in financial_items
                if item.dividends_and_other_cash_distributions is not None]
    
    if div_values:
        div_years=sum(1 for d in div_values if d < 0)
        if div_years >= (len(div_values)//2+1):
            score += 1
            details.append(f"Paid dividends in {div_years}/{len(div_values)} years")
        else:
            details.append(f"Limited Diviend History:{div_years}/{len(div_values) - div_years} years")
    else:
        details.append("No divided Available")

    return AnalysisResults(score,"; ".join(details))

def analyze_valuation_graham(financial_items,market_cap):
    """Evaluate valuation using Graham's methods: Net-Net and Graham Number."""
    if not financial_items or not market_cap or market_cap <= 0:
        return AnalysisResults(0,"Insufficient Data for Valuation")
    
    score = 0
    details = []

    #extract needed metrics
    latest = financial_items[-1]
    current_assets = getattr(latest,'current_assets',0) or 0
    total_liabs = getattr(latest,'total_liabilities',0) or 0
    book_value_ps = getattr(latest,'book_value_per_share',0) or 0
    eps = getattr(latest,'earnings_per_share',0) or 0
    shares = getattr(latest,'outstanding_shares',0) or 0

    #calculate per share values
    price_per_share = market_cap/shares if shares > 0 else 0

    #1. Net-Net valuation
    ncav = current_assets - total_liabs
    ncav_per_share =  ncav/shares if shares > 0 else 0

    details.append(f"NCAV per share:${ncav_per_share:.2f} vs Price:${price_per_share:.2f}")

    if ncav > market_cap:
        score += 4
        details.append("Net-Net: Trading below NCAV:(Classic Graham Value)")
    elif ncav_per_share >= (price_per_share*0.68):
        score += 2
        details.append("Moderate discount to NCAV")

    #2. Graham Number Valuation
    if eps > 0 and book_value_ps > 0:
        graham_number = math.sqrt(22.5*eps*book_value_ps)
        details.append(f"Graham Number:${graham_number:.2f}")

        if price_per_share > 0:
            margin = (graham_number-price_per_share)/price_per_share
            details.append(f"Margin of Safety:{margin:.1%}")

            if margin > 0.5:
                score += 3
                details.append("Large Margin of Safety(>50%)")
            elif margin > 0.2:
                score += 1
                details.append("Moderate Margin of Safety(>20%)")
            else:
                details.append("Low Margin of Safety")
        else:
            details.append("Cannot compute Graham Number (EPS or Book Value <= 0)")
        
        return AnalysisResults(score,"; ".join(details))
        
def generate_graham_output(ticker,analysis_data,model_name,model_provider):
    template= ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a Benjamin Graham AI agent making investment decisions based on:
            - Margin of safety (buying below intrinsic value)
            - Strong financials (low debt, good liquidity)
            - Stable earnings history
            - Dividend record
            - Conservative valuation

            Return a clear recommendation: bullish, bearish, or neutral, with confidence level and reasoning.
            """
        ),
        (
            "human",
            """Based on this analysis for {ticker}:
            {analysis_data}

            Return JSON exactly as:
            {{
            "signal": "bullish" or "bearish" or "neutral",
            "confidence": float (0-100),
            "reasoning": "string"
            }}
            """ 
        )
    ])
    prompt = template.invoke({
        "analysis_data":json.dumps(analysis_data,indent=2),
        "ticker":ticker
    })

    def create_default_signal():
        return BenGrahamSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Error in analysis; defaulting to neutral"
        )
    
    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=BenGrahamSignal,
        agent_name="ben_graham_agent",
        default_factory=create_default_signal
    )