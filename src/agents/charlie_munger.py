from graph.state import AgentState, show_agent_reasoning
from tools.api import get_financial_metrics, get_market_cap, search_line_items, get_insider_trades, get_company_news
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from dataclasses import dataclass, field
import json
from typing_extensions import Literal
from typing import Dict, List, Any, Optional
from util.progress import progress
from util.llm import call_llm

class CharlieMungerSignal(BaseModel):
    signal:Literal["bullish","bearish","neutral"]
    confidence:float
    reasoning:str

@dataclass
class AnalysisResults:
    score:float
    details:str
    extra_dict:Dict = field(default_factory=dict)

def get_attr(item,attr,default=0):
    """safely get attribute va;ue woth default"""
    if hasattr(item,attr) and getattr(item,attr) is not None:
        return getattr(item,attr)
    return default

def calc_average(values):
    """Calculate average of a list of values"""
    return sum(values) / len(values) if values else 0

def get_signal(score):
    """Convert score to investment signal"""
    if score >= 7.5:
        return "bullish"
    elif score <= 4.5:
        return "bearish"
    return "neutral"

def charlie_munger_agent(state):
    """Analyzes stocks using charlie munger's value investing principles"""
    data = state["data"]
    tickers = data["tickers"]
    end_date=data["end_date"]
    metadata = data["metadata"]

    analysis_data = {}
    munger_analysis = {}

    for ticker in tickers:
        progress.update_status("charlie_munger_agent",ticker,"Gathering Financial Data")
        metrics = get_financial_metrics(ticker,end_date,period="annual",limit=10)
        financials=search_line_items(ticker,
                                    ["revenue", "net_income", "operating_income", "return_on_invested_capital",
            "gross_margin", "operating_margin", "free_cash_flow", "capital_expenditure",
            "cash_and_equivalents", "total_debt", "shareholders_equity", 
            "outstanding_shares", "research_and_development", 
            "goodwill_and_intangible_assets"], end_date,period="annual",limit=10
                                    )
        market_cap = get_market_cap(ticker,end_date)
        insider_trades = get_insider_trades(ticker,end_date,start_date=None,limit=100)
        company_news= get_company_news(ticker,end_date,start_date=None,limit=100)

        # Running analysis Functions
        moat = analyze_moat_strength(metrics,financials)
        management = analyze_management_quality(financials,insider_trades)
        predictability = analyze_predictability(financials)
        valuation=calculate_munger_valuation(financials,market_cap)

        #Calculate weighted score
        total_score=(
            moat.score*0.35+
            management.score*0.25+
            predictability.score*0.25+
            valuation.score*0.15
        )

        # Generate signal
        signal = get_signal(total_score)

        # Store analysis data
        analysis_data[ticker] = {
            "signal":signal,
            "score":total_score,
            "max_score":10,
            "moat_analysis":{"score":moat.score,"details":moat.details},
            "management_analysis":{"score":management.score,"details":management.details},
            "predictability_analysis":{"score":predictability.score,"details":predictability.details},
            "valuation_analysis":{"score":valuation.score,"details":valuation.details},
            "news_sentiment":analyze_news_sentiment(company_news)
        }

        # Add valuation extra data if available
        if valuation.extra_data:
            analysis_data[ticker]["valuation_analysis"].update(valuation.extra_data)

        # Generate final output
        progress.update_status("charlie_munger_agent",ticker,"Generating investment recommendation")
        recommendation = generate_munger_output(ticker,analysis_data=analysis_data,
            model_name=metadata["model_name"],
            model_provider=metadata["model_provider"]
        )
        munger_analysis[ticker]={
            "signal":recommendation.signal,
            "confidence":recommendation.confidence,
            "reasoning":recommendation.reasoning
        }

        progress.update_status("charlie_munger_agent",ticker,"Analysis Complete")

    # Create message and update state
    message = HumanMessage(content=json.dumps(munger_analysis),name="charlie_munger_agent")

    if metadata["show_reasoning"]:
        show_agent_reasoning(munger_analysis,"Charlie Munger Agent")

    state["data"]["analyst_signals"]["charlie_munger_agent"] = munger_analysis

    return {"messages":[message],"data":state["data"]}

def analyze_moat_strength(metrics,financials):
    """Evaluate business moatr using munger's criteria"""
    if not metrics or not financials:
        return AnalysisResults(0,"Insufficient data for moat analysis")
    
    score=0
    details=[]

    # 1. Return on invested capital analysis
    roic_values=[get_attr(item,'return_on_investment_capital')for item in financials]
    roic_values=[r for r in roic_values if r > 0]

    if roic_values:
        high_roic_count = sum(1 for r in roic_values if r > 0.15)

        if high_roic_count >= len(roic_values)*0.8:
            score += 3
            details.append(f"Excellent ROIC: >15% in {high_roic_count}/{len(roic_values)} periods")
        elif high_roic_count >= len(roic_values)*0.5:
            score += 2
            details.append(f"Good ROIC: >15% in {high_roic_count}/{len(roic_values)} periods")
        elif high_roic_count >= len(roic_values)*0.5:
            score += 1
            details.append(f"Mixed ROIC: >15% in {high_roic_count}/{len(roic_values)} periods")
        else:
            details.append("Poor ROIC: Below 15% Threshold")
    else:
        details.append("No ROIC data available")

    # 2. Gross margin analysis (pricing power)
    margins = [get_attr(item,'gross_margin') for item in financials]
    margins = [m for m in margins if m != 0]

    if len(margins) >= 3:
        # check margin improvement trend
        improving_periods = sum(1 for i in range(1,len(margins)) if margins[i] >= margins[i-1])

        if improving_periods >= len(margins)*0.7: # improving in 70% of periods
            score += 2
            details.append("Strong pricing power: Consistently improving margins")
        elif calc_average(margins) > 0.3:
            score += 1
            details.append(f"Good pricing power: Average margin{calc_average(margins):.1%}")
        else:
            details.append("Limited pricing power")

    # 3. Capital intensity analysis
    if len(financials) >= 3:
        capex_ratios = []
        for item in financials:
            capex = abs(get_attr(item,'capital_expenditure'))
            revenue = get_attr(item,'revenue')
            if revenue > 0:
                capex_ratios.append(capex/revenue)

        if capex_ratios:
            avg_capex = calc_average(capex_ratios)
            if avg_capex < 0.05:
                score += 2
                details.append(f"Low capital requirements:{avg_capex:.1%} of revenue")
            elif avg_capex < 0.10:
                score += 1
                details.append(f"Moderate capital requirements:{avg_capex:.1%} of revenue")
            else:
                details.append(f"High capital requirements:{avg_capex:.1%} of revenue")

    # 4. Intellectual property assessment
    r_and_d = [get_attr(item,'research_and_development') for item in financials]
    goodwill = [get_attr(item,'goodwill_and_intangible_assets') for item in financials] 

    if any(r > 0 for r in r_and_d):
        score += 1
        details.append("Invests in R&D, building intellectual property")

    if any(g > 0 for g in goodwill):
        score += 1
        details.append("Significant intangible assets, suggesting brand value")

    # Normalize score to 0-10 range (max raw score is 9)
    final_score = min(10, score*10 / 9)
    return AnalysisResults(final_score,"; ".join(details))

def analyze_management_quality(financials,insider_trades):
    if not financials:
        return AnalysisResults(0,"Insufficient data for management analysis")
    
    score = 0
    details = []

    # 1. Cash conversion (FCF to Net Income)
    fcf_values = [get_attr(item,'free_cash_role') for item in financials]
    net_income = [get_attr(item,'net_income') for item in financials]

    if len(fcf_values) ==  len(net_income):
        fcf_ni_ratios = []
        for i in range(len(fcf_values)):
            if net_income[i] > 0:
                fcf_ni_ratios.append(fcf_values[i]/net_income[i])

        if fcf_ni_ratios:
            avg_ratio = calc_average(fcf_ni_ratios)
            if avg_ratio > 1.1:
                score += 3
                details.append(f"Excellent cash conversion: FCF/NI ratio of {avg_ratio:.2f}")
            elif avg_ratio > 0.9:
                score += 2
                details.append(f"Good cash conversion: FCF/NI ratio of {avg_ratio:.2f}")
            elif avg_ratio > 0.7:
                score += 1
                details.append(f"Moderate cash conversion: FCF/NI ratio of {avg_ratio:.2f}")
            else:
                details.append(f"Poor cash conversion: FCF/NI ratio of {avg_ratio:.2f}")

    # 2. Debt Management
    debt = [get_attr(item,'total_debt') for item in financials]
    equity = [get_attr(item,'shareholders_equity') for item in financials]

    if debt and equity and debt[0] > 0 and equity[0] > 0:
        # Use most recent period
        de_ratio = debt[0]/equity[0]

        if de_ratio < 0.3:
            score += 3
            details.append(f"Conservative debt: D/E ratio of {de_ratio:.2f}")
        elif de_ratio < 0.7:
            score += 2
            details.append(f"Prudent debt: D/E ratio of {de_ratio:.2f}")
        elif de_ratio < 1.5:
            score += 1
            details.append(f"Moderate debt: D/E ratio of {de_ratio:.2f}")
        else:
            details.append(f"High Debt: D/E ratio of {de_ratio:.2f}")

    # 3. Cash Efficiency
    cash = [get_attr(item,'cash_and_equivalents') for item in financials]
    revenue = [get_attr(item,'revenue') for item in financials]

    if cash and revenue and cash[0] > 0 and revenue[0] > 0:
        cash_ratio = cash[0]/revenue[0]

        if 0.1 <= cash_ratio <= 0.25:
            score += 2
            details.append(f"Prudent cash management:{cash_ratio:.2f} of revenue")
        elif 0.05 <= cash_ratio < 0.1 or 0.25 < cash_ratio <= 0.4:
            score += 1
            details.append(f"Acceptable cash position:{cash_ratio:.2f} of revenue")
        elif cash_ratio > 0.4:
            details.append(f"Excess cash:{cash_ratio:.2f} of revenue")
        else:
            details.append(f"Low Cash reserves:{cash_ratio:.2f} of revenue")

    # 4. Insider ownership
    if insider_trades:
        buys= sum(1 for trade in insider_trades if hasattr(trade, 'transaction_type') and trade.transaction_type 
                and trade.transaction_type.lower() in ['buy','purchase'])
        sells = sum(1 for trade in insider_trades if hasattr(trade, 'transaction_type') and trade.transaction_type 
                and trade.transaction_type.lower() in ['sell','sale'])

        total = buys+sells

        if total > 0:
            buy_ratio = buys / total

            if buy_ratio > 0.7:
                score += 2
                details.append(f"Strong insider buying:{buys}/{total} transactions")    
            elif buy_ratio > 0.4:
                score += 1
                details.append(f"Balanced insider buying:{buys}/{total} transactions")
            elif buy_ratio < 0.1 and sells > 5:
                score -= 1
                details.append(f"Strong insider buying:{buys}/{total} transactions")
            else:
                details.append(f"Mixed insider activity:{buys}/{total} transactions")

    # 5. Share count trends
    shares = [get_attr(item,'outstanding_shares') for item in financials]

    if len(shares) >= 3 and all(s > 0 for s in shares):
            if shares[0] < shares[-1] * 0.95:
                score += 2
                details.append("Shareholder-friendly: Reducing share count")
            elif shares[0] < shares[-1] * 1.05:
                score += 1
                details.append("Stable share count: Limiting dilution")
            elif shares[0] > shares[-1] * 1.2:
                score -= 1
                details.append("Concerning dilution: Significant share increase")
            else:
                details.append("Moderate share count increase")

    final_score = max(0,min(10,score*10/12))

    return AnalysisResults(final_score,"; ".join(details))
    
def analyze_predictability(financials):
    """Assess business predictability based on financial stability."""
    if not financials or len(financials) < 5:
        return AnalysisResults(0,"Insufficient data for predictability analysis (need 5+ years)")
    
    score = 0
    details = []

    # 1. Revenue stability
    revenues = [get_attr(item, 'revenue') for item in financials]
    revenues = [r for r in revenues if r > 0]
    
    if len(revenues) >= 5:
        # Calculate growth rates
        growth_rates = [(revenues[i] / revenues[i+1] - 1) for i in range(len(revenues)-1)]
        
        avg_growth = calc_average(growth_rates)
        growth_volatility = calc_average([abs(r - avg_growth) for r in growth_rates])
        
        if avg_growth > 0.05 and growth_volatility < 0.1:  # Steady growth
            score += 3
            details.append(f"Highly predictable revenue: {avg_growth:.1%} growth, low volatility")
        elif avg_growth > 0 and growth_volatility < 0.2:  # Moderate growth
            score += 2
            details.append(f"Moderately predictable revenue: {avg_growth:.1%} growth")
        elif avg_growth > 0:  # Positive but volatile
            score += 1
            details.append(f"Growing but volatile revenue: {avg_growth:.1%} growth")
        else:
            details.append(f"Declining or unpredictable revenue: {avg_growth:.1%} growth")

    # 2. Operating income stability
    op_income = [get_attr(item, 'operating_income') for item in financials]
    
    if len(op_income) >= 5:
        positive_periods = sum(1 for oi in op_income if oi > 0)
        
        if positive_periods == len(op_income):  # Always profitable
            score += 3
            details.append("Highly predictable: Positive operating income in all periods")
        elif positive_periods >= len(op_income) * 0.8:  # Mostly profitable
            score += 2
            details.append(f"Predictable: Income positive in {positive_periods}/{len(op_income)} periods")
        elif positive_periods >= len(op_income) * 0.6:  # Somewhat profitable
            score += 1
            details.append(f"Somewhat predictable: Income positive in {positive_periods}/{len(op_income)} periods")
        else:
            details.append(f"Unpredictable: Income positive in only {positive_periods}/{len(op_income)} periods")

    # 3. Margin consistency
    margins = [get_attr(item, 'operating_margin') for item in financials]
    margins = [m for m in margins if m != 0]
    
    if len(margins) >= 5:
        avg_margin = calc_average(margins)
        margin_volatility = calc_average([abs(m - avg_margin) for m in margins])
        
        if margin_volatility < 0.03:  # Very stable
            score += 2
            details.append(f"Highly predictable margins: {avg_margin:.1%} avg, minimal volatility")
        elif margin_volatility < 0.07:  # Moderately stable
            score += 1
            details.append(f"Moderately predictable margins: {avg_margin:.1%} avg")
        else:
            details.append(f"Unpredictable margins: {avg_margin:.1%} avg, high volatility")
    
    # 4. Cash flow stability
    fcf = [get_attr(item, 'free_cash_flow') for item in financials]
    
    if len(fcf) >= 5:
        positive_fcf = sum(1 for f in fcf if f > 0)
        
        if positive_fcf == len(fcf):  # Always positive
            score += 2
            details.append("Highly predictable cash flow: Positive in all periods")
        elif positive_fcf >= len(fcf) * 0.8:  # Mostly positive
            score += 1
            details.append(f"Predictable cash flow: Positive in {positive_fcf}/{len(fcf)} periods")
        else:
            details.append(f"Unpredictable cash flow: Positive in only {positive_fcf}/{len(fcf)} periods")

    # Normalize score to 0-10 range (max raw score is 10)
    final_score = min(10, score)
    return AnalysisResults(final_score, "; ".join(details))

def calculate_munger_valuation(financials,market_cap):
    """Calculate intrinsic value using Munger's approach."""
    if not financials or not market_cap or market_cap <= 0:
        return AnalysisResults(0, "Insufficient data for valuation")
    
    # Get FCF values (Munger's preferred metric)
    fcf_values = [get_attr(item, 'free_cash_flow') for item in financials]
    fcf_values = [f for f in fcf_values if f > 0]
    
    if len(fcf_values) < 3:
        return AnalysisResults(0, "Insufficient FCF history for valuation")
    
    score = 0
    details = []
    
    # Normalize earnings (average of recent years)
    normalized_fcf = sum(fcf_values[:min(5, len(fcf_values))]) / min(5, len(fcf_values))
    
    # Calculate FCF yield
    fcf_yield = normalized_fcf / market_cap
    
    # Score based on FCF yield
    if fcf_yield > 0.08:  # >8% yield (P/FCF < 12.5x)
        score += 4
        details.append(f"Excellent value: {fcf_yield:.1%} FCF yield")
    elif fcf_yield > 0.05:  # >5% yield (P/FCF < 20x)
        score += 3
        details.append(f"Good value: {fcf_yield:.1%} FCF yield")
    elif fcf_yield > 0.03:  # >3% yield (P/FCF < 33x)
        score += 1
        details.append(f"Fair value: {fcf_yield:.1%} FCF yield")
    else:
        details.append(f"Expensive: Only {fcf_yield:.1%} FCF yield")
    
    # Calculate simple intrinsic value range
    conservative_value = normalized_fcf * 10  # 10% yield
    reasonable_value = normalized_fcf * 15  # 6.7% yield
    optimistic_value = normalized_fcf * 20  # 5% yield
    
    # Calculate margin of safety
    margin = (reasonable_value - market_cap) / market_cap
    
    if margin > 0.3:  # >30% upside
        score += 3
        details.append(f"Large margin of safety: {margin:.1%} upside")
    elif margin > 0.1:  # >10% upside
        score += 2
        details.append(f"Moderate margin of safety: {margin:.1%} upside")
    elif margin > -0.1:  # Within 10% of reasonable value
        score += 1
        details.append(f"Fair price: Within 10% of reasonable value")
    else:
        details.append(f"Expensive: {-margin:.1%} premium to reasonable value")
    
    # Check earnings trend
    if len(fcf_values) >= 6:
        recent_avg = sum(fcf_values[:3]) / 3
        older_avg = sum(fcf_values[-3:]) / 3
        
        if recent_avg > older_avg * 1.2:  # >20% growth
            score += 3
            details.append("Growing FCF trend adds to intrinsic value")
        elif recent_avg > older_avg:  # Positive growth
            score += 2
            details.append("Stable to growing FCF supports valuation")
        else:
            details.append("Declining FCF trend is concerning")
    else:
        # Simpler calculation for shorter history
        if len(fcf_values) >= 3 and fcf_values[0] > fcf_values[-1]:
            score += 2
            details.append("Growing FCF trend supports valuation")
    
    # Normalize score to 0-10 range (max raw score is 10)
    final_score = min(10, score)
    
    # Add valuation data to extra_data
    extra_data = {
        "intrinsic_value_range": {
            "conservative": conservative_value,
            "reasonable": reasonable_value,
            "optimistic": optimistic_value
        },
        "fcf_yield": fcf_yield,
        "normalized_fcf": normalized_fcf
    }
    
    return AnalysisResults(final_score, "; ".join(details), extra_data) 

def analyze_news_sentiment(news_items):
    """Simple qualitative analysis of recent news."""
    if not news_items or len(news_items) == 0:
        return "No news data available"
    return f"Qualitative review of {len(news_items)} recent news items would be needed"

def generate_munger_output(
    ticker,
    analysis_data,
    model_name,
    model_provider,
):
    """Generate investment recommendation using Charlie Munger's principles."""
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a Charlie Munger AI agent evaluating investments using these principles:
            
            1. Focus on business quality and predictability
            2. Look for durable competitive advantages (moats)
            3. Value high returns on invested capital
            4. Prioritize management integrity and competence
            5. Demand a margin of safety in valuation
            6. Avoid complexity and businesses you don't understand
            7. "Invert, always invert" - focus on avoiding mistakes
            
            In your analysis:
            - Apply mental models from multiple disciplines
            - Emphasize long-term economics over short-term metrics
            - Be skeptical of excessive leverage or dilution
            - Use Charlie's direct, pithy conversational style
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
        return CharlieMungerSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Error in analysis, defaulting to neutral stance."
        )

    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=CharlieMungerSignal,
        agent_name="charlie_munger_agent",
        default_factory=default_signal,
    )