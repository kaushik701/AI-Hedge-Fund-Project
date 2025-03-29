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

class CathieWoodSignal(BaseModel):
    signal=Literal['bullish','bearish','neutral']
    confidence:float
    reasoning:str

@dataclass
class AnalysisResults:
    score:int
    details:str

def get_safe_value(item,attr_name,default=0):
    """Safely extract attribute value with fallback."""
    if hasattr(item,attr_name):
        value=getattr(item,attr_name)
        return value if value is not None else default
    return default

def get_growth_rate(values):
    """Calculate growth rate between first and last values"""
    if len(values) < 2 or not values[0]:
        return 0
    return (values[-1]-values[0])/abs(values[0])

def map_scores(score,max_score,normalized_max=5):
    """Normalize a score to the desired range"""
    if max_score <= 0:
        return 0
    return (score/max_score)*normalized_max

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
def cathie_wood_agent(state):
    """Analyzes stocks using Cathie Woods's innovation-focused investment approach"""
    data = state['data']
    end_date = data['end_date']
    tickers = data['tickers']
    metadata = state['metadata']

    analysis_results = {}
    wood_analysis = {}

    for ticker in tickers:
        progress.update_status("cathie_wood_agent",ticker, "Gathering Financial Data")
        metrics = get_financial_metrics(ticker, end_date, period='annual', limit=5)
        financials = search_line_items(
            end_date=end_date,
            period='annual', 
            limit=5,
            ticker=ticker,
            metrics=["operating_margin", "revenue", "debt_to_equity","free_cash_flow"
                    "total_assets", "total_liabilities", "gross_margin","research_and_development","capital_expenditure","operating_expense",
                    "dividends_and_other_cash_distributions", "outstanding_shares"])
        
        market_cap = get_market_cap(ticker,end_date)

        disruptive = analyze_disruptive_potential(metrics,financials)
        innovation = analyze_innovation_growth(metrics,financials)
        valuation = analyze_cathie_wood_valuation(metrics,financials) 

        total_score = disruptive.score + innovation.score + valuation.score
        max_score = 15
        signal = get_signal(total_score, max_score)

        analysis_results[ticker] = {
            "signal":signal,
            "score":total_score,
            "max_score":max_score,
            "disruptive_analysis":{"score":disruptive.score,"details":disruptive.details},
            "innovation_analysis":{"score":innovation.score,"details":innovation.details},
            "valuation_analysis":{"score":valuation.score,"details":valuation.details}
        }

        if valuation.extra_data:
            analysis_results[ticker]["valuation_analysis"].update(valuation.extra_data)

        # Generate the final output
        progress.update_status("cathie_wood_agent",ticker,"Generating Analysis")
        recommendation = generate_wood_output(
            ticker=ticker,
            analysis_data = analysis_results,
            model_name = metadata["model_name"],
            model_provider = metadata["model_provider"]
        )

        wood_analysis[ticker] = {
            "signal":recommendation.signal,
            "confidence":recommendation.confidence,
            "reasoning":recommendation.reasoning
        }

        progress.update_status("cathie_wood_agent",ticker,"DONE")
    
    # Store and return results

    message = HumanMessage(content=json.dumps(wood_analysis),name="cathie_wood_agent")
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(wood_analysis,"Bill Graham Agent")

    state["data"]["analyst_signals"]["cathie_wood_agent"] = wood_analysis
    return {"messages":[message],"data":state["data"]}

def analyze_disruptive_potential(metrics,financials):
    if not metrics or not financials:
        return AnalysisResults(0,"Insufficient Data for business quality analysis")

    score = 0
    details = []

    #1. Positive earnings consistency
    revenues = [item.revenue for item in financials if get_safe_value(item,'revenue')]
    if len(revenues) >= 3:
        # Calculate year-over-year growth rates
        growth_rates=[]
        for i in range(len(revenues)-1):
            if revenues[i]:
                growth_rate= (revenues[i+1]-revenues[i])/abs(revenues[i])
                growth_rate.append(growth_rate)

        # Check for accelerating growth        
        if len(growth_rates) >= 2 and growth_rates[-1] > growth_rates[0]:
            score += 2
            details.append(f"Accelerating growth:{growth_rates[-1]*100:.1f}% vs {growth_rates[0]*100:.1f}%")

        # Score based on latest growth rate
        latest_growth = growth_rates[-1] if growth_rates else 0
        if latest_growth > 1.0:
            score += 3
            details.append(f"Exceptional Growth:{latest_growth*100:.1f}%")
        elif latest_growth > 0.5:
            score += 2
            details.append(f"Strong growth:{latest_growth*100:.1f}%")
        elif latest_growth > 0.2:
            score += 1
            details.append(f"Moderate growth:{latest_growth*100:.1f}%")
    else:
        details.append("Insufficient revenue history")

    #2. Margins and Cash Flow
    margins = [get_safe_value(item,'gross_margin') for item in financials]
    if len(margins) >= 2:
        # Check margin trend
        margin_trend = margins[-1] - margins[0]
        if margin_trend > 0.05:
            score += 2
            details.append(f"Expanding margins: +{margin_trend*100:.1f}%")
        elif margin_trend > 0:
            score += 1
            details.append(f"Improving margins: +{margin_trend*100:.1f}%")

        # Score absolute margin level
        if margins[-1] > 0.5:
            score += 2
            details.append(f"High margin business:{margins[-1]*100:.1f}%")
    else:
        details.append("Insufficient margin data")

    #3.Operating Leverages
    
    revenues = [get_safe_value(item,'revenue') for item in financials]
    expenses = [get_safe_value(item,'operating_expense') for item in financials]

    if len(revenues) >=2 and len(expenses)>= 2:
        rev_growth = get_growth_rate(revenues)
        exp_growth = get_growth_rate(expenses)

        if rev_growth > exp_growth:
            score += 2
            details.append("Strong operating leverage")

    #4. R&D investment
    rd_expenses = [get_safe_value(item,'research_and_development') for item in financials]

    if rd_expenses and revenues:
        rd_intensity=rd_expenses[-1]/revenues[-1] if revenues[-1] else 0

        if rd_intensity > 0.15:
            score += 3
            details.append(f"High R&D investment:{rd_intensity*100:.if}% of revenue")
        
        elif rd_intensity > 0.08:
            score += 2
            details.append(f"Good R&D investment:{rd_intensity*100:.1f}% of revenue")
        
        elif rd_intensity > 0.05:
            score += 1
            details.append(f"Modest R&D investment:{rd_intensity*100:.1f}% of revenue")

    return AnalysisResults(map_scores(score,12,5),"; ".join(details))

def analyze_innovation_growth(financials):
    if not financials:
        return AnalysisResults(0,"NO DATA FOR INNOVATION ANALYSIS")
    
    score=0
    details=[]

    #1.R&D Investment Growth 
    rd_expenses = [get_safe_value(item,'research_and_development') for item in financials]
    revenues = [get_safe_value(item,'revenue') for item in financials]

    if len(rd_expenses) >= 2 and all(rd_expenses):
        # R&D growth rate
        rd_growth = get_growth_rate(rd_expenses)

        if rd_growth > 0.5:
            score += 3
            details.append(f"Strong R&D growth:{rd_growth*100:,1f}%")
        elif rd_growth > 0.2:
            score += 2
            details.append(f"Good R&D growth:{rd_growth*100:.1f}%")

        # R&D intensity trend
        if len(revenues) >= 2and revenues[0] and revenues[-1]:
            rd_intensity_start = rd_expenses[0] / revenues[0]
            rd_intensity_end = rd_expenses[-1] / revenues[-1]

            if rd_intensity_end > rd_intensity_start:
                score +=2
                details.append(f"Increasing R&D intensity:{rd_intensity_end*100:,1f}% vs {rd_intensity_start*100:.1f}")
    else:
        details.append("Limited R&D data") 

    #2. Free Cash Flow analysis
    fcf = [get_safe_value(item,'free_cash_flow')for item in financials]

    if len(fcf) >= 2:
        fcf_growth = get_growth_rate(fcf)
        positive_periods = sum(1 for f in fcf if f > 0)

        if fcf_growth > 0.3 and positive_periods == len(fcf):
            score += 3
            details.append("Strong and consistent FCF growth")

        elif positive_periods >= len(fcf)*0.75:
            score += 2
            details.append(f"Consistent FCF: positive in {positive_periods}/{len(fcf)}periods")

        elif positive_periods >= len(fcf)*0.5:
            score += 1
            details.append(f"Consistent FCF: positive in {positive_periods}/{len(fcf)}periods")

    else:
        details.append("Limited FCF data")

    # 3. Operating Margin Trends
    margins = [get_safe_value(item,'operating_margin') for item in financials]

    if len(margins) >= 2:
        margin_trend = margins[-1]-margins[0]

        if margins[-1] > 0.15 and margin_trend > 0:
            score += 3
            details.append(f"Strong and improving margins:{margins[-1]*100:.1f}%")
        elif margins[-1] > 0.10:
            score += 2
            details.append(f"Healthy margins:{margins[-1]*100:.1f}%")
        elif margin_trend > 0:
            score += 1
            details.append("Improving efficiency")

    # 4.Capital Investment
    capex = [get_safe_value(item,'capital_expenditure')for item in financials]

    if capex and revenues:
        capex_intensity = abs(capex[-1])/revenues[-1] if revenues[-1] else 0
        capex_growth = get_growth_rate([abs(c)for c in capex]) if len(capex) >= 2 else 0

        if capex_intensity > 0.1 and capex_growth > 0.2:
            score += 2
            details.append("strong infrastructure investment")
        elif capex_intensity > 0.05:
            score += 1
            details.append("moderate infrastructure investment")

    #5. Growth Reinvestment (vs Dividends)
    dividends = [get_safe_value(item,'dividends_and_other_cash_distributions') for item in financials]

    if dividends and fcf and fcf[-1]:
        payout = abs(dividends[-1])/fcf[-1] if fcf[-1] > 0 else 1

        if payout < 0.2:
            score += 2
            details.append("Strong focus on reinvestment vs dividends")
        elif payout < 0.4:
            score += 1
            details.append("Moderate focus on reinvestment")

    return AnalysisResults(map_scores(score,15,5),"; ".join(details))


def analyze_cathie_wood_valuation(financials,market_cap):
    """Calculate growth-focused valuation using high growth assumptions."""
    if not financials or not market_cap or market_cap <= 0:
        return AnalysisResults(0,"Insufficient Data for Valuation")
    

    #extract latest FCF
    latest=financials[-1] if financials else None
    fcf = get_safe_value(latest,'free_cash_flow')

    if fcf<= 0:
        return AnalysisResults(0,f"No positive FCF ({fcf}) for valuation")
    
    growth_rate = 0.21
    discount_rate = 0.155
    terminal_multiple = 26
    projection_years = 5

    present_value = 0
    for year in range(1,projection_years+1):
        future_fcf = fcf*(1+growth_rate)**year
        pv = future_fcf/((1+discount_rate)**year)
        present_value += pv
    
    terminal_value = (fcf*(1+growth_rate)**projection_years*terminal_multiple)/((1+growth_rate)**projection_years)

    # intrinsic value and margin of safety
    intrinsic_value = present_value+terminal_value
    margin_of_safety = (intrinsic_value-market_cap)/market_cap

    #score based on margin of safety
    score=0
    if margin_of_safety > 0.5:
        score += 5
        assessment = "Exceptional growth opportunity"
    elif margin_of_safety > 0.3:
        score += 4
        assessment = "Strong growth opportunity"
    elif margin_of_safety > 0.2:
        score += 3
        assessment = "Good growth opportunity"
    elif margin_of_safety > 0:
        score += 1
        assessment = "Limited upside"
    else:
        assessment = "OverValuated based on 5-year potential"

    details = f"High-growth DCF:${intrinsic_value:,.0f}vs market cap: ${market_cap:,.0f}; Margin of Safety: {margin_of_safety:.1%};{assessment}"

    extra_data = {
        "intrinsic_value":intrinsic_value,
        "margin_of_safety":margin_of_safety,
    }
    return AnalysisResults(score,details,extra_data)
        
def generate_wood_output(ticker, analysis_data,model_name,model_provider):
    """Generate investment recommendation using Cathie Wood's innovation-focused approach."""
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a Cathie Wood AI agent evaluating investments using these principles:
            
            1. Focus on disruptive innovation and transformative technologies
            2. Look for exponential growth potential and large addressable markets
            3. Value companies with high R&D investment and breakthrough potential
            4. Consider 5+ year time horizons for disruptive investments
            5. Prioritize future winners over current profitability
            
            Be specific in your reasoning by:
            - Identifying the disruptive technology or innovation
            - Highlighting key growth metrics that show potential
            - Discussing the long-term vision and market transformation
            - Examining R&D investment and innovation pipeline
            
            Use Cathie Wood's optimistic, future-focused voice.
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
        return CathieWoodSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Error in analysis, defaulting to neutral stance."
        )

    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=CathieWoodSignal,
        agent_name="cathie_wood_agent",
        default_factory=default_signal,
    )