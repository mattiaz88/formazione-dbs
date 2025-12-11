"""
Lesson 3: Practical LLM Applications - Exercise Solutions
Course: Development of Agentic AI Systems

This file contains solutions for all exercises in the Lesson 3 Applications notebook.
"""

import os
import json
import re
from typing import Optional, List, Dict, Any
from datetime import datetime


# =============================================================================
# EXERCISE 1.1 SOLUTION: Build a Custom Conversational Assistant
# =============================================================================

def exercise_1_1_solution():
    """
    Solution for Exercise 1.1: Custom Conversational Assistant
    
    This example creates a financial analysis assistant that helps with
    budget planning and cost analysis for advertising campaigns.
    """
    
    FINANCIAL_ASSISTANT_PROMPT = """You are a financial analyst specializing in advertising budget management.

Your expertise includes:
- Campaign budget planning and allocation
- Cost efficiency analysis (CPM, CPA, ROAS)
- Media mix optimization
- ROI calculations and forecasting

Guidelines for responses:
- Always ask for specific numbers when doing calculations
- Provide clear breakdowns of costs and allocations
- Suggest industry benchmarks when relevant
- Highlight potential cost savings opportunities
- Use currency formatting consistently (default to EUR)

When the user provides budget data, calculate and present:
1. Key cost metrics
2. Efficiency ratios
3. Recommendations for optimization"""

    # Example conversation flow
    example_conversation = [
        {
            "user": "I have a total budget of 150,000 EUR for Q1. How should I allocate it across TV, digital, and OOH?",
            "expected_topics": ["allocation percentages", "channel recommendations", "benchmark references"]
        },
        {
            "user": "The TV CPM is around 12 EUR and digital is 3 EUR. Does my allocation make sense?",
            "expected_topics": ["CPM comparison", "reach vs cost trade-offs", "revised recommendations"]
        },
        {
            "user": "What if I want to maximize reach in the 25-34 demographic?",
            "expected_topics": ["demographic targeting", "channel effectiveness by age", "budget reallocation"]
        }
    ]
    
    return {
        "system_prompt": FINANCIAL_ASSISTANT_PROMPT,
        "example_conversation": example_conversation,
        "implementation_notes": """
To implement this solution:

1. Initialize the ConversationalAssistant:
   financial_assistant = ConversationalAssistant(
       system_prompt=FINANCIAL_ASSISTANT_PROMPT,
       provider="groq"
   )

2. Test with the example conversation:
   for exchange in example_conversation:
       response = financial_assistant.chat(exchange["user"])
       print(f"User: {exchange['user']}")
       print(f"Assistant: {response}")
       print("---")

3. Verify that:
   - Context is maintained (the assistant remembers the 150,000 EUR budget)
   - Calculations are consistent across turns
   - Recommendations evolve based on new information
"""
    }


# =============================================================================
# EXERCISE 2.1 SOLUTION: Create a Custom Document Extractor
# =============================================================================

def exercise_2_1_solution():
    """
    Solution for Exercise 2.1: Custom Document Extractor
    
    This example creates an invoice extractor that can process
    vendor invoices and extract billing information.
    """
    
    invoice_schema = {
        "invoice_number": {
            "type": "string",
            "description": "Unique invoice identifier or number",
            "required": True
        },
        "vendor_name": {
            "type": "string",
            "description": "Name of the vendor or supplier",
            "required": True
        },
        "invoice_date": {
            "type": "string",
            "description": "Date the invoice was issued (YYYY-MM-DD format)",
            "required": True
        },
        "due_date": {
            "type": "string",
            "description": "Payment due date (YYYY-MM-DD format)",
            "required": False
        },
        "total_amount": {
            "type": "number",
            "description": "Total invoice amount (numeric value only)",
            "required": True
        },
        "currency": {
            "type": "string",
            "description": "Currency code (EUR, USD, etc.)",
            "required": True
        },
        "line_items": {
            "type": "list",
            "description": "List of individual line items with description and amount",
            "required": False
        },
        "tax_amount": {
            "type": "number",
            "description": "Total tax amount if applicable",
            "required": False
        },
        "payment_terms": {
            "type": "string",
            "description": "Payment terms (e.g., Net 30, Due on Receipt)",
            "required": False
        }
    }
    
    sample_invoice_document = """
INVOICE

Vendor: MediaBuy Solutions SRL
Invoice No: INV-2024-0892
Date: November 15, 2024
Due Date: December 15, 2024

Bill To:
TechCorp Italia SpA
Via Roma 123
20121 Milano, Italy

Description                              Quantity    Unit Price    Amount
------------------------------------------------------------------------
TV Advertising Spots (30s)                    45       €1,200.00   €54,000.00
Digital Banner Campaign                        1      €18,500.00   €18,500.00
Production Services                            1       €8,000.00    €8,000.00
Agency Fee (15%)                               1      €12,075.00   €12,075.00

                                             Subtotal:             €92,575.00
                                             VAT (22%):            €20,366.50
                                             ----------------------------------------
                                             TOTAL DUE:           €112,941.50

Payment Terms: Net 30
Bank: Banca Intesa - IBAN: IT60X0542811101000000123456

Thank you for your business.
"""
    
    expected_extraction = {
        "invoice_number": "INV-2024-0892",
        "vendor_name": "MediaBuy Solutions SRL",
        "invoice_date": "2024-11-15",
        "due_date": "2024-12-15",
        "total_amount": 112941.50,
        "currency": "EUR",
        "line_items": [
            {"description": "TV Advertising Spots (30s)", "amount": 54000.00},
            {"description": "Digital Banner Campaign", "amount": 18500.00},
            {"description": "Production Services", "amount": 8000.00},
            {"description": "Agency Fee (15%)", "amount": 12075.00}
        ],
        "tax_amount": 20366.50,
        "payment_terms": "Net 30"
    }
    
    return {
        "schema": invoice_schema,
        "sample_document": sample_invoice_document,
        "expected_extraction": expected_extraction,
        "implementation_code": """
# Implementation:
from document_extractor import DocumentExtractor

invoice_extractor = DocumentExtractor(
    schema=invoice_schema,
    provider="groq"
)

result = invoice_extractor.extract(sample_invoice_document)
print(json.dumps(result, indent=2, ensure_ascii=False))

# Validation should return:
# - is_valid: True
# - All required fields present
# - Numeric fields properly typed
"""
    }


# =============================================================================
# EXERCISE 2.2 SOLUTION: Extend the Document Analysis Agent
# =============================================================================

def exercise_2_2_solution():
    """
    Solution for Exercise 2.2: Extend Document Analysis Agent with New Tool
    
    This solution adds a tool for extracting and analyzing named entities
    (organizations, locations, products) from documents.
    """
    
    tool_implementation = '''
from langchain_core.tools import tool

@tool
def extract_entities(text: str) -> str:
    """
    Extract named entities (organizations, locations, products) from text.
    Use this to identify key actors, places, and items mentioned in a document.
    """
    import re
    
    # Organization patterns (simplified - in production use NER models)
    org_patterns = [
        r'(?:Inc\.|Ltd\.|SRL|SpA|GmbH|LLC|Corp\.?)',
        r'[A-Z][a-z]+ (?:Group|Company|Corporation|Agency|Media|Solutions)',
    ]
    
    # Location patterns (major cities/countries)
    locations = ['Milan', 'Milano', 'Rome', 'Roma', 'Italy', 'Italia', 
                 'New York', 'London', 'Paris', 'Berlin', 'Tokyo']
    
    # Product/Brand patterns
    product_patterns = [
        r'Campaign:?\s*([A-Z][A-Za-z0-9\s]+)',
        r'"([^"]+)"',  # Quoted items often are product/campaign names
    ]
    
    entities = {
        "organizations": [],
        "locations": [],
        "products_campaigns": []
    }
    
    # Find organizations
    for pattern in org_patterns:
        matches = re.findall(r'[\w\s]+' + pattern, text)
        entities["organizations"].extend([m.strip() for m in matches])
    
    # Find locations
    for loc in locations:
        if loc.lower() in text.lower():
            entities["locations"].append(loc)
    
    # Find products/campaigns
    for pattern in product_patterns:
        matches = re.findall(pattern, text)
        entities["products_campaigns"].extend(matches[:5])  # Limit to avoid noise
    
    # Remove duplicates
    for key in entities:
        entities[key] = list(set(entities[key]))
    
    result = "Entities found:\\n"
    result += f"- Organizations: {entities['organizations'] or 'None identified'}\\n"
    result += f"- Locations: {entities['locations'] or 'None identified'}\\n"
    result += f"- Products/Campaigns: {entities['products_campaigns'] or 'None identified'}"
    
    return result


@tool
def summarize_section(text: str, max_sentences: int = 3) -> str:
    """
    Create a brief summary of a text section.
    Use this to condense long paragraphs into key points.
    Specify max_sentences to control summary length (default: 3).
    """
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    if not sentences:
        return "No substantive content to summarize."
    
    # Simple extractive summarization: take first N sentences
    # In production, use proper summarization techniques
    summary_sentences = sentences[:max_sentences]
    
    return "Summary: " + ". ".join(summary_sentences) + "."
'''
    
    integration_code = '''
# Integrate new tools with the existing agent:

from langchain.agents import create_tool_calling_agent, AgentExecutor

# Combine existing tools with new ones
extended_tools = [
    extract_dates, 
    extract_monetary_values, 
    extract_percentages, 
    analyze_sentiment,
    extract_entities,      # New tool
    summarize_section      # New tool
]

# Create extended agent
extended_agent = create_tool_calling_agent(llm, extended_tools, agent_prompt)
extended_executor = AgentExecutor(
    agent=extended_agent, 
    tools=extended_tools, 
    verbose=True
)

# Test with a document
test_result = extended_executor.invoke({
    "input": """Analyze this document and provide:
    1. All entities (organizations, locations, products)
    2. Key dates and monetary values
    3. A brief summary
    
    [Your document text here]"""
})
'''
    
    return {
        "tool_implementation": tool_implementation,
        "integration_code": integration_code,
        "notes": """
The extract_entities tool uses pattern matching for demonstration.
In production, consider using:
- spaCy NER models
- Hugging Face transformers for entity recognition
- Custom fine-tuned models for domain-specific entities

The summarize_section tool uses simple extractive summarization.
For better results, consider:
- LLM-based abstractive summarization
- TextRank or similar algorithms
- Fine-tuned summarization models
"""
    }


# =============================================================================
# EXERCISE 3.1 SOLUTION: Create a Custom Report Template
# =============================================================================

def exercise_3_1_solution():
    """
    Solution for Exercise 3.1: Custom Report Template
    
    This solution creates a monthly performance dashboard template.
    """
    
    MONTHLY_DASHBOARD_TEMPLATE = """
# Monthly Performance Dashboard

## {month} {year}
**Client:** {client}
**Prepared by:** Campaign Analytics Team
**Date Generated:** {report_date}

---

## Executive Overview

{executive_summary}

---

## Campaign Performance at a Glance

| Metric | This Month | Last Month | Change | Target | Status |
|--------|------------|------------|--------|--------|--------|
| Reach | {reach_current}% | {reach_previous}% | {reach_change:+.1f}% | {reach_target}% | {reach_status} |
| Frequency | {freq_current} | {freq_previous} | {freq_change:+.1f} | {freq_target} | {freq_status} |
| Impressions | {impressions_current:,} | {impressions_previous:,} | {impressions_change:+.1f}% | - | - |
| GRP | {grp_current:.1f} | {grp_previous:.1f} | {grp_change:+.1f} | {grp_target} | {grp_status} |

---

## Budget Status

**Monthly Budget:** {currency} {budget_monthly:,.0f}
**Spent to Date:** {currency} {budget_spent:,.0f}
**Remaining:** {currency} {budget_remaining:,.0f}
**Utilization:** {budget_utilization:.1f}%

### Spend by Channel

| Channel | Allocated | Spent | Utilization |
|---------|-----------|-------|-------------|
{channel_breakdown}

---

## Key Insights

{key_insights}

---

## Recommendations

{recommendations}

---

## Next Month Outlook

{outlook}

---

*Report generated automatically. For questions, contact analytics@company.com*
"""
    
    sample_data = {
        "month": "November",
        "year": 2024,
        "client": "TechCorp Italia",
        "report_date": "2024-12-01",
        "reach_current": 52.3,
        "reach_previous": 48.1,
        "reach_change": 4.2,
        "reach_target": 50.0,
        "reach_status": "On Target",
        "freq_current": 4.1,
        "freq_previous": 3.8,
        "freq_change": 0.3,
        "freq_target": 4.0,
        "freq_status": "On Target",
        "impressions_current": 6200000,
        "impressions_previous": 5800000,
        "impressions_change": 6.9,
        "grp_current": 214.4,
        "grp_previous": 182.8,
        "grp_change": 31.6,
        "grp_target": 200.0,
        "grp_status": "Exceeding",
        "currency": "EUR",
        "budget_monthly": 65000,
        "budget_spent": 58500,
        "budget_remaining": 6500,
        "budget_utilization": 90.0,
        "channel_breakdown": """| TV | 35,000 | 32,500 | 92.9% |
| Digital | 20,000 | 18,200 | 91.0% |
| OOH | 10,000 | 7,800 | 78.0% |""",
        "executive_summary": "November showed strong performance across all key metrics, with reach exceeding target by 2.3 percentage points. The campaign is on track for Q4 objectives.",
        "key_insights": """1. TV continues to be the primary reach driver, contributing 45% of total reach
2. Digital efficiency improved with CPM down 8% month-over-month
3. OOH underperformance due to weather-related visibility issues
4. Young adult segment (18-24) showed strongest engagement growth""",
        "recommendations": """1. Reallocate 10% of OOH budget to digital for December
2. Increase frequency in evening prime time slots
3. Consider extending campaign into early January for holiday momentum
4. Test new creative variants in digital channels""",
        "outlook": "December forecast indicates potential for 55% reach with current budget. Holiday programming provides premium inventory opportunities. Recommend early booking for New Year's Eve special placements."
    }
    
    implementation_code = '''
# Add the template to TemplateReportGenerator:

template_generator = TemplateReportGenerator(provider="groq")
template_generator.TEMPLATES["monthly_dashboard"] = MONTHLY_DASHBOARD_TEMPLATE

# Generate the report:
dashboard = template_generator.generate(
    template_name="monthly_dashboard",
    data=sample_data
)

print(dashboard)
'''
    
    return {
        "template": MONTHLY_DASHBOARD_TEMPLATE,
        "sample_data": sample_data,
        "implementation_code": implementation_code,
        "notes": """
Key features of this template:
1. Clear visual hierarchy with headers and sections
2. Tabular data for easy comparison
3. Month-over-month change indicators
4. Budget tracking with channel breakdown
5. LLM-generated narrative sections (insights, recommendations, outlook)

To make this template dynamic with LLM generation:
- Use _generate_analysis() for executive_summary
- Use _generate_recommendations() for recommendations
- Add a new _generate_outlook() method for the outlook section
"""
    }


# =============================================================================
# EXERCISE 3.2 SOLUTION: Build a Complete Report Generation System
# =============================================================================

def exercise_3_2_solution():
    """
    Solution for Exercise 3.2: Enhanced Report Generation Agent
    
    This solution adds trend analysis and forecasting tools to create
    a comprehensive report generation system.
    """
    
    tool_implementations = '''
from langchain_core.tools import tool
import json

# Additional campaign data for trend analysis
HISTORICAL_DATA = {
    "Q1_2024": {"reach": 42.1, "frequency": 3.5, "impressions": 4200000, "budget_spent": 145000},
    "Q2_2024": {"reach": 44.5, "frequency": 3.6, "impressions": 4800000, "budget_spent": 155000},
    "Q3_2024": {"reach": 45.2, "frequency": 3.8, "impressions": 5100000, "budget_spent": 148500},
    "Q4_2024": {"reach": 52.3, "frequency": 4.1, "impressions": 6200000, "budget_spent": 185000}
}


@tool
def analyze_performance_trend(metric: str) -> str:
    """
    Analyze the trend of a specific metric over the last 4 quarters.
    Available metrics: reach, frequency, impressions, budget_spent
    Use this to understand historical performance patterns.
    """
    if metric not in ["reach", "frequency", "impressions", "budget_spent"]:
        return f"Invalid metric. Choose from: reach, frequency, impressions, budget_spent"
    
    values = [HISTORICAL_DATA[q][metric] for q in ["Q1_2024", "Q2_2024", "Q3_2024", "Q4_2024"]]
    quarters = ["Q1_2024", "Q2_2024", "Q3_2024", "Q4_2024"]
    
    # Calculate trend
    total_change = values[-1] - values[0]
    avg_quarterly_change = total_change / 3
    
    # Calculate growth rate
    growth_rate = ((values[-1] / values[0]) - 1) * 100 if values[0] > 0 else 0
    
    # Determine trend direction
    if avg_quarterly_change > 0:
        trend = "upward"
    elif avg_quarterly_change < 0:
        trend = "downward"
    else:
        trend = "stable"
    
    result = f"""Trend Analysis for {metric.upper()}:

Historical Values:
"""
    for q, v in zip(quarters, values):
        if metric == "impressions" or metric == "budget_spent":
            result += f"  - {q}: {v:,.0f}\\n"
        else:
            result += f"  - {q}: {v}\\n"
    
    result += f"""
Trend Direction: {trend.upper()}
Total Change (Q1 to Q4): {total_change:,.2f}
Average Quarterly Change: {avg_quarterly_change:,.2f}
Overall Growth Rate: {growth_rate:.1f}%
"""
    return result


@tool
def generate_forecast(metric: str, quarters_ahead: int = 2) -> str:
    """
    Generate a simple forecast for a metric based on historical trends.
    Specify the metric and how many quarters ahead to forecast (default: 2).
    Available metrics: reach, frequency, impressions, budget_spent
    """
    if metric not in ["reach", "frequency", "impressions", "budget_spent"]:
        return f"Invalid metric. Choose from: reach, frequency, impressions, budget_spent"
    
    if quarters_ahead < 1 or quarters_ahead > 4:
        return "Please specify quarters_ahead between 1 and 4."
    
    values = [HISTORICAL_DATA[q][metric] for q in ["Q1_2024", "Q2_2024", "Q3_2024", "Q4_2024"]]
    
    # Simple linear trend extrapolation
    avg_quarterly_change = (values[-1] - values[0]) / 3
    
    forecasts = []
    last_value = values[-1]
    forecast_quarters = ["Q1_2025", "Q2_2025", "Q3_2025", "Q4_2025"]
    
    for i in range(quarters_ahead):
        forecast_value = last_value + (avg_quarterly_change * (i + 1))
        # Apply bounds (reach can not exceed 100%, values can not be negative)
        if metric == "reach":
            forecast_value = min(100, max(0, forecast_value))
        else:
            forecast_value = max(0, forecast_value)
        forecasts.append((forecast_quarters[i], forecast_value))
    
    result = f"""Forecast for {metric.upper()}:

Last Known Value (Q4_2024): {values[-1]:,.2f}
Forecasting Method: Linear Trend Extrapolation

Projected Values:
"""
    for q, v in forecasts:
        if metric == "impressions" or metric == "budget_spent":
            result += f"  - {q}: {v:,.0f}\\n"
        else:
            result += f"  - {q}: {v:.2f}\\n"
    
    result += """
Note: Forecast based on historical trend. Actual results may vary based on
market conditions, budget changes, and campaign strategy adjustments.
"""
    return result


@tool
def suggest_visualizations(report_type: str) -> str:
    """
    Suggest appropriate visualizations for a report type.
    Report types: executive, detailed, trend_analysis, comparison
    Use this when creating reports that would benefit from charts.
    """
    visualizations = {
        "executive": [
            {"type": "KPI Cards", "description": "Large format cards showing key metrics with targets"},
            {"type": "Gauge Charts", "description": "Show progress toward reach and frequency goals"},
            {"type": "Donut Chart", "description": "Budget allocation by channel"},
        ],
        "detailed": [
            {"type": "Line Chart", "description": "Daily/weekly reach accumulation over campaign period"},
            {"type": "Stacked Bar Chart", "description": "Impressions by channel over time"},
            {"type": "Scatter Plot", "description": "Reach vs. Frequency correlation"},
            {"type": "Heatmap", "description": "Performance by day of week and daypart"},
            {"type": "Waterfall Chart", "description": "Budget breakdown showing flow from allocation to spend"},
        ],
        "trend_analysis": [
            {"type": "Multi-line Chart", "description": "Quarter-over-quarter metric trends"},
            {"type": "Area Chart", "description": "Cumulative performance over time"},
            {"type": "Sparklines", "description": "Inline trend indicators for each metric"},
        ],
        "comparison": [
            {"type": "Grouped Bar Chart", "description": "Side-by-side campaign metric comparison"},
            {"type": "Radar Chart", "description": "Multi-dimensional campaign performance comparison"},
            {"type": "Bullet Charts", "description": "Actual vs target with tolerance ranges"},
        ]
    }
    
    if report_type not in visualizations:
        return f"Unknown report type. Choose from: {list(visualizations.keys())}"
    
    result = f"Recommended Visualizations for {report_type.upper()} Report:\\n\\n"
    for viz in visualizations[report_type]:
        result += f"- {viz['type']}: {viz['description']}\\n"
    
    return result
'''
    
    agent_creation_code = '''
# Create enhanced report generation agent:

enhanced_report_tools = [
    get_campaign_data,
    calculate_campaign_kpis,
    compare_campaigns,
    format_report_section,
    analyze_performance_trend,   # New tool
    generate_forecast,           # New tool
    suggest_visualizations       # New tool
]

enhanced_report_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an advanced report generation assistant for advertising campaigns.

Your enhanced capabilities:
1. Retrieve campaign data from the database
2. Calculate performance KPIs
3. Compare campaigns
4. Analyze historical trends
5. Generate forecasts based on past performance
6. Suggest appropriate visualizations
7. Format professional report sections

When generating comprehensive reports:
- Start with current campaign data
- Include trend analysis to show historical context
- Add forecasts when discussing future outlook
- Suggest visualizations to enhance presentation
- Structure output professionally with clear sections

Always provide data-driven insights and actionable recommendations."""),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

enhanced_report_agent = create_tool_calling_agent(
    llm, 
    enhanced_report_tools, 
    enhanced_report_prompt
)

enhanced_report_executor = AgentExecutor(
    agent=enhanced_report_agent, 
    tools=enhanced_report_tools, 
    verbose=True,
    max_iterations=10  # Allow more iterations for complex reports
)
'''
    
    test_query = '''
# Test the enhanced agent with a comprehensive report request:

result = enhanced_report_executor.invoke({
    "input": """Generate a comprehensive annual review report for our advertising campaigns that includes:

1. Current Q4_2024 performance summary with KPIs
2. Trend analysis for reach, frequency, and impressions over all quarters
3. Comparison between Q3_2024 and Q4_2024 performance
4. Forecast for Q1_2025 and Q2_2025
5. Suggested visualizations for presenting this report
6. Strategic recommendations for the upcoming year

Format this as a professional annual review document."""
})

print(result["output"])
'''
    
    return {
        "tool_implementations": tool_implementations,
        "agent_creation_code": agent_creation_code,
        "test_query": test_query,
        "notes": """
The enhanced report generation system adds three powerful capabilities:

1. TREND ANALYSIS (analyze_performance_trend)
   - Examines historical data across quarters
   - Calculates growth rates and change patterns
   - Identifies upward, downward, or stable trends

2. FORECASTING (generate_forecast)
   - Projects future values based on historical trends
   - Uses linear trend extrapolation (simple but effective)
   - Includes appropriate caveats about forecast uncertainty

3. VISUALIZATION SUGGESTIONS (suggest_visualizations)
   - Recommends appropriate chart types for different reports
   - Tailored suggestions based on report purpose
   - Helps create more impactful presentations

For production use, consider:
- More sophisticated forecasting models (ARIMA, Prophet)
- Integration with actual visualization libraries (matplotlib, plotly)
- Caching of frequently requested reports
- User preference storage for customization
"""
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Display all solutions with explanations."""
    
    print("=" * 80)
    print("LESSON 3: PRACTICAL LLM APPLICATIONS - SOLUTIONS")
    print("=" * 80)
    
    # Exercise 1.1
    print("\n\n" + "=" * 80)
    print("EXERCISE 1.1: Custom Conversational Assistant")
    print("=" * 80)
    sol_1_1 = exercise_1_1_solution()
    print("\nSystem Prompt:")
    print(sol_1_1["system_prompt"])
    print("\nImplementation Notes:")
    print(sol_1_1["implementation_notes"])
    
    # Exercise 2.1
    print("\n\n" + "=" * 80)
    print("EXERCISE 2.1: Custom Document Extractor")
    print("=" * 80)
    sol_2_1 = exercise_2_1_solution()
    print("\nSchema:")
    print(json.dumps(sol_2_1["schema"], indent=2))
    print("\nExpected Extraction:")
    print(json.dumps(sol_2_1["expected_extraction"], indent=2))
    
    # Exercise 2.2
    print("\n\n" + "=" * 80)
    print("EXERCISE 2.2: Extended Document Analysis Agent")
    print("=" * 80)
    sol_2_2 = exercise_2_2_solution()
    print("\nTool Implementation:")
    print(sol_2_2["tool_implementation"][:500] + "...")
    print("\nNotes:")
    print(sol_2_2["notes"])
    
    # Exercise 3.1
    print("\n\n" + "=" * 80)
    print("EXERCISE 3.1: Custom Report Template")
    print("=" * 80)
    sol_3_1 = exercise_3_1_solution()
    print("\nTemplate Preview (first 500 chars):")
    print(sol_3_1["template"][:500] + "...")
    print("\nNotes:")
    print(sol_3_1["notes"])
    
    # Exercise 3.2
    print("\n\n" + "=" * 80)
    print("EXERCISE 3.2: Enhanced Report Generation System")
    print("=" * 80)
    sol_3_2 = exercise_3_2_solution()
    print("\nNotes:")
    print(sol_3_2["notes"])
    
    print("\n\n" + "=" * 80)
    print("All solutions demonstrated. Import individual solutions for full details.")
    print("=" * 80)


if __name__ == "__main__":
    main()
