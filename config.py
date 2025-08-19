"""
Configuration settings for the Mondelez Market Analysis System.
This file contains default values that can be adjusted as needed.
"""

# Data generation configuration
DATA_CONFIG = {
    'start_date': '2025-01-01',
    'periods': 7,
    'brands': ["Oreo", "ChipsAhoy", "Ritz", "belVita", "NutterButter"],
    'regions': ["Northeast", "Southeast", "Midwest", "West", "Southwest"],
    'age_groups': ["18-24", "25-34", "35-44", "45-54", "55+"],
}

# LLM configuration
LLM_CONFIG = {
    'default_model': 'llama3-8b-8192',
    'temperature': 0.2,
    'max_tokens': 2000,
}

# Tool configuration
TOOL_CONFIG = {
    'forecasting': {
        'default_horizon': 3,
        'default_arima_order': (1, 1, 0),
        'fallback_to_simple_forecast': True,
    }
}

# Query parsing configuration
QUERY_PARSER_CONFIG = {
    'default_brand': 'oreo',
    'default_metric': 'market_share',
}
