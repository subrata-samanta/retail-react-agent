import pandas as pd
import numpy as np
from datetime import datetime

def generate_market_share_data(start_date='2025-01-01', periods=7, 
                              brands=None, regions=None, seed=42):
    """
    Generate synthetic market share data for specified Mondelez brands and regions.
    
    Parameters:
    - start_date (str): Starting date for data generation
    - periods (int): Number of months to generate data for
    - brands (list): List of brands to include
    - regions (list): List of regions to include
    - seed (int): Random seed for reproducibility
    
    Returns:
    - pd.DataFrame: Market share dataset
    """
    np.random.seed(seed)  # For reproducibility
    
    # Default Mondelez brands if not provided
    if brands is None:
        brands = ["Oreo", "ChipsAhoy", "Ritz", "belVita", "NutterButter"]
    if regions is None:
        regions = ["Northeast", "Southeast", "Midwest", "West", "Southwest"]
    
    # Convert start_date to datetime if it's a string
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    
    # Generate months
    months = pd.date_range(start=start_date, periods=periods, freq='M')
    
    # Base values and trends for each brand (customized for Mondelez brands)
    base_values = {
        "Oreo": 24.0,        # Market leader
        "ChipsAhoy": 18.5,   # Strong performer
        "Ritz": 16.0,        # Popular cracker brand
        "belVita": 12.5,     # Growing breakfast biscuit
        "NutterButter": 10.0 # Classic peanut butter cookie
    }
    
    trends = {
        "Oreo": 0.4,         # Strong growth
        "ChipsAhoy": 0.2,    # Moderate growth
        "Ritz": 0.25,        # Steady growth
        "belVita": 0.45,     # Fast growing health-oriented brand
        "NutterButter": 0.1  # Slower growth for mature brand
    }
    
    # Region multipliers to create regional differences
    region_multipliers = {region: 1.0 + (i-len(regions)/2)*0.05 
                         for i, region in enumerate(regions)}
    
    data = []
    
    # Generate data for each combination
    for month_idx, month in enumerate(months):
        month_str = month.strftime('%Y-%m')
        
        for brand in brands:
            base_share = base_values.get(brand, 10.0)
            trend = trends.get(brand, 0.1)
            
            for region in regions:
                multiplier = region_multipliers.get(region, 1.0)
                
                # Calculate market share with trend, regional adjustment and noise
                share = (base_share + month_idx * trend) * multiplier + np.random.normal(0, 0.3)
                
                # Add price data
                premium = 0.5 if brand in brands[:2] else 0  # First two brands are premium
                price = 3.99 + premium + np.random.normal(0, 0.1)
                
                # Add promotion flag
                promo = 1 if np.random.rand() < 0.3 else 0
                
                data.append({
                    "Month": month_str,
                    "Brand": brand,
                    "Region": region,
                    "MarketShare": round(max(share, 0.1), 1),
                    "Price": round(price, 2),
                    "OnPromotion": promo
                })
    
    return pd.DataFrame(data)

def get_oreo_market_share():
    """Legacy function for backward compatibility"""
    return generate_market_share_data()
