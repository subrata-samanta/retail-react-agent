import pandas as pd
import numpy as np
from datetime import datetime

def generate_penetration_data(start_date='2025-01-01', periods=7, 
                             brands=None, age_groups=None, regions=None, seed=43):
    """
    Generate synthetic penetration data for specified Mondelez brands, age groups and regions.
    
    Parameters:
    - start_date (str): Starting date for data generation
    - periods (int): Number of months to generate data for
    - brands (list): List of brands to include
    - age_groups (list): List of age groups to include
    - regions (list): List of regions to include
    - seed (int): Random seed for reproducibility
    
    Returns:
    - pd.DataFrame: Penetration dataset
    """
    np.random.seed(seed)  # For reproducibility
    
    # Default values if not provided
    if brands is None:
        brands = ["Oreo", "ChipsAhoy", "Ritz", "belVita", "NutterButter"]
    if age_groups is None:
        age_groups = ["18-24", "25-34", "35-44", "45-54", "55+"]

    regions = ["Northeast", "Southeast", "Midwest", "West", "Southwest"]
    
    # Convert start_date to datetime if it's a string
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    
    # Generate months
    months = pd.date_range(start=start_date, periods=periods, freq='M')
    
    # Base values for each brand (customized for Mondelez)
    base_values = {
        "Oreo": 12.0,         # High penetration
        "ChipsAhoy": 9.5,     # Good penetration
        "Ritz": 10.0,         # Strong household presence
        "belVita": 7.0,       # Growing penetration
        "NutterButter": 6.0   # Niche but loyal following
    }
    
    # Generate age group preferences for Mondelez brands
    age_multipliers = {
        "18-24": {"Oreo": 1.3, "ChipsAhoy": 1.2, "Ritz": 0.9, "belVita": 0.8, "NutterButter": 0.9},
        "25-34": {"Oreo": 1.2, "ChipsAhoy": 1.1, "Ritz": 1.0, "belVita": 1.2, "NutterButter": 0.9},
        "35-44": {"Oreo": 1.1, "ChipsAhoy": 1.0, "Ritz": 1.1, "belVita": 1.3, "NutterButter": 1.0},
        "45-54": {"Oreo": 0.9, "ChipsAhay": 0.9, "Ritz": 1.2, "belVita": 1.1, "NutterButter": 1.1},
        "55+": {"Oreo": 0.8, "ChipsAhoy": 0.8, "Ritz": 1.1, "belVita": 0.9, "NutterButter": 1.2}
    }
    
    # Region preferences (multipliers)
    region_multipliers = {region: 1.0 + (i-len(regions)/2)*0.05 
                         for i, region in enumerate(regions)}
    
    data = []
    
    # Generate data for each combination
    for month_idx, month in enumerate(months):
        month_str = month.strftime('%Y-%m')
        
        for brand in brands:
            base_pen = base_values.get(brand, 5.0)
            brand_trend = 0.3 if brand == brands[0] else 0.1  # First brand growing faster
            
            for age_group in age_groups:
                age_mult = age_multipliers.get(age_group, {}).get(brand, 1.0)
                
                for region in regions:
                    region_mult = region_multipliers.get(region, 1.0)
                    
                    # Calculate penetration with trends and adjustments
                    penetration = (base_pen + month_idx * brand_trend) * age_mult * region_mult + np.random.normal(0, 0.2)
                    
                    # Add purchase frequency data
                    purch_freq = 2.0 + (0.5 if brand == brands[0] else 0) + np.random.normal(0, 0.2)
                    
                    # Add loyalty score
                    loyalty = 65 + (10 if brand == brands[0] else 0) + np.random.normal(0, 3)
                    
                    data.append({
                        "Month": month_str,
                        "Brand": brand,
                        "AgeGroup": age_group,
                        "Region": region,
                        "Penetration": round(max(penetration, 0.1), 1),
                        "PurchaseFrequency": round(purch_freq, 1),
                        "LoyaltyScore": round(loyalty, 1)
                    })
    
    return pd.DataFrame(data)

def get_oreo_penetration():
    """Legacy function for backward compatibility"""
    return generate_penetration_data()
