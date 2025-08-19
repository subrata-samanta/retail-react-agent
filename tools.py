from langchain.tools import BaseTool
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
import io
import base64

# Import data generator functions instead of specific dataset functions
from data_ci_market_share import generate_market_share_data
from data_panel_penetration import generate_penetration_data

class MarketShareTool(BaseTool):
    name = "market_share_tool"
    description = "Retrieves and analyzes market share data. Specify brand, region (optional), and time period."
    
    def _run(self, query: str = ""):
        df = generate_market_share_data()
        
        # Parse query for filters
        query = query.lower()
        brand_filter = self._extract_brand(query)
        region_filter = self._extract_region(query)
        
        # Filter data
        filtered_df = df[df['Brand'].str.lower() == brand_filter]
        if region_filter:
            filtered_df = filtered_df[filtered_df['Region'] == region_filter]
        
        # Aggregate if necessary
        if region_filter is None:
            filtered_df = filtered_df.groupby(['Month', 'Brand'], as_index=False).agg({
                'MarketShare': 'mean',
                'Price': 'mean',
                'OnPromotion': 'sum'
            })
        
        return filtered_df.to_string()
    
    def _extract_brand(self, query):
        brands = ["oreo", "chipsahoy", "ritz", "belvita", "nutterbutter"]
        for brand in brands:
            if brand in query:
                return brand
        return "oreo"  # default
    
    def _extract_region(self, query):
        regions = ["northeast", "southeast", "midwest", "west", "southwest"]
        for region in regions:
            if region in query:
                return region.capitalize()
        return None

class PenetrationTool(BaseTool):
    name = "penetration_tool"
    description = "Retrieves and analyzes penetration data. Specify brand, age group (optional), region (optional), and time period."
    
    def _run(self, query: str = ""):
        df = generate_penetration_data()
        
        # Parse query for filters
        query = query.lower()
        brand_filter = self._extract_brand(query)
        region_filter = self._extract_region(query)
        age_filter = self._extract_age_group(query)
        
        # Filter data
        filtered_df = df[df['Brand'].str.lower() == brand_filter]
        if region_filter:
            filtered_df = filtered_df[filtered_df['Region'] == region_filter]
        if age_filter:
            filtered_df = filtered_df[filtered_df['AgeGroup'] == age_filter]
        
        # Aggregate if necessary
        if region_filter is None and age_filter is None:
            filtered_df = filtered_df.groupby(['Month', 'Brand'], as_index=False).agg({
                'Penetration': 'mean',
                'PurchaseFrequency': 'mean',
                'LoyaltyScore': 'mean'
            })
        
        return filtered_df.to_string()
    
    def _extract_brand(self, query):
        brands = ["oreo", "chipsahoy", "ritz", "belvita", "nutterbutter"]
        for brand in brands:
            if brand in query:
                return brand
        return "oreo"  # default
    
    def _extract_region(self, query):
        regions = ["northeast", "southeast", "midwest", "west", "southwest"]
        for region in regions:
            if region in query:
                return region.capitalize()
        return None
    
    def _extract_age_group(self, query):
        age_groups = ["18-24", "25-34", "35-44", "45-54", "55+""]
        for age in age_groups:
            if age in query:
                return age
        return None

class ComparisonTool(BaseTool):
    name = "comparison_tool"
    description = "Compares market share and penetration data with advanced analytics. Specify brands, regions, age groups, or metrics for comparison."
    
    def _run(self, query: str = ""):
        # Get data
        ms_df = generate_market_share_data()
        pen_df = generate_penetration_data()
        
        # Extract parameters from query
        brand = self._extract_brand(query)
        region = self._extract_region(query)
        age_group = self._extract_age_group(query)
        
        # Filter and prepare data
        ms_filtered = ms_df[ms_df['Brand'].str.lower() == brand]
        pen_filtered = pen_df[pen_df['Brand'].str.lower() == brand]
        
        if region:
            ms_filtered = ms_filtered[ms_filtered['Region'] == region]
            pen_filtered = pen_filtered[pen_filtered['Region'] == region]
        
        if age_group:
            pen_filtered = pen_filtered[pen_filtered['AgeGroup'] == age_group]
        
        # Aggregate data
        ms_agg = ms_filtered.groupby('Month', as_index=False)['MarketShare'].mean()
        
        if age_group:
            pen_agg = pen_filtered.groupby('Month', as_index=False)['Penetration'].mean()
        else:
            pen_agg = pen_filtered.groupby(['Month'], as_index=False)['Penetration'].mean()
        
        # Merge datasets
        merged_df = pd.merge(ms_agg, pen_agg, on='Month')
        
        # Calculate correlation and perform analysis
        correlation = merged_df["MarketShare"].corr(merged_df["Penetration"])
        
        # Simple regression analysis
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            merged_df["MarketShare"], merged_df["Penetration"]
        )
        
        # Create analysis text
        title = f"{brand.capitalize()} Market Share vs Penetration Analysis"
        if region:
            title += f" in {region}"
        if age_group:
            title += f" for age group {age_group}"
            
        analysis = f"""
        {title}:
        
        1. Data Summary:
           {merged_df.to_string()}
        
        2. Statistical Analysis:
           - Correlation coefficient: {correlation:.3f}
           - Regression slope: {slope:.3f}
           - R-squared: {r_value**2:.3f}
           - p-value: {p_value:.4f}
           
        3. Trend Analysis:
           - Market Share trend: {self._get_trend(merged_df['MarketShare'])}
           - Penetration trend: {self._get_trend(merged_df['Penetration'])}
           
        4. Monthly Growth Analysis:
           {self._calculate_growth_rates(merged_df)}
           
        5. Insight:
           {self._generate_insight(correlation, merged_df, brand, region, age_group)}
        """
        
        return analysis
    
    def _extract_brand(self, query):
        brands = ["oreo", "chipsahoy", "ritz", "belvita", "nutterbutter"]
        for brand in brands:
            if brand in query.lower():
                return brand
        return "oreo"  # default
    
    def _extract_region(self, query):
        regions = ["northeast", "southeast", "midwest", "west", "southwest"]
        for region in regions:
            if region in query.lower():
                return region.capitalize()
        return None
    
    def _extract_age_group(self, query):
        age_groups = ["18-24", "25-34", "35-44", "45-54", "55+""]
        for age in age_groups:
            if age in query:
                return age
        return None
    
    def _get_trend(self, series):
        if series.iloc[-1] > series.iloc[0]:
            return f"Increasing over the period ({series.iloc[0]:.1f}% to {series.iloc[-1]:.1f}%)"
        else:
            return f"Decreasing over the period ({series.iloc[0]:.1f}% to {series.iloc[-1]:.1f}%)"
    
    def _calculate_growth_rates(self, df):
        ms_growth = [(df['MarketShare'].iloc[i] - df['MarketShare'].iloc[i-1]) / df['MarketShare'].iloc[i-1] * 100 
                     for i in range(1, len(df))]
        pen_growth = [(df['Penetration'].iloc[i] - df['Penetration'].iloc[i-1]) / df['Penetration'].iloc[i-1] * 100 
                      for i in range(1, len(df))]
        
        result = ""
        for i in range(len(ms_growth)):
            result += f"Month {df['Month'].iloc[i+1]}: Market Share growth: {ms_growth[i]:.1f}%, Penetration growth: {pen_growth[i]:.1f}%\n"
        
        return result
    
    def _generate_insight(self, correlation, df, brand, region, age_group):
        insights = []
        
        # Correlation insight
        if correlation > 0.8:
            insights.append(f"Strong positive correlation suggests that {brand.capitalize()}'s market share and penetration move closely together.")
        elif correlation > 0.5:
            insights.append(f"Moderate positive correlation suggests some relationship between market share and penetration.")
        elif correlation > -0.3:
            insights.append(f"Weak correlation suggests that market share and penetration may be influenced by different factors.")
        else:
            insights.append(f"Negative correlation suggests that as penetration increases, market share actually decreases, which might indicate declining purchase frequency among buyers.")
            
        # Trend comparison
        ms_trend = df['MarketShare'].iloc[-1] - df['MarketShare'].iloc[0]
        pen_trend = df['Penetration'].iloc[-1] - df['Penetration'].iloc[0]
        
        if ms_trend > 0 and pen_trend > 0:
            insights.append(f"{brand.capitalize()} is growing both in penetration and market share, indicating successful consumer acquisition and retention.")
        elif ms_trend > 0 and pen_trend <= 0:
            insights.append(f"Market share is growing while penetration is flat or declining, suggesting existing consumers are buying more.")
        elif ms_trend <= 0 and pen_trend > 0:
            insights.append(f"Penetration is growing but market share isn't, suggesting new consumers buy less than average.")
        else:
            insights.append(f"Both metrics are declining, suggesting {brand.capitalize()} may be losing ground to competitors.")
        
        # Demographic/regional specific insights
        if age_group:
            insights.append(f"Analysis for the {age_group} age group shows distinct patterns that may require targeted strategies.")
        
        if region:
            insights.append(f"Regional analysis for {region} reveals market dynamics that differ from the national average.")
        
        return " ".join(insights)

class CompetitorAnalysisTool(BaseTool):
    name = "competitor_analysis_tool"
    description = "Analyzes and compares multiple brands' performance. Specify brands, metrics (market share, penetration, etc.), and regions for comparison."
    
    def _run(self, query: str = ""):
        # Get data
        ms_df = generate_market_share_data()
        pen_df = generate_penetration_data()
        
        # Extract parameters from query
        brands_to_compare = self._extract_brands(query)
        region = self._extract_region(query)
        
        # Filter by region if specified
        if region:
            ms_df = ms_df[ms_df['Region'] == region]
            pen_df = pen_df[pen_df['Region'] == region]
        
        # Create analysis based on market share trends
        ms_analysis = self._analyze_market_shares(ms_df, brands_to_compare)
        
        # Create analysis based on penetration
        pen_analysis = self._analyze_penetration(pen_df, brands_to_compare)
        
        # Combined analysis
        title = "Competitor Analysis"
        if region:
            title += f" for {region} region"
            
        analysis = f"""
        {title}:
        
        1. Market Share Analysis:
        {ms_analysis}
        
        2. Penetration Analysis:
        {pen_analysis}
        
        3. Competitive Landscape Overview:
        {self._create_competitive_landscape(ms_df, pen_df, brands_to_compare)}
        """
        
        return analysis
    
    def _extract_brands(self, query):
        brands = []
        all_brands = ["oreo", "chipsahoy", "ritz", "belvita", "nutterbutter"]
        
        for brand in all_brands:
            if brand in query.lower():
                brands.append(brand)
        
        # If no brands specified, use all
        if not brands:
            brands = all_brands
            
        return brands
    
    def _extract_region(self, query):
        regions = ["northeast", "southeast", "midwest", "west", "southwest"]
        for region in regions:
            if region in query.lower():
                return region.capitalize()
        return None
    
    def _analyze_market_shares(self, df, brands):
        result = ""
        
        for brand in brands:
            brand_df = df[df['Brand'].str.lower() == brand].groupby('Month', as_index=False)['MarketShare'].mean()
            
            if len(brand_df) > 0:
                start_share = brand_df['MarketShare'].iloc[0]
                end_share = brand_df['MarketShare'].iloc[-1]
                change = end_share - start_share
                change_pct = (change / start_share) * 100 if start_share > 0 else 0
                
                result += f"- {brand.capitalize()}: Started at {start_share:.1f}%, ended at {end_share:.1f}% "
                result += f"({'+' if change >= 0 else ''}{change:.1f} points, {'+' if change_pct >= 0 else ''}{change_pct:.1f}%)\n"
        
        return result
    
    def _analyze_penetration(self, df, brands):
        result = ""
        
        for brand in brands:
            brand_df = df[df['Brand'].str.lower() == brand].groupby('Month', as_index=False)['Penetration'].mean()
            
            if len(brand_df) > 0:
                start_pen = brand_df['Penetration'].iloc[0]
                end_pen = brand_df['Penetration'].iloc[-1]
                change = end_pen - start_pen
                change_pct = (change / start_pen) * 100 if start_pen > 0 else 0
                
                result += f"- {brand.capitalize()}: Started at {start_pen:.1f}%, ended at {end_pen:.1f}% "
                result += f"({'+' if change >= 0 else ''}{change:.1f} points, {'+' if change_pct >= 0 else ''}{change_pct:.1f}%)\n"
        
        return result
    
    def _create_competitive_landscape(self, ms_df, pen_df, brands):
        # Aggregate data for the latest month
        latest_month = ms_df['Month'].max()
        
        ms_latest = ms_df[ms_df['Month'] == latest_month]
        ms_summary = ms_latest.groupby('Brand', as_index=False)['MarketShare'].mean()
        
        pen_latest = pen_df[pen_df['Month'] == latest_month]
        pen_summary = pen_latest.groupby('Brand', as_index=False)['Penetration'].mean()
        
        # Merge datasets
        combined = pd.merge(ms_summary, pen_summary, on='Brand')
        combined = combined[combined['Brand'].str.lower().isin([b.lower() for b in brands])]
        
        # Calculate share of market
        total_share = combined['MarketShare'].sum()
        combined['ShareOfMarket'] = combined['MarketShare'] / total_share * 100
        
        # Sort by market share
        combined = combined.sort_values('MarketShare', ascending=False)
        
        result = "Current competitive standing (based on latest data):\n"
        for _, row in combined.iterrows():
            result += f"- {row['Brand']}: {row['MarketShare']:.1f}% market share ({row['ShareOfMarket']:.1f}% of measured brands), "
            result += f"{row['Penetration']:.1f}% penetration\n"
            
        return result

class ForecastingTool(BaseTool):
    name = "forecasting_tool"
    description = "Forecasts future market share or penetration based on current trends. Specify brand, metric (market share or penetration), and time horizon."
    
    def _run(self, query: str = ""):
        # Parse query for parameters
        query = query.lower()
        metric = self._extract_metric(query)
        brand = self._extract_brand(query)
        horizon = self._extract_forecast_horizon(query)
        
        # Get appropriate data
        if metric == "market_share":
            df = generate_market_share_data()
            df_filtered = df[df['Brand'].str.lower() == brand]
            df_agg = df_filtered.groupby('Month', as_index=False)['MarketShare'].mean()
            metric_name = "Market Share"
            values = df_agg['MarketShare'].values
        else:
            df = generate_penetration_data()
            df_filtered = df[df['Brand'].str.lower() == brand]
            df_agg = df_filtered.groupby('Month', as_index=False)['Penetration'].mean()
            metric_name = "Penetration"
            values = df_agg['Penetration'].values
        
        # Perform forecasting
        forecast = self._generate_forecast(values, horizon)
        
        # Prepare forecast data
        last_month = pd.to_datetime(df_agg['Month'].iloc[-1])
        forecast_months = pd.date_range(start=last_month + pd.DateOffset(months=1), periods=horizon, freq='M')
        forecast_months_str = [m.strftime('%Y-%m') for m in forecast_months]
        
        # Prepare forecast results
        forecast_result = "\nForecast:\n"
        for i, month in enumerate(forecast_months_str):
            forecast_result += f"{month}: {forecast[i]:.2f}%\n"
        
        analysis = f"""
        {brand.capitalize()} {metric_name} Forecast Analysis:
        
        Historical Data:
        {df_agg.to_string()}
        
        {forecast_result}
        
        Forecast Insights:
        {self._generate_forecast_insight(values[-1], forecast, brand, metric_name)}
        """
        
        return analysis
    
    def _extract_metric(self, query):
        if "market share" in query or "marketshare" in query:
            return "market_share"
        else:
            return "penetration"
    
    def _extract_brand(self, query):
        brands = ["oreo", "chipsahoy", "ritz", "belvita", "nutterbutter"]
        for brand in brands:
            if brand in query:
                return brand
        return "oreo"  # default
    
    def _extract_forecast_horizon(self, query):
        horizon = 3  # default
        if "month" in query:
            for i in range(1, 13):
                if str(i) in query:
                    horizon = i
                    break
        return horizon
    
    def _generate_forecast(self, values, horizon):
        # Simple ARIMA forecasting
        try:
            model = ARIMA(values, order=(1,1,0))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=horizon)
            return forecast
        except:
            # Fallback to simple trend-based forecast if ARIMA fails
            last_value = values[-1]
            if len(values) >= 2:
                avg_change = np.mean([values[i] - values[i-1] for i in range(1, len(values))])
                return [last_value + avg_change * (i+1) for i in range(horizon)]
            else:
                return [last_value] * horizon
    
    def _generate_forecast_insight(self, last_value, forecast_values, brand, metric):
        trend = forecast_values[-1] - last_value
        trend_pct = (trend / last_value) * 100 if last_value > 0 else 0
        
        if trend > 0:
            direction = "increasing"
            sentiment = "positive" if trend_pct > 5 else "modestly positive"
        elif trend < 0:
            direction = "decreasing"
            sentiment = "concerning" if trend_pct < -5 else "slightly concerning"
        else:
            direction = "stable"
            sentiment = "neutral"
            
        insight = f"The forecast shows an overall {direction} trend for {brand.capitalize()}'s {metric.lower()}, "
        insight += f"with a projected change of {trend:.2f} percentage points ({trend_pct:.1f}%) from current levels. "
        insight += f"This suggests a {sentiment} outlook for the brand in this metric."
        
        # Add volatility assessment if we have enough forecast points
        if len(forecast_values) >= 3:
            volatility = np.std(forecast_values)
            if volatility > 1.0:
                insight += f" There is significant volatility in the forecast (std dev: {volatility:.2f}), suggesting uncertainty."
            elif volatility > 0.5:
                insight += f" There is moderate volatility in the forecast (std dev: {volatility:.2f})."
            else:
                insight += f" The forecast shows relatively stable progression (std dev: {volatility:.2f})."
                
        return insight
