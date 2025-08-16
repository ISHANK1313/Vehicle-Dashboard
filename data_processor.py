"""
Advanced Data Processing Module for Vehicle Registration Dashboard
Handles complex data transformations, validations, and analytics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import sqlite3
import json
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class VehicleCategory(Enum):
    """Enum for vehicle categories"""
    TWO_WHEELER = "2W"
    THREE_WHEELER = "3W"
    FOUR_WHEELER = "4W"
    LIGHT_MOTOR_VEHICLE = "LMV"
    HEAVY_MOTOR_VEHICLE = "HMV"
    BUS = "BUS"
    TRAILER = "TRAILER"

@dataclass
class RegistrationRecord:
    """Data class for vehicle registration record"""
    year: int
    quarter: int
    state: str
    category: str
    manufacturer: str
    registrations: int
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    @property
    def period(self) -> str:
        return f"{self.year}-Q{self.quarter}"
    
    def validate(self) -> bool:
        """Validate record data"""
        if self.year < 2020 or self.year > datetime.now().year:
            return False
        if self.quarter < 1 or self.quarter > 4:
            return False
        if self.registrations < 0:
            return False
        return True

class DataValidator:
    """Validates and cleans vehicle registration data"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Validate and clean dataframe
        Returns: (cleaned_df, list_of_issues)
        """
        issues = []
        
        if df.empty:
            issues.append("DataFrame is empty")
            return df, issues
        
        # Required columns
        required_columns = ['year', 'quarter', 'state', 'category', 'manufacturer', 'registrations']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
            return df, issues
        
        original_rows = len(df)
        
        # Data type validation and conversion
        try:
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            df['quarter'] = pd.to_numeric(df['quarter'], errors='coerce')
            df['registrations'] = pd.to_numeric(df['registrations'], errors='coerce')
        except Exception as e:
            issues.append(f"Data type conversion error: {str(e)}")
        
        # Remove rows with invalid data
        df = df.dropna(subset=['year', 'quarter', 'registrations'])
        
        # Validate year range
        current_year = datetime.now().year
        df = df[(df['year'] >= 2020) & (df['year'] <= current_year)]
        
        # Validate quarter range
        df = df[(df['quarter'] >= 1) & (df['quarter'] <= 4)]
        
        # Validate registrations (non-negative)
        df = df[df['registrations'] >= 0]
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['year', 'quarter', 'state', 'category', 'manufacturer'])
        
        # Clean string columns
        string_columns = ['state', 'category', 'manufacturer']
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()
        
        final_rows = len(df)
        if final_rows < original_rows:
            issues.append(f"Removed {original_rows - final_rows} invalid records")
        
        logger.info(f"Data validation completed. {final_rows} valid records from {original_rows} original records")
        
        return df, issues
    
    @staticmethod
    def detect_outliers(df: pd.DataFrame, column: str = 'registrations') -> pd.DataFrame:
        """Detect outliers using IQR method"""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        
        return outliers

class AdvancedAnalytics:
    """Advanced analytics for vehicle registration data"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare data for advanced analytics"""
        if self.df.empty:
            return
        
        # Create additional calculated columns
        self.df['year_quarter'] = self.df['year'].astype(str) + '-Q' + self.df['quarter'].astype(str)
        self.df['log_registrations'] = np.log1p(self.df['registrations'])
        
        # Sort by time periods
        self.df = self.df.sort_values(['year', 'quarter'])
    
    def calculate_market_concentration(self) -> Dict[str, float]:
        """Calculate market concentration metrics"""
        concentration_metrics = {}
        
        for category in self.df['category'].unique():
            category_data = self.df[self.df['category'] == category]
            manufacturer_shares = (category_data.groupby('manufacturer')['registrations'].sum() / 
                                 category_data['registrations'].sum() * 100).sort_values(ascending=False)
            
            # Herfindahl-Hirschman Index (HHI)
            hhi = (manufacturer_shares ** 2).sum()
            
            # Top 3 concentration ratio
            cr3 = manufacturer_shares.head(3).sum()
            
            # Top 5 concentration ratio
            cr5 = manufacturer_shares.head(5).sum()
            
            concentration_metrics[category] = {
                'hhi': hhi,
                'cr3': cr3,
                'cr5': cr5,
                'number_of_players': len(manufacturer_shares)
            }
        
        return concentration_metrics
    
    def calculate_growth_volatility(self) -> pd.DataFrame:
        """Calculate growth rate volatility for each manufacturer-category combination"""
        volatility_data = []
        
        for category in self.df['category'].unique():
            for manufacturer in self.df['manufacturer'].unique():
                subset = self.df[
                    (self.df['category'] == category) & 
                    (self.df['manufacturer'] == manufacturer)
                ].sort_values(['year', 'quarter'])
                
                if len(subset) > 2:
                    # Calculate period-over-period growth rates
                    subset['growth_rate'] = subset['registrations'].pct_change() * 100
                    
                    volatility_data.append({
                        'category': category,
                        'manufacturer': manufacturer,
                        'mean_growth': subset['growth_rate'].mean(),
                        'std_growth': subset['growth_rate'].std(),
                        'cv_growth': subset['growth_rate'].std() / abs(subset['growth_rate'].mean()) if subset['growth_rate'].mean() != 0 else np.inf,
                        'periods': len(subset)
                    })
        
        return pd.DataFrame(volatility_data)
    
    def seasonal_decomposition(self) -> Dict[str, pd.DataFrame]:
        """Perform seasonal decomposition for each category"""
        seasonal_results = {}
        
        for category in self.df['category'].unique():
            category_data = (self.df[self.df['category'] == category]
                           .groupby(['year', 'quarter'])['registrations']
                           .sum().reset_index())
            
            if len(category_data) >= 8:  # Need at least 2 years for seasonal analysis
                # Calculate seasonal indices
                seasonal_avg = category_data.groupby('quarter')['registrations'].mean()
                overall_avg = category_data['registrations'].mean()
                seasonal_indices = seasonal_avg / overall_avg
                
                # Calculate trend
                category_data['trend'] = category_data['registrations'].rolling(window=4, center=True).mean()
                
                # Calculate seasonal component
                category_data['seasonal'] = category_data['quarter'].map(seasonal_indices) * overall_avg
                
                # Calculate residual
                category_data['residual'] = (category_data['registrations'] - 
                                           category_data['trend'].fillna(overall_avg) - 
                                           category_data['seasonal'] + overall_avg)
                
                seasonal_results[category] = category_data
        
        return seasonal_results
    
    def calculate_compound_annual_growth_rate(self) -> pd.DataFrame:
        """Calculate CAGR for each manufacturer-category combination"""
        cagr_data = []
        
        for category in self.df['category'].unique():
            for manufacturer in self.df['manufacturer'].unique():
                yearly_data = (self.df[
                    (self.df['category'] == category) & 
                    (self.df['manufacturer'] == manufacturer)
                ].groupby('year')['registrations'].sum().reset_index())
                
                if len(yearly_data) >= 2:
                    first_year = yearly_data.iloc[0]
                    last_year = yearly_data.iloc[-1]
                    
                    years = last_year['year'] - first_year['year']
                    
                    if years > 0 and first_year['registrations'] > 0:
                        cagr = ((last_year['registrations'] / first_year['registrations']) ** (1/years) - 1) * 100
                        
                        cagr_data.append({
                            'category': category,
                            'manufacturer': manufacturer,
                            'cagr': cagr,
                            'start_year': first_year['year'],
                            'end_year': last_year['year'],
                            'start_registrations': first_year['registrations'],
                            'end_registrations': last_year['registrations']
                        })
        
        return pd.DataFrame(cagr_data)
    
    def market_share_trends(self) -> pd.DataFrame:
        """Analyze market share trends over time"""
        share_trends = []
        
        for year in self.df['year'].unique():
            for category in self.df['category'].unique():
                year_category_data = self.df[
                    (self.df['year'] == year) & 
                    (self.df['category'] == category)
                ]
                
                if not year_category_data.empty:
                    total_registrations = year_category_data['registrations'].sum()
                    
                    for manufacturer in year_category_data['manufacturer'].unique():
                        manufacturer_registrations = year_category_data[
                            year_category_data['manufacturer'] == manufacturer
                        ]['registrations'].sum()
                        
                        market_share = (manufacturer_registrations / total_registrations) * 100
                        
                        share_trends.append({
                            'year': year,
                            'category': category,
                            'manufacturer': manufacturer,
                            'market_share': market_share,
                            'registrations': manufacturer_registrations
                        })
        
        return pd.DataFrame(share_trends)
    
    def identify_emerging_players(self, growth_threshold: float = 50.0, 
                                share_threshold: float = 1.0) -> pd.DataFrame:
        """Identify emerging manufacturers based on growth and market share"""
        # Get CAGR data
        cagr_df = self.calculate_compound_annual_growth_rate()
        
        # Get latest market share
        latest_year = self.df['year'].max()
        latest_shares = self.market_share_trends()
        latest_shares = latest_shares[latest_shares['year'] == latest_year]
        
        # Merge CAGR and market share data
        emerging = cagr_df.merge(
            latest_shares[['category', 'manufacturer', 'market_share']],
            on=['category', 'manufacturer'],
            how='inner'
        )
        
        # Filter for emerging players
        emerging_players = emerging[
            (emerging['cagr'] > growth_threshold) & 
            (emerging['market_share'] > share_threshold)
        ].sort_values('cagr', ascending=False)
        
        return emerging_players
    
    def calculate_market_penetration(self) -> pd.DataFrame:
        """Calculate market penetration rates by state and category"""
        penetration_data = []
        
        # Approximate population data (in millions) for major states
        state_population = {
            'UTTAR PRADESH': 231.0,
            'MAHARASHTRA': 123.1,
            'BIHAR': 124.8,
            'WEST BENGAL': 102.6,
            'MADHYA PRADESH': 85.0,
            'TAMIL NADU': 77.8,
            'RAJASTHAN': 81.0,
            'KARNATAKA': 67.6,
            'GUJARAT': 70.1,
            'ANDHRA PRADESH': 53.9,
            'ODISHA': 47.0,
            'TELANGANA': 39.4,
            'KERALA': 35.7,
            'JHARKHAND': 38.6,
            'ASSAM': 35.6
        }
        
        for state in self.df['state'].unique():
            if state in state_population:
                for category in self.df['category'].unique():
                    state_category_data = self.df[
                        (self.df['state'] == state) & 
                        (self.df['category'] == category)
                    ]
                    
                    if not state_category_data.empty:
                        total_vehicles = state_category_data['registrations'].sum()
                        penetration_rate = (total_vehicles / (state_population[state] * 1000000)) * 1000  # Per 1000 people
                        
                        penetration_data.append({
                            'state': state,
                            'category': category,
                            'total_vehicles': total_vehicles,
                            'population_millions': state_population[state],
                            'penetration_per_1000': penetration_rate
                        })
        
        return pd.DataFrame(penetration_data)

class ForecastingEngine:
    """Simple forecasting engine for vehicle registrations"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._prepare_time_series()
    
    def _prepare_time_series(self):
        """Prepare time series data for forecasting"""
        if self.df.empty:
            return
        
        # Create time series for total registrations
        self.time_series = (self.df.groupby(['year', 'quarter'])['registrations']
                           .sum().reset_index())
        self.time_series['period'] = range(len(self.time_series))
        self.time_series['year_quarter'] = (self.time_series['year'].astype(str) + 
                                          '-Q' + self.time_series['quarter'].astype(str))
    
    def simple_linear_forecast(self, periods_ahead: int = 4) -> pd.DataFrame:
        """Simple linear trend forecasting"""
        if len(self.time_series) < 4:
            return pd.DataFrame()
        
        # Fit linear trend
        X = self.time_series['period'].values.reshape(-1, 1)
        y = self.time_series['registrations'].values
        
        # Simple linear regression
        n = len(X)
        x_mean = X.mean()
        y_mean = y.mean()
        
        slope = np.sum((X.flatten() - x_mean) * (y - y_mean)) / np.sum((X.flatten() - x_mean) ** 2)
        intercept = y_mean - slope * x_mean
        
        # Generate forecasts
        forecasts = []
        last_period = self.time_series['period'].max()
        last_year = self.time_series['year'].max()
        last_quarter = self.time_series['quarter'].max()
        
        for i in range(1, periods_ahead + 1):
            future_period = last_period + i
            forecast_value = intercept + slope * future_period
            
            # Calculate future year and quarter
            if last_quarter + i > 4:
                future_year = last_year + ((last_quarter + i - 1) // 4)
                future_quarter = ((last_quarter + i - 1) % 4) + 1
            else:
                future_year = last_year
                future_quarter = last_quarter + i
            
            forecasts.append({
                'year': future_year,
                'quarter': future_quarter,
                'period': future_period,
                'forecast_registrations': max(0, forecast_value),  # Ensure non-negative
                'year_quarter': f"{future_year}-Q{future_quarter}"
            })
        
        return pd.DataFrame(forecasts)
    
    def moving_average_forecast(self, window: int = 4, periods_ahead: int = 4) -> pd.DataFrame:
        """Moving average forecasting"""
        if len(self.time_series) < window:
            return pd.DataFrame()
        
        # Calculate moving average
        moving_avg = self.time_series['registrations'].rolling(window=window).mean().iloc[-1]
        
        # Generate forecasts
        forecasts = []
        last_year = self.time_series['year'].max()
        last_quarter = self.time_series['quarter'].max()
        
        for i in range(1, periods_ahead + 1):
            # Calculate future year and quarter
            if last_quarter + i > 4:
                future_year = last_year + ((last_quarter + i - 1) // 4)
                future_quarter = ((last_quarter + i - 1) % 4) + 1
            else:
                future_year = last_year
                future_quarter = last_quarter + i
            
            forecasts.append({
                'year': future_year,
                'quarter': future_quarter,
                'forecast_registrations': moving_avg,
                'year_quarter': f"{future_year}-Q{future_quarter}"
            })
        
        return pd.DataFrame(forecasts)

class ReportGenerator:
    """Generate comprehensive analysis reports"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.analytics = AdvancedAnalytics(df)
        self.forecasting = ForecastingEngine(df)
    
    def generate_executive_summary(self) -> Dict[str, any]:
        """Generate executive summary of the data"""
        summary = {}
        
        if self.df.empty:
            return {"error": "No data available for analysis"}
        
        # Basic metrics
        summary['total_registrations'] = self.df['registrations'].sum()
        summary['time_period'] = f"{self.df['year'].min()}-{self.df['year'].max()}"
        summary['categories_analyzed'] = self.df['category'].nunique()
        summary['manufacturers_analyzed'] = self.df['manufacturer'].nunique()
        summary['states_covered'] = self.df['state'].nunique()
        
        # Growth analysis
        yearly_totals = self.df.groupby('year')['registrations'].sum()
        if len(yearly_totals) > 1:
            latest_growth = ((yearly_totals.iloc[-1] - yearly_totals.iloc[-2]) / yearly_totals.iloc[-2]) * 100
            summary['latest_yoy_growth'] = latest_growth
        
        # Market concentration
        concentration = self.analytics.calculate_market_concentration()
        if concentration:
            avg_hhi = np.mean([metrics['hhi'] for metrics in concentration.values()])
            summary['average_market_concentration'] = avg_hhi
        
        # Top performers
        top_category = self.df.groupby('category')['registrations'].sum().idxmax()
        top_manufacturer = self.df.groupby('manufacturer')['registrations'].sum().idxmax()
        
        summary['top_category'] = top_category
        summary['top_manufacturer'] = top_manufacturer
        
        return summary
    
    def generate_risk_assessment(self) -> Dict[str, List[str]]:
        """Generate risk assessment based on data patterns"""
        risks = {
            'high_risk': [],
            'medium_risk': [],
            'low_risk': [],
            'opportunities': []
        }
        
        # Market concentration risks
        concentration = self.analytics.calculate_market_concentration()
        for category, metrics in concentration.items():
            if metrics['hhi'] > 2500:  # Highly concentrated market
                risks['high_risk'].append(f"High market concentration in {category} (HHI: {metrics['hhi']:.0f})")
            elif metrics['hhi'] > 1500:
                risks['medium_risk'].append(f"Moderate market concentration in {category}")
        
        # Growth volatility risks
        volatility = self.analytics.calculate_growth_volatility()
        if not volatility.empty:
            high_volatility = volatility[volatility['cv_growth'] > 2.0]  # High coefficient of variation
            for _, row in high_volatility.iterrows():
                risks['medium_risk'].append(f"High growth volatility: {row['manufacturer']} in {row['category']}")
        
        # Emerging opportunities
        emerging = self.analytics.identify_emerging_players(growth_threshold=30.0)
        if not emerging.empty:
            for _, row in emerging.head(3).iterrows():
                risks['opportunities'].append(f"Emerging player: {row['manufacturer']} in {row['category']} ({row['cagr']:.1f}% CAGR)")
        
        return risks
    
    def export_to_json(self, filename: str = None) -> str:
        """Export complete analysis to JSON"""
        if filename is None:
            filename = f"vehicle_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'data_period': f"{self.df['year'].min()}-{self.df['year'].max()}",
                'total_records': len(self.df)
            },
            'executive_summary': self.generate_executive_summary(),
            'risk_assessment': self.generate_risk_assessment(),
            'market_concentration': self.analytics.calculate_market_concentration(),
            'growth_volatility': self.analytics.calculate_growth_volatility().to_dict('records'),
            'cagr_analysis': self.analytics.calculate_compound_annual_growth_rate().to_dict('records'),
            'emerging_players': self.analytics.identify_emerging_players().to_dict('records'),
            'forecasts': {
                'linear_forecast': self.forecasting.simple_linear_forecast().to_dict('records'),
                'moving_average_forecast': self.forecasting.moving_average_forecast().to_dict('records')
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Analysis report exported to {filename}")
        return filename

# Utility functions
def load_sample_data() -> pd.DataFrame:
    """Load sample data for testing and demonstration"""
    from vehicle_dashboard import VahanDataScraper
    scraper = VahanDataScraper()
    return scraper._generate_sample_data()

def run_full_analysis(df: pd.DataFrame = None) -> Dict:
    """Run complete analysis pipeline"""
    if df is None:
        df = load_sample_data()
    
    # Validate data
    cleaned_df, issues = DataValidator.validate_dataframe(df)
    
    if cleaned_df.empty:
        return {"error": "No valid data for analysis", "issues": issues}
    
    # Generate comprehensive report
    report_generator = ReportGenerator(cleaned_df)
    
    # Export detailed analysis
    json_file = report_generator.export_to_json()
    
    return {
        "status": "success",
        "data_issues": issues,
        "records_analyzed": len(cleaned_df),
        "executive_summary": report_generator.generate_executive_summary(),
        "report_file": json_file
    }

if __name__ == "__main__":
    # Run analysis on sample data
    print("🚗 Vehicle Registration Data Analysis")
    print("=" * 50)
    
    result = run_full_analysis()
    
    if result.get("status") == "success":
        print(f"✅ Analysis completed successfully!")
        print(f"📊 Records analyzed: {result['records_analyzed']}")
        print(f"📄 Detailed report: {result['report_file']}")
        
        # Print executive summary
        summary = result['executive_summary']
        print(f"\n📈 Executive Summary:")
        for key, value in summary.items():
            if key != 'error':
                print(f"  • {key.replace('_', ' ').title()}: {value}")
    else:
        print(f"❌ Analysis failed: {result.get('error')}")
        if result.get('issues'):
            print("Issues found:")
            for issue in result['issues']:
                print(f"  • {issue}")