"""
Configuration settings for Vehicle Registration Dashboard
"""

import os
from datetime import datetime
from typing import Dict, List

class Config:
    """Main configuration class"""
    
    # Application Settings
    APP_NAME = "Vehicle Registration Dashboard"
    APP_VERSION = "1.0.0"
    COMPANY_NAME = "Financially Free"
    
    # Data Sources
    VAHAN_BASE_URL = "https://vahan.parivahan.gov.in/vahan4dashboard"
    ANALYTICS_URL = "https://analytics.parivahan.gov.in/analytics/publicdashboard/vahan"
    
    # Database Configuration
    DATABASE_PATH = os.getenv('DATABASE_PATH', 'vehicle_data.db')
    
    # Scraping Configuration
    REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', '30'))
    MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))
    DATA_REFRESH_INTERVAL = int(os.getenv('DATA_REFRESH_INTERVAL', '3600'))  # 1 hour
    
    # Browser Configuration
    HEADLESS_BROWSER = os.getenv('HEADLESS_BROWSER', 'true').lower() == 'true'
    CHROME_OPTIONS = os.getenv('CHROME_OPTIONS', '--no-sandbox,--disable-dev-shm-usage').split(',')
    
    # Vehicle Categories Mapping
    VEHICLE_CATEGORIES = {
        '2W': 'Two Wheeler',
        '3W': 'Three Wheeler', 
        '4W': 'Four Wheeler',
        'LMV': 'Light Motor Vehicle',
        'HMV': 'Heavy Motor Vehicle',
        'BUS': 'Bus',
        'TRAILER': 'Trailer'
    }
    
    # Major Manufacturers to Track
    MAJOR_MANUFACTURERS = [
        'HERO MOTOCORP LTD',
        'BAJAJ AUTO LTD',
        'TVS MOTOR COMPANY',
        'MARUTI SUZUKI INDIA LIMITED',
        'HYUNDAI MOTOR INDIA LTD',
        'TATA MOTORS LTD',
        'MAHINDRA AND MAHINDRA LTD',
        'HONDA CARS INDIA LTD',
        'TOYOTA KIRLOSKAR MOTOR',
        'ASHOK LEYLAND LTD',
        'FORCE MOTORS LIMITED',
        'EICHER MOTORS LTD',
        'HONDA MOTORCYCLE AND SCOOTER INDIA (P) LTD',
        'YAMAHA MOTOR INDIA PVT LTD',
        'ROYAL ENFIELD'
    ]
    
    # Indian States and UTs
    INDIAN_STATES = [
        'ANDHRA PRADESH', 'ARUNACHAL PRADESH', 'ASSAM', 'BIHAR', 'CHHATTISGARH',
        'GOA', 'GUJARAT', 'HARYANA', 'HIMACHAL PRADESH', 'JHARKHAND',
        'KARNATAKA', 'KERALA', 'MADHYA PRADESH', 'MAHARASHTRA', 'MANIPUR',
        'MEGHALAYA', 'MIZORAM', 'NAGALAND', 'ODISHA', 'PUNJAB',
        'RAJASTHAN', 'SIKKIM', 'TAMIL NADU', 'TELANGANA', 'TRIPURA',
        'UTTAR PRADESH', 'UTTARAKHAND', 'WEST BENGAL',
        # Union Territories
        'ANDAMAN & NICOBAR ISLANDS', 'CHANDIGARH', 'DADRA & NAGAR HAVELI AND DAMAN & DIU',
        'DELHI', 'JAMMU & KASHMIR', 'LADAKH', 'LAKSHADWEEP', 'PUDUCHERRY'
    ]
    
    # UI Configuration
    STREAMLIT_CONFIG = {
        'page_title': f"{APP_NAME} | {COMPANY_NAME}",
        'page_icon': "🚗",
        'layout': "wide",
        'initial_sidebar_state': "expanded"
    }
    
    # Color Schemes for Visualizations
    COLOR_SCHEMES = {
        'primary': ['#1e3a8a', '#3b82f6', '#60a5fa', '#93c5fd', '#dbeafe'],
        'secondary': ['#059669', '#10b981', '#34d399', '#6ee7b7', '#d1fae5'],
        'accent': ['#dc2626', '#ef4444', '#f87171', '#fca5a5', '#fecaca'],
        'neutral': ['#374151', '#6b7280', '#9ca3af', '#d1d5db', '#f3f4f6']
    }
    
    # Chart Configuration
    CHART_CONFIG = {
        'height': 400,
        'margin': {'l': 40, 'r': 40, 't': 40, 'b': 40},
        'font_family': 'Arial, sans-serif',
        'title_font_size': 16,
        'axis_font_size': 12
    }
    
    # Data Processing Settings
    DATA_PROCESSING = {
        'max_records_per_query': 50000,
        'chunk_size': 1000,
        'outlier_threshold': 3.0,  # Standard deviations
        'min_data_points_for_analysis': 4,
        'forecast_periods': 4  # Quarters ahead
    }
    
    # Logging Configuration
    LOGGING_CONFIG = {
        'level': os.getenv('LOG_LEVEL', 'INFO'),
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'filename': 'dashboard.log',
        'max_bytes': 10 * 1024 * 1024,  # 10MB
        'backup_count': 5
    }
    
    # API Rate Limiting
    RATE_LIMITING = {
        'requests_per_minute': 60,
        'burst_limit': 10,
        'backoff_factor': 1.5
    }
    
    # Cache Settings
    CACHE_CONFIG = {
        'ttl_seconds': 3600,  # 1 hour
        'max_entries': 1000,
        'persist_to_disk': True
    }
    
    # Feature Flags
    FEATURE_FLAGS = {
        'enable_real_time_scraping': True,
        'enable_forecasting': True,
        'enable_advanced_analytics': True,
        'enable_data_export': True,
        'enable_automated_reports': True,
        'enable_email_alerts': False,
        'enable_api_endpoints': False
    }
    
    # Data Quality Thresholds
    DATA_QUALITY = {
        'min_completeness_ratio': 0.8,  # 80% complete data required
        'max_outlier_ratio': 0.05,      # Max 5% outliers
        'min_consistency_score': 0.9,   # 90% consistency required
        'data_freshness_hours': 24      # Data must be < 24 hours old
    }
    
    # Investment Analysis Parameters
    INVESTMENT_METRICS = {
        'high_growth_threshold': 20.0,     # 20% YoY growth
        'market_leader_threshold': 15.0,   # 15% market share
        'emerging_player_threshold': 50.0, # 50% CAGR for emerging
        'volatility_threshold': 2.0,       # Coefficient of variation
        'concentration_hhi_threshold': 2500 # HHI concentration index
    }
    
    # Seasonal Adjustment Factors
    SEASONAL_FACTORS = {
        1: 0.90,  # Q1 - Lower due to post-festival period
        2: 1.10,  # Q2 - Higher due to new financial year
        3: 0.95,  # Q3 - Moderate due to monsoon
        4: 1.15   # Q4 - Highest due to festival season
    }
    
    # Regional Economic Indicators (GDP per capita in thousands)
    REGIONAL_INDICATORS = {
        'MAHARASHTRA': 220,
        'TAMIL NADU': 180,
        'GUJARAT': 190,
        'KARNATAKA': 170,
        'HARYANA': 210,
        'PUNJAB': 150,
        'KERALA': 160,
        'TELANGANA': 180,
        'ANDHRA PRADESH': 130,
        'RAJASTHAN': 110,
        'WEST BENGAL': 100,
        'MADHYA PRADESH': 80,
        'UTTAR PRADESH': 70,
        'BIHAR': 50,
        'ODISHA': 80,
        'JHARKHAND': 70
    }
    
    @classmethod
    def get_database_url(cls) -> str:
        """Get database connection URL"""
        return f"sqlite:///{cls.DATABASE_PATH}"
    
    @classmethod
    def get_chrome_options(cls) -> List[str]:
        """Get Chrome browser options"""
        options = cls.CHROME_OPTIONS.copy()
        if cls.HEADLESS_BROWSER:
            options.append('--headless')
        return options
    
    @classmethod
    def validate_config(cls) -> List[str]:
        """Validate configuration settings"""
        issues = []
        
        # Check required directories
        db_dir = os.path.dirname(cls.DATABASE_PATH)
        if db_dir and not os.path.exists(db_dir):
            try:
                os.makedirs(db_dir, exist_ok=True)
            except Exception as e:
                issues.append(f"Cannot create database directory: {e}")
        
        # Validate thresholds
        if cls.INVESTMENT_METRICS['high_growth_threshold'] <= 0:
            issues.append("High growth threshold must be positive")
        
        if not (0 <= cls.DATA_QUALITY['min_completeness_ratio'] <= 1):
            issues.append("Completeness ratio must be between 0 and 1")
        
        # Check seasonal factors sum
        seasonal_sum = sum(cls.SEASONAL_FACTORS.values())
        if abs(seasonal_sum - 4.0) > 0.1:  # Should average to 1.0 per quarter
            issues.append(f"Seasonal factors should sum to 4.0, got {seasonal_sum}")
        
        return issues

class DevelopmentConfig(Config):
    """Development environment configuration"""
    
    # Override for development
    DATABASE_PATH = 'dev_vehicle_data.db'
    DATA_REFRESH_INTERVAL = 300  # 5 minutes for faster testing
    HEADLESS_BROWSER = False     # Show browser for debugging
    
    FEATURE_FLAGS = {
        **Config.FEATURE_FLAGS,
        'enable_real_time_scraping': False,  # Use sample data in dev
        'enable_email_alerts': False,
        'enable_api_endpoints': True
    }
    
    LOGGING_CONFIG = {
        **Config.LOGGING_CONFIG,
        'level': 'DEBUG'
    }

class ProductionConfig(Config):
    """Production environment configuration"""
    
    # Production optimizations
    DATA_REFRESH_INTERVAL = 3600  # 1 hour
    HEADLESS_BROWSER = True
    
    FEATURE_FLAGS = {
        **Config.FEATURE_FLAGS,
        'enable_real_time_scraping': True,
        'enable_email_alerts': True,
        'enable_api_endpoints': False  # Disable for security
    }
    
    LOGGING_CONFIG = {
        **Config.LOGGING_CONFIG,
        'level': 'INFO'
    }
    
    # Enhanced security settings
    RATE_LIMITING = {
        **Config.RATE_LIMITING,
        'requests_per_minute': 30,  # More conservative
        'burst_limit': 5
    }

class TestingConfig(Config):
    """Testing environment configuration"""
    
    DATABASE_PATH = ':memory:'  # In-memory database for tests
    DATA_REFRESH_INTERVAL = 60  # 1 minute for quick tests
    
    FEATURE_FLAGS = {
        **Config.FEATURE_FLAGS,
        'enable_real_time_scraping': False,  # Always use mock data
        'enable_email_alerts': False,
        'enable_automated_reports': False
    }
    
    LOGGING_CONFIG = {
        **Config.LOGGING_CONFIG,
        'level': 'WARNING'  # Reduce noise in tests
    }

def get_config() -> Config:
    """Get configuration based on environment"""
    env = os.getenv('ENVIRONMENT', 'development').lower()
    
    if env == 'production':
        return ProductionConfig()
    elif env == 'testing':
        return TestingConfig()
    else:
        return DevelopmentConfig()

# Global configuration instance
config = get_config()

# Validate configuration on import
config_issues = config.validate_config()
if config_issues:
    import logging
    logger = logging.getLogger(__name__)
    for issue in config_issues:
        logger.warning(f"Configuration issue: {issue}")

# Educational content and messages
EDUCATIONAL_CONTENT = {
    'dashboard_intro': """
    Welcome to the Vehicle Registration Analytics Dashboard! This tool is designed to help you learn 
    investment analysis through real-world automobile sector data. 
    
    Remember: This is for educational purposes only, not financial advice.
    """,
    
    'yoy_explanation': """
    Year-over-Year (YoY) Growth measures the percentage change in vehicle registrations compared to 
    the same period in the previous year. This metric helps identify long-term trends and cyclical patterns.
    """,
    
    'qoq_explanation': """
    Quarter-over-Quarter (QoQ) Growth shows short-term performance changes. High QoQ volatility might 
    indicate seasonal effects or market disruptions worth investigating.
    """,
    
    'market_share_explanation': """
    Market share analysis reveals competitive positioning. Look for:
    - Dominant players (>20% share)
    - Emerging competitors (growing share)
    - Market fragmentation patterns
    """,
    
    'investment_insights': {
        'growth_trends': "Consistent YoY growth above 15% often indicates strong market demand",
        'market_concentration': "HHI above 2500 suggests concentrated market with potential pricing power",
        'seasonal_patterns': "Understanding seasonality helps predict quarterly performance variations",
        'geographic_distribution': "State-wise analysis reveals regional market opportunities",
        'emerging_players': "High CAGR with increasing market share signals potential investment opportunities"
    }
}

# Disclaimer and Legal
DISCLAIMERS = {
    'main_disclaimer': """
    IMPORTANT DISCLAIMER: This dashboard is created exclusively for educational purposes as part of 
    Financially Free's mission to provide practical investment education. All data analysis, insights, 
    and recommendations are for learning purposes only and should NOT be used for actual investment 
    decisions. Always consult qualified financial advisors before making investment choices.
    """,
    
    'data_disclaimer': """
    Data is sourced from official government databases (Ministry of Road Transport & Highways). 
    While we strive for accuracy, users should verify critical information from original sources. 
    Real-time scraping may occasionally fail due to website changes or connectivity issues.
    """,
    
    'educational_purpose': """
    This tool is designed to teach investment analysis concepts including:
    - Sector analysis techniques
    - Growth metrics interpretation  
    - Market share evaluation
    - Trend identification
    - Risk assessment frameworks
    """
}

# Export commonly used configurations
__all__ = [
    'Config', 'DevelopmentConfig', 'ProductionConfig', 'TestingConfig',
    'get_config', 'config', 'EDUCATIONAL_CONTENT', 'DISCLAIMERS'
]