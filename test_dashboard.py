"""
Comprehensive test suite for Vehicle Registration Dashboard
Run with: python -m pytest test_dashboard.py -v
"""

import pytest
import pandas as pd
import numpy as np
import sqlite3
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import json

# Import modules to test
try:
    from vehicle_dashboard import (
        VahanDataScraper, DatabaseManager, DataAnalyzer, 
        DashboardUI, Config
    )
    from data_processor import (
        DataValidator, AdvancedAnalytics, ForecastingEngine,
        ReportGenerator, RegistrationRecord
    )
    from config import get_config, TestingConfig
except ImportError as e:
    pytest.skip(f"Could not import required modules: {e}", allow_module_level=True)

class TestConfig:
    """Test configuration and setup"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample test data"""
        np.random.seed(42)
        data = []
        
        states = ['MAHARASHTRA', 'UTTAR PRADESH', 'TAMIL NADU', 'KARNATAKA']
        manufacturers = ['HERO MOTOCORP LTD', 'BAJAJ AUTO LTD', 'MARUTI SUZUKI INDIA LIMITED']
        categories = ['2W', '4W', 'LMV']
        
        for year in [2022, 2023, 2024]:
            for quarter in [1, 2, 3, 4]:
                for state in states:
                    for category in categories:
                        for manufacturer in manufacturers:
                            registrations = np.random.randint(100, 5000)
                            data.append({
                                'year': year,
                                'quarter': quarter,
                                'state': state,
                                'category': category,
                                'manufacturer': manufacturer,
                                'registrations': registrations,
                                'timestamp': datetime.now()
                            })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp:
            db_path = tmp.name
        
        yield db_path
        
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
    
    @pytest.fixture
    def mock_scraper_response(self):
        """Mock response from web scraper"""
        return {
            'total_registrations': 1000000,
            'registration_data': [
                {'year': 2024, 'quarter': 1, 'state': 'MAHARASHTRA', 
                 'category': '2W', 'manufacturer': 'HERO MOTOCORP LTD', 'registrations': 5000}
            ],
            'timestamp': datetime.now()
        }

class TestDataValidator:
    """Test data validation functionality"""
    
    def test_validate_valid_dataframe(self, sample_data):
        """Test validation of valid dataframe"""
        cleaned_df, issues = DataValidator.validate_dataframe(sample_data)
        
        assert not cleaned_df.empty
        assert len(issues) <= 1  # May have info about removed duplicates
        assert cleaned_df['year'].dtype in [np.int64, np.float64]
        assert cleaned_df['registrations'].min() >= 0
    
    def test_validate_empty_dataframe(self):
        """Test validation of empty dataframe"""
        empty_df = pd.DataFrame()
        cleaned_df, issues = DataValidator.validate_dataframe(empty_df)
        
        assert cleaned_df.empty
        assert "DataFrame is empty" in issues
    
    def test_validate_missing_columns(self):
        """Test validation with missing required columns"""
        invalid_df = pd.DataFrame({'year': [2023], 'invalid_col': ['test']})
        cleaned_df, issues = DataValidator.validate_dataframe(invalid_df)
        
        assert cleaned_df.empty or len(issues) > 0
        assert any("Missing required columns" in issue for issue in issues)
    
    def test_validate_invalid_data_types(self, sample_data):
        """Test validation with invalid data types"""
        # Introduce invalid data
        invalid_df = sample_data.copy()
        invalid_df.loc[0, 'year'] = 'invalid_year'
        invalid_df.loc[1, 'registrations'] = -100  # Negative registrations
        
        cleaned_df, issues = DataValidator.validate_dataframe(invalid_df)
        
        # Should have fewer rows after cleaning
        assert len(cleaned_df) < len(invalid_df)
    
    def test_detect_outliers(self, sample_data):
        """Test outlier detection"""
        # Add obvious outliers
        outlier_df = sample_data.copy()
        outlier_df.loc[0, 'registrations'] = 1000000  # Very high value
        
        outliers = DataValidator.detect_outliers(outlier_df)
        assert not outliers.empty
        assert 1000000 in outliers['registrations'].values

class TestDatabaseManager:
    """Test database operations"""
    
    def test_init_database(self, temp_db):
        """Test database initialization"""
        db_manager = DatabaseManager(temp_db)
        
        # Check if tables were created
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='vehicle_registrations'"
            )
            assert cursor.fetchone() is not None
    
    def test_save_and_load_data(self, temp_db, sample_data):
        """Test saving and loading data"""
        db_manager = DatabaseManager(temp_db)
        
        # Save data
        db_manager.save_data(sample_data)
        
        # Load data
        loaded_df = db_manager.load_data()
        
        assert not loaded_df.empty
        assert len(loaded_df) == len(sample_data)
        assert set(loaded_df.columns).issuperset({'year', 'quarter', 'state', 'category', 'manufacturer', 'registrations'})
    
    def test_load_empty_database(self, temp_db):
        """Test loading from empty database"""
        db_manager = DatabaseManager(temp_db)
        loaded_df = db_manager.load_data()
        
        # Should return empty dataframe gracefully
        assert loaded_df.empty

class TestDataAnalyzer:
    """Test data analysis functionality"""
    
    def test_analyzer_initialization(self, sample_data):
        """Test analyzer initialization"""
        analyzer = DataAnalyzer(sample_data)
        
        assert not analyzer.df.empty
        assert 'period' in analyzer.df.columns
        assert analyzer.df['year'].dtype in [np.int64, np.float64]
    
    def test_calculate_yoy_growth(self, sample_data):
        """Test YoY growth calculation"""
        analyzer = DataAnalyzer(sample_data)
        yoy_df = analyzer.calculate_yoy_growth()
        
        if not yoy_df.empty:
            assert 'yoy_growth_rate' in yoy_df.columns
            assert 'current_registrations' in yoy_df.columns
            assert 'previous_registrations' in yoy_df.columns
            
            # Check for reasonable growth rates (not infinite)
            assert not yoy_df['yoy_growth_rate'].isin([np.inf, -np.inf]).any()
    
    def test_calculate_qoq_growth(self, sample_data):
        """Test QoQ growth calculation"""
        analyzer = DataAnalyzer(sample_data)
        qoq_df = analyzer.calculate_qoq_growth()
        
        if not qoq_df.empty:
            assert 'qoq_growth_rate' in qoq_df.columns
            assert 'period' in qoq_df.columns
            
            # Check data integrity
            assert not qoq_df['qoq_growth_rate'].isin([np.inf, -np.inf]).any()
    
    def test_get_market_share(self, sample_data):
        """Test market share calculation"""
        analyzer = DataAnalyzer(sample_data)
        market_share_df = analyzer.get_market_share()
        
        if not market_share_df.empty:
            assert 'market_share' in market_share_df.columns
            assert 'category' in market_share_df.columns
            assert 'manufacturer' in market_share_df.columns
            
            # Market shares should be between 0 and 100
            assert (market_share_df['market_share'] >= 0).all()
            assert (market_share_df['market_share'] <= 100).all()
            
            # Market shares for each category should sum to ~100%
            for category in market_share_df['category'].unique():
                category_shares = market_share_df[market_share_df['category'] == category]['market_share'].sum()
                assert abs(category_shares - 100) < 0.01  # Allow small floating point errors
    
    def test_get_top_performers(self, sample_data):
        """Test top performers identification"""
        analyzer = DataAnalyzer(sample_data)
        
        # Test by registrations
        top_registrations = analyzer.get_top_performers('registrations', 5)
        if not top_registrations.empty:
            assert len(top_registrations) <= 5
            assert 'registrations' in top_registrations.columns
        
        # Test by YoY growth
        top_growth = analyzer.get_top_performers('yoy_growth', 5)
        if not top_growth.empty:
            assert len(top_growth) <= 5
    
    def test_generate_insights(self, sample_data):
        """Test insights generation"""
        analyzer = DataAnalyzer(sample_data)
        insights = analyzer.generate_insights()
        
        assert isinstance(insights, dict)
        assert len(insights) > 0
        
        # Should not have error if data is valid
        assert 'error' not in insights or insights['error'] != "No data available for analysis"

class TestAdvancedAnalytics:
    """Test advanced analytics functionality"""
    
    def test_market_concentration(self, sample_data):
        """Test market concentration calculation"""
        analytics = AdvancedAnalytics(sample_data)
        concentration = analytics.calculate_market_concentration()
        
        assert isinstance(concentration, dict)
        
        for category, metrics in concentration.items():
            assert 'hhi' in metrics
            assert 'cr3' in metrics
            assert 'cr5' in metrics
            assert 'number_of_players' in metrics
            
            # HHI should be between 0 and 10000
            assert 0 <= metrics['hhi'] <= 10000
            
            # Concentration ratios should be between 0 and 100
            assert 0 <= metrics['cr3'] <= 100
            assert 0 <= metrics['cr5'] <= 100
    
    def test_growth_volatility(self, sample_data):
        """Test growth volatility calculation"""
        analytics = AdvancedAnalytics(sample_data)
        volatility_df = analytics.calculate_growth_volatility()
        
        if not volatility_df.empty:
            assert 'mean_growth' in volatility_df.columns
            assert 'std_growth' in volatility_df.columns
            assert 'cv_growth' in volatility_df.columns
            
            # Standard deviation should be non-negative
            assert (volatility_df['std_growth'] >= 0).all()
    
    def test_cagr_calculation(self, sample_data):
        """Test CAGR calculation"""
        analytics = AdvancedAnalytics(sample_data)
        cagr_df = analytics.calculate_compound_annual_growth_rate()
        
        if not cagr_df.empty:
            assert 'cagr' in cagr_df.columns
            assert 'start_year' in cagr_df.columns
            assert 'end_year' in cagr_df.columns
            
            # CAGR should be reasonable (between -100% and 1000%)
            assert (cagr_df['cagr'] >= -100).all()
            assert (cagr_df['cagr'] <= 1000).all()
    
    def test_emerging_players(self, sample_data):
        """Test emerging players identification"""
        analytics = AdvancedAnalytics(sample_data)
        emerging_df = analytics.identify_emerging_players(growth_threshold=10.0, share_threshold=0.1)
        
        # Should return dataframe (may be empty)
        assert isinstance(emerging_df, pd.DataFrame)
        
        if not emerging_df.empty:
            assert 'cagr' in emerging_df.columns
            assert 'market_share' in emerging_df.columns
            
            # All players should meet the thresholds
            assert (emerging_df['cagr'] >= 10.0).all()
            assert (emerging_df['market_share'] >= 0.1).all()

class TestForecastingEngine:
    """Test forecasting functionality"""
    
    def test_forecasting_initialization(self, sample_data):
        """Test forecasting engine initialization"""
        forecasting = ForecastingEngine(sample_data)
        
        if not forecasting.time_series.empty:
            assert 'period' in forecasting.time_series.columns
            assert 'registrations' in forecasting.time_series.columns
    
    def test_linear_forecast(self, sample_data):
        """Test linear trend forecasting"""
        forecasting = ForecastingEngine(sample_data)
        forecast_df = forecasting.simple_linear_forecast(periods_ahead=4)
        
        if not forecast_df.empty:
            assert len(forecast_df) == 4
            assert 'forecast_registrations' in forecast_df.columns
            assert 'year_quarter' in forecast_df.columns
            
            # Forecasts should be non-negative
            assert (forecast_df['forecast_registrations'] >= 0).all()
    
    def test_moving_average_forecast(self, sample_data):
        """Test moving average forecasting"""
        forecasting = ForecastingEngine(sample_data)
        forecast_df = forecasting.moving_average_forecast(window=4, periods_ahead=4)
        
        if not forecast_df.empty:
            assert len(forecast_df) == 4
            assert 'forecast_registrations' in forecast_df.columns
            
            # All forecasts should be the same (moving average)
            assert forecast_df['forecast_registrations'].nunique() == 1

class TestVahanDataScraper:
    """Test web scraping functionality"""
    
    @patch('vehicle_dashboard.webdriver.Chrome')
    def test_scraper_initialization(self, mock_chrome):
        """Test scraper initialization"""
        scraper = VahanDataScraper()
        
        assert scraper.session is not None
        assert scraper.chrome_options is not None
    
    def test_generate_sample_data(self):
        """Test sample data generation"""
        scraper = VahanDataScraper()
        sample_df = scraper._generate_sample_data()
        
        assert not sample_df.empty
        assert 'year' in sample_df.columns
        assert 'quarter' in sample_df.columns
        assert 'state' in sample_df.columns
        assert 'category' in sample_df.columns
        assert 'manufacturer' in sample_df.columns
        assert 'registrations' in sample_df.columns
        
        # Check data validity
        assert (sample_df['year'] >= 2022).all()
        assert (sample_df['quarter'].isin([1, 2, 3, 4])).all()
        assert (sample_df['registrations'] >= 0).all()
    
    @patch('vehicle_dashboard.webdriver.Chrome')
    def test_scrape_with_mock(self, mock_chrome, mock_scraper_response):
        """Test scraping with mocked browser"""
        # Setup mock
        mock_driver = Mock()
        mock_chrome.return_value = mock_driver
        
        scraper = VahanDataScraper()
        
        # Mock the _extract_dashboard_data method
        with patch.object(scraper, '_extract_dashboard_data', return_value=mock_scraper_response):
            result_df = scraper.scrape_vehicle_data()
        
        assert isinstance(result_df, pd.DataFrame)

class TestRegistrationRecord:
    """Test registration record data class"""
    
    def test_record_creation(self):
        """Test registration record creation"""
        record = RegistrationRecord(
            year=2024,
            quarter=1,
            state="MAHARASHTRA",
            category="2W",
            manufacturer="HERO MOTOCORP LTD",
            registrations=5000
        )
        
        assert record.year == 2024
        assert record.quarter == 1
        assert record.period == "2024-Q1"
        assert record.timestamp is not None
    
    def test_record_validation(self):
        """Test record validation"""
        # Valid record
        valid_record = RegistrationRecord(2024, 1, "STATE", "2W", "MANUFACTURER", 1000)
        assert valid_record.validate() == True
        
        # Invalid year
        invalid_year = RegistrationRecord(2050, 1, "STATE", "2W", "MANUFACTURER", 1000)
        assert invalid_year.validate() == False
        
        # Invalid quarter
        invalid_quarter = RegistrationRecord(2024, 5, "STATE", "2W", "MANUFACTURER", 1000)
        assert invalid_quarter.validate() == False
        
        # Negative registrations
        negative_registrations = RegistrationRecord(2024, 1, "STATE", "2W", "MANUFACTURER", -100)
        assert negative_registrations.validate() == False

class TestReportGenerator:
    """Test report generation functionality"""
    
    def test_executive_summary(self, sample_data):
        """Test executive summary generation"""
        report_gen = ReportGenerator(sample_data)
        summary = report_gen.generate_executive_summary()
        
        assert isinstance(summary, dict)
        assert 'total_registrations' in summary
        assert 'time_period' in summary
        assert 'categories_analyzed' in summary
        assert 'manufacturers_analyzed' in summary
        
        # Values should be reasonable
        assert summary['total_registrations'] > 0
        assert summary['categories_analyzed'] > 0
        assert summary['manufacturers_analyzed'] > 0
    
    def test_risk_assessment(self, sample_data):
        """Test risk assessment generation"""
        report_gen = ReportGenerator(sample_data)
        risks = report_gen.generate_risk_assessment()
        
        assert isinstance(risks, dict)
        assert 'high_risk' in risks
        assert 'medium_risk' in risks
        assert 'low_risk' in risks
        assert 'opportunities' in risks
        
        # Each should be a list
        for risk_type in risks.values():
            assert isinstance(risk_type, list)
    
    def test_export_to_json(self, sample_data, temp_db):
        """Test JSON export functionality"""
        report_gen = ReportGenerator(sample_data)
        
        # Use temporary directory
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            json_file = os.path.join(temp_dir, "test_report.json")
            result_file = report_gen.export_to_json(json_file)
            
            assert os.path.exists(result_file)
            
            # Verify JSON structure
            with open(result_file, 'r') as f:
                report_data = json.load(f)
            
            assert 'metadata' in report_data
            assert 'executive_summary' in report_data
            assert 'risk_assessment' in report_data

class TestIntegration:
    """Integration tests for complete workflows"""
    
    def test_end_to_end_analysis_pipeline(self, temp_db):
        """Test complete analysis pipeline"""
        # Generate sample data
        scraper = VahanDataScraper()
        sample_df = scraper._generate_sample_data()
        
        # Validate data
        cleaned_df, issues = DataValidator.validate_dataframe(sample_df)
        assert not cleaned_df.empty
        
        # Store in database
        db_manager = DatabaseManager(temp_db)
        db_manager.save_data(cleaned_df)
        
        # Load and analyze
        loaded_df = db_manager.load_data()
        analyzer = DataAnalyzer(loaded_df)
        
        # Generate insights
        insights = analyzer.generate_insights()
        assert isinstance(insights, dict)
        assert len(insights) > 0
        
        # Advanced analytics
        analytics = AdvancedAnalytics(loaded_df)
        concentration = analytics.calculate_market_concentration()
        assert isinstance(concentration, dict)
        
        # Forecasting
        forecasting = ForecastingEngine(loaded_df)
        forecast = forecasting.simple_linear_forecast()
        assert isinstance(forecast, pd.DataFrame)
        
        # Report generation
        report_gen = ReportGenerator(loaded_df)
        summary = report_gen.generate_executive_summary()
        assert isinstance(summary, dict)
    
    def test_error_handling_empty_data(self, temp_db):
        """Test error handling with empty data"""
        empty_df = pd.DataFrame()
        
        # Should handle empty data gracefully
        cleaned_df, issues = DataValidator.validate_dataframe(empty_df)
        assert cleaned_df.empty
        assert len(issues) > 0
        
        # Analyzer should handle empty data
        analyzer = DataAnalyzer(empty_df)
        insights = analyzer.generate_insights()
        assert "error" in insights or len(insights) == 0
        
        # Advanced analytics should handle empty data
        analytics = AdvancedAnalytics(empty_df)
        concentration = analytics.calculate_market_concentration()
        assert isinstance(concentration, dict)
    
    def test_data_consistency_across_modules(self, sample_data):
        """Test data consistency across different modules"""
        # Validate data
        cleaned_df, _ = DataValidator.validate_dataframe(sample_data)
        
        # Analyze with different modules
        analyzer = DataAnalyzer(cleaned_df)
        analytics = AdvancedAnalytics(cleaned_df)
        
        # Check if both modules see the same data shape
        assert len(analyzer.df) == len(analytics.df)
        assert set(analyzer.df.columns) == set(analytics.df.columns)
        
        # Check if calculations are consistent
        total_from_analyzer = analyzer.df['registrations'].sum()
        total_from_analytics = analytics.df['registrations'].sum()
        assert total_from_analyzer == total_from_analytics

class TestPerformance:
    """Performance and load testing"""
    
    def test_large_dataset_performance(self):
        """Test performance with large dataset"""
        # Generate larger dataset
        np.random.seed(42)
        large_data = []
        
        states = ['STATE_' + str(i) for i in range(20)]
        manufacturers = ['MANUFACTURER_' + str(i) for i in range(50)]
        categories = ['2W', '3W', '4W', 'LMV', 'HMV']
        
        for year in range(2020, 2025):
            for quarter in [1, 2, 3, 4]:
                for state in states:
                    for category in categories:
                        for manufacturer in manufacturers[:10]:  # Limit to avoid too much data
                            registrations = np.random.randint(100, 10000)
                            large_data.append({
                                'year': year,
                                'quarter': quarter,
                                'state': state,
                                'category': category,
                                'manufacturer': manufacturer,
                                'registrations': registrations
                            })
        
        large_df = pd.DataFrame(large_data)
        
        # Test validation performance
        import time
        start_time = time.time()
        cleaned_df, issues = DataValidator.validate_dataframe(large_df)
        validation_time = time.time() - start_time
        
        assert validation_time < 10.0  # Should complete within 10 seconds
        assert not cleaned_df.empty
        
        # Test analysis performance
        start_time = time.time()
        analyzer = DataAnalyzer(cleaned_df)
        yoy_df = analyzer.calculate_yoy_growth()
        analysis_time = time.time() - start_time
        
        assert analysis_time < 15.0  # Should complete within 15 seconds
    
    def test_memory_usage(self, sample_data):
        """Test memory usage patterns"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Perform multiple operations
        for _ in range(10):
            analyzer = DataAnalyzer(sample_data.copy())
            _ = analyzer.calculate_yoy_growth()
            _ = analyzer.calculate_qoq_growth()
            _ = analyzer.get_market_share()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024

class TestDataQuality:
    """Test data quality and validation rules"""
    
    def test_data_completeness(self, sample_data):
        """Test data completeness checks"""
        # Remove some data to test completeness
        incomplete_df = sample_data.copy()
        incomplete_df.loc[:10, 'manufacturer'] = None
        incomplete_df.loc[11:20, 'registrations'] = None
        
        cleaned_df, issues = DataValidator.validate_dataframe(incomplete_df)
        
        # Should handle missing data appropriately
        assert len(cleaned_df) < len(incomplete_df)
    
    def test_data_consistency(self, sample_data):
        """Test data consistency rules"""
        # Add inconsistent data
        inconsistent_df = sample_data.copy()
        
        # Add impossible combinations
        inconsistent_df.loc[0, 'year'] = 1990  # Too old
        inconsistent_df.loc[1, 'quarter'] = 5  # Invalid quarter
        inconsistent_df.loc[2, 'registrations'] = -500  # Negative registrations
        
        cleaned_df, issues = DataValidator.validate_dataframe(inconsistent_df)
        
        # Should remove inconsistent records
        assert (cleaned_df['year'] >= 2020).all()
        assert (cleaned_df['quarter'].isin([1, 2, 3, 4])).all()
        assert (cleaned_df['registrations'] >= 0).all()
    
    def test_outlier_detection_accuracy(self):
        """Test outlier detection accuracy"""
        # Create data with known outliers
        normal_data = np.random.normal(1000, 100, 1000)
        outlier_data = [10000, 15000, 20000]  # Clear outliers
        
        test_data = pd.DataFrame({
            'registrations': np.concatenate([normal_data, outlier_data])
        })
        
        outliers = DataValidator.detect_outliers(test_data)
        
        # Should detect the obvious outliers
        assert len(outliers) >= 3
        assert 10000 in outliers['registrations'].values
        assert 15000 in outliers['registrations'].values
        assert 20000 in outliers['registrations'].values

class TestConfigValidation:
    """Test configuration validation"""
    
    def test_config_loading(self):
        """Test configuration loading"""
        config = get_config()
        
        assert config.APP_NAME is not None
        assert config.DATABASE_PATH is not None
        assert isinstance(config.VEHICLE_CATEGORIES, dict)
        assert isinstance(config.MAJOR_MANUFACTURERS, list)
    
    def test_config_validation(self):
        """Test configuration validation"""
        config = TestingConfig()
        issues = config.validate_config()
        
        # Should have no critical issues
        assert isinstance(issues, list)
    
    def test_feature_flags(self):
        """Test feature flag functionality"""
        config = TestingConfig()
        
        assert isinstance(config.FEATURE_FLAGS, dict)
        assert 'enable_real_time_scraping' in config.FEATURE_FLAGS
        assert config.FEATURE_FLAGS['enable_real_time_scraping'] == False  # Disabled in testing

# Utility functions for testing
def create_test_data_with_trends():
    """Create test data with specific trends for testing"""
    data = []
    
    # Create data with clear growth trend
    base_registrations = 1000
    growth_rate = 0.1  # 10% quarterly growth
    
    for year in [2022, 2023, 2024]:
        for quarter in [1, 2, 3, 4]:
            registrations = int(base_registrations * (1 + growth_rate) ** ((year - 2022) * 4 + quarter - 1))
            
            data.append({
                'year': year,
                'quarter': quarter,
                'state': 'TEST_STATE',
                'category': '2W',
                'manufacturer': 'TEST_MANUFACTURER',
                'registrations': registrations
            })
    
    return pd.DataFrame(data)

def run_performance_benchmark():
    """Run performance benchmark tests"""
    print("Running performance benchmarks...")
    
    # Generate large test dataset
    start_time = time.time()
    scraper = VahanDataScraper()
    large_sample = scraper._generate_sample_data()
    
    # Duplicate data to make it larger
    for _ in range(5):
        large_sample = pd.concat([large_sample, large_sample], ignore_index=True)
    
    data_generation_time = time.time() - start_time
    print(f"Data generation: {data_generation_time:.2f}s for {len(large_sample)} records")
    
    # Test analysis performance
    start_time = time.time()
    analyzer = DataAnalyzer(large_sample)
    yoy_df = analyzer.calculate_yoy_growth()
    qoq_df = analyzer.calculate_qoq_growth()
    market_share_df = analyzer.get_market_share()
    analysis_time = time.time() - start_time
    
    print(f"Analysis time: {analysis_time:.2f}s")
    print(f"YoY records: {len(yoy_df)}")
    print(f"QoQ records: {len(qoq_df)}")
    print(f"Market share records: {len(market_share_df)}")

# Test execution functions
def run_quick_tests():
    """Run quick smoke tests"""
    import subprocess
    import sys
    
    print("Running quick tests...")
    
    # Run basic import tests
    try:
        from vehicle_dashboard import VahanDataScraper, DataAnalyzer
        from data_processor import DataValidator, AdvancedAnalytics
        print("✅ All imports successful")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test sample data generation
    try:
        scraper = VahanDataScraper()
        sample_df = scraper._generate_sample_data()
        assert not sample_df.empty
        print(f"✅ Sample data generation: {len(sample_df)} records")
    except Exception as e:
        print(f"❌ Sample data generation failed: {e}")
        return False
    
    # Test basic analysis
    try:
        analyzer = DataAnalyzer(sample_df)
        insights = analyzer.generate_insights()
        assert isinstance(insights, dict)
        print("✅ Basic analysis successful")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return False
    
    print("✅ All quick tests passed!")
    return True

def run_full_test_suite():
    """Run complete test suite"""
    import subprocess
    import sys
    
    print("Running full test suite...")
    
    try:
        # Run pytest with verbose output
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            'test_dashboard.py', 
            '-v', 
            '--tb=short',
            '--durations=10'
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        # Run quick tests
        success = run_quick_tests()
        sys.exit(0 if success else 1)
    
    elif len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        # Run performance benchmarks
        run_performance_benchmark()
    
    else:
        # Run full test suite
        print("🧪 Vehicle Registration Dashboard - Test Suite")
        print("=" * 60)
        
        success = run_full_test_suite()
        
        if success:
            print("\n🎉 All tests passed successfully!")
        else:
            print("\n❌ Some tests failed. Check output above.")
        
        sys.exit(0 if success else 1)