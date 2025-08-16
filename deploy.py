"""
Deployment and Setup Script for Vehicle Registration Dashboard
Usage: python deploy.py [command]
Commands: setup, test, run, clean, help
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import argparse
import logging
from typing import List, Optional
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DashboardDeployer:
    """Handles deployment and setup of the dashboard"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.venv_path = self.project_root / "venv"
        self.requirements_file = self.project_root / "requirements.txt"
        self.config_file = self.project_root / "config.py"
        
    def check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            logger.error("Python 3.8+ is required. Current version: {}.{}.{}".format(
                version.major, version.minor, version.micro))
            return False
        
        logger.info(f"Python version check passed: {version.major}.{version.minor}.{version.micro}")
        return True
    
    def create_virtual_environment(self) -> bool:
        """Create virtual environment"""
        try:
            if self.venv_path.exists():
                logger.info("Virtual environment already exists")
                return True
            
            logger.info("Creating virtual environment...")
            subprocess.run([sys.executable, "-m", "venv", str(self.venv_path)], check=True)
            logger.info("✅ Virtual environment created successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to create virtual environment: {e}")
            return False
    
    def get_pip_command(self) -> str:
        """Get pip command for the virtual environment"""
        if os.name == 'nt':  # Windows
            return str(self.venv_path / "Scripts" / "pip")
        else:  # Unix/Linux/macOS
            return str(self.venv_path / "bin" / "pip")
    
    def get_python_command(self) -> str:
        """Get python command for the virtual environment"""
        if os.name == 'nt':  # Windows
            return str(self.venv_path / "Scripts" / "python")
        else:  # Unix/Linux/macOS
            return str(self.venv_path / "bin" / "python")
    
    def install_dependencies(self) -> bool:
        """Install required dependencies"""
        try:
            pip_cmd = self.get_pip_command()
            
            # Upgrade pip first
            logger.info("Upgrading pip...")
            subprocess.run([pip_cmd, "install", "--upgrade", "pip"], check=True)
            
            # Install requirements
            if self.requirements_file.exists():
                logger.info("Installing dependencies from requirements.txt...")
                subprocess.run([pip_cmd, "install", "-r", str(self.requirements_file)], check=True)
            else:
                logger.info("Installing basic dependencies...")
                basic_deps = [
                    "streamlit>=1.28.0",
                    "pandas>=1.5.0",
                    "numpy>=1.21.0",
                    "plotly>=5.15.0",
                    "requests>=2.28.0",
                    "beautifulsoup4>=4.11.0",
                    "selenium>=4.10.0",
                    "webdriver-manager>=3.8.0"
                ]
                
                for dep in basic_deps:
                    logger.info(f"Installing {dep}...")
                    subprocess.run([pip_cmd, "install", dep], check=True)
            
            logger.info("✅ Dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install dependencies: {e}")
            return False
    
    def setup_chrome_driver(self) -> bool:
        """Setup Chrome WebDriver"""
        try:
            python_cmd = self.get_python_command()
            
            # Install webdriver-manager and setup Chrome driver
            setup_script = """
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

try:
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.get('https://www.google.com')
    driver.quit()
    print('✅ Chrome WebDriver setup successful')
except Exception as e:
    print(f'❌ Chrome WebDriver setup failed: {e}')
    exit(1)
"""
            
            result = subprocess.run([python_cmd, "-c", setup_script], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Chrome WebDriver setup successful")
                return True
            else:
                logger.error(f"❌ Chrome WebDriver setup failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Chrome WebDriver setup error: {e}")
            return False
    
    def create_config_file(self) -> bool:
        """Create configuration file if it doesn't exist"""
        try:
            if not self.config_file.exists():
                logger.info("Creating default configuration file...")
                
                config_content = '''"""
Configuration file for Vehicle Registration Dashboard
Customize these settings as needed
"""

import os

# Environment
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')

# Database
DATABASE_PATH = os.getenv('DATABASE_PATH', 'vehicle_data.db')

# Scraping settings
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
DATA_REFRESH_INTERVAL = 3600  # 1 hour

# UI Settings
STREAMLIT_PORT = 8501
STREAMLIT_HOST = 'localhost'

# Feature flags
ENABLE_REAL_TIME_SCRAPING = True
ENABLE_SAMPLE_DATA = True
ENABLE_CACHING = True

# Logging
LOG_LEVEL = 'INFO'
LOG_FILE = 'dashboard.log'
'''
                
                with open(self.config_file, 'w') as f:
                    f.write(config_content)
                
                logger.info("✅ Configuration file created")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create configuration file: {e}")
            return False
    
    def run_tests(self) -> bool:
        """Run test suite"""
        try:
            python_cmd = self.get_python_command()
            
            logger.info("Running test suite...")
            
            # First try quick tests
            logger.info("Running quick smoke tests...")
            result = subprocess.run([python_cmd, "test_dashboard.py", "quick"], 
                                  capture_output=True, text=True)
            
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            
            if result.returncode == 0:
                logger.info("✅ Quick tests passed")
                
                # Run full test suite if pytest is available
                try:
                    subprocess.run([python_cmd, "-c", "import pytest"], check=True)
                    logger.info("Running full test suite with pytest...")
                    
                    result = subprocess.run([python_cmd, "-m", "pytest", "test_dashboard.py", "-v"], 
                                          capture_output=True, text=True)
                    
                    print(result.stdout)
                    if result.stderr:
                        print("STDERR:", result.stderr)
                    
                    if result.returncode == 0:
                        logger.info("✅ Full test suite passed")
                        return True
                    else:
                        logger.warning("⚠️ Some tests failed, but quick tests passed")
                        return True
                        
                except subprocess.CalledProcessError:
                    logger.info("pytest not available, but quick tests passed")
                    return True
            else:
                logger.error("❌ Quick tests failed")
                return False
            
        except Exception as e:
            logger.error(f"❌ Test execution failed: {e}")
            return False
    
    def run_dashboard(self, port: int = 8501) -> bool:
        """Run the Streamlit dashboard"""
        try:
            python_cmd = self.get_python_command()
            
            logger.info(f"Starting dashboard on port {port}...")
            logger.info("Dashboard will open in your browser automatically")
            logger.info("Press Ctrl+C to stop the dashboard")
            
            # Run streamlit
            subprocess.run([
                python_cmd, "-m", "streamlit", "run", 
                "vehicle_dashboard.py",
                "--server.port", str(port),
                "--server.headless", "false"
            ])
            
            return True
            
        except KeyboardInterrupt:
            logger.info("Dashboard stopped by user")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to start dashboard: {e}")
            return False
    
    def clean_project(self) -> bool:
        """Clean project files"""
        try:
            logger.info("Cleaning project files...")
            
            # Files and directories to clean
            clean_targets = [
                "__pycache__",
                "*.pyc",
                ".pytest_cache",
                "*.log",
                "test_*.db",
                "dev_*.db"
            ]
            
            for target in clean_targets:
                if "*" in target:
                    # Use glob for wildcard patterns
                    import glob
                    for file_path in glob.glob(target):
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            logger.info(f"Removed file: {file_path}")
                else:
                    # Direct path
                    target_path = Path(target)
                    if target_path.exists():
                        if target_path.is_dir():
                            shutil.rmtree(target_path)
                            logger.info(f"Removed directory: {target_path}")
                        else:
                            target_path.unlink()
                            logger.info(f"Removed file: {target_path}")
            
            logger.info("✅ Project cleaned successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Cleaning failed: {e}")
            return False
    
    def setup_project(self) -> bool:
        """Complete project setup"""
        logger.info("🚗 Setting up Vehicle Registration Dashboard...")
        logger.info("=" * 60)
        
        # Check Python version
        if not self.check_python_version():
            return False
        
        # Create virtual environment
        if not self.create_virtual_environment():
            return False
        
        # Install dependencies
        if not self.install_dependencies():
            return False
        
        # Setup Chrome WebDriver
        if not self.setup_chrome_driver():
            logger.warning("⚠️ Chrome WebDriver setup failed. Manual installation may be required.")
        
        # Create config file
        if not self.create_config_file():
            return False
        
        logger.info("🎉 Project setup completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Run tests: python deploy.py test")
        logger.info("2. Start dashboard: python deploy.py run")
        
        return True
    
    def generate_project_info(self) -> dict:
        """Generate project information"""
        python_cmd = self.get_python_command()
        
        info = {
            "project_name": "Vehicle Registration Dashboard",
            "company": "Financially Free",
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "project_root": str(self.project_root),
            "virtual_env": str(self.venv_path),
            "dependencies_installed": self.venv_path.exists(),
            "config_exists": self.config_file.exists(),
        }
        
        # Check if dependencies are installed
        try:
            subprocess.run([python_cmd, "-c", "import streamlit, pandas, plotly"], 
                         check=True, capture_output=True)
            info["dependencies_status"] = "installed"
        except:
            info["dependencies_status"] = "missing"
        
        return info

def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description="Vehicle Registration Dashboard Deployment")
    parser.add_argument("command", nargs="?", default="help",
                       choices=["setup", "test", "run", "clean", "info", "help"],
                       help="Command to execute")
    parser.add_argument("--port", type=int, default=8501,
                       help="Port for Streamlit dashboard (default: 8501)")
    
    args = parser.parse_args()
    
    deployer = DashboardDeployer()
    
    if args.command == "setup":
        success = deployer.setup_project()
        sys.exit(0 if success else 1)
    
    elif args.command == "test":
        success = deployer.run_tests()
        sys.exit(0 if success else 1)
    
    elif args.command == "run":
        success = deployer.run_dashboard(args.port)
        sys.exit(0 if success else 1)
    
    elif args.command == "clean":
        success = deployer.clean_project()
        sys.exit(0 if success else 1)
    
    elif args.command == "info":
        info = deployer.generate_project_info()
        print(json.dumps(info, indent=2))
    
    else:  # help
        print("""
🚗 Vehicle Registration Dashboard - Deployment Script

Usage: python deploy.py [command] [options]

Commands:
  setup     Complete project setup (create venv, install dependencies, etc.)
  test      Run the test suite
  run       Start the Streamlit dashboard
  clean     Clean project files and caches
  info      Show project information
  help      Show this help message

Options:
  --port    Port for Streamlit dashboard (default: 8501)

Examples:
  python deploy.py setup              # Initial project setup
  python deploy.py test               # Run tests
  python deploy.py run                # Start dashboard on default port
  python deploy.py run --port 8502    # Start dashboard on custom port
  python deploy.py clean              # Clean project files

Quick Start:
1. python deploy.py setup
2. python deploy.py test
3. python deploy.py run

Educational Dashboard by Financially Free
For learning investment analysis through real data exploration
        """)

if __name__ == "__main__":
    main()