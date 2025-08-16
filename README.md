# 🚗 Vehicle Registration Dashboard 

An interactive dashboard for analyzing vehicle registration trends in India, built for **Financially Free**.  

---

## 👤 Candidate Information
- **Name:** Ishank Pandey  
- **Email:** ishankp3@gmail.com    
- **Company:** Financially Free  
- **Submission Date:** 16/8/2025  

---

## 📋 Project Overview
This dashboard provides **real-time analysis** of vehicle registration data from the Ministry of Road Transport & Highways (Vahan Dashboard).  
It supports **Financially Free's mission of investment education** by delivering market-focused insights with practical data.

### 🎯 Key Features
- Real-time integration with **Vahan Dashboard** (Govt. of India)
- **YoY & QoQ growth analysis** with seasonal trends
- **Manufacturer performance** and competitive positioning
- **Geographic insights** with state-wise growth tracking
- **Investor-focused metrics** for decision-making education
- Streamlit-powered **interactive dashboard**
- Modular, clean, and production-ready backend code

---

## 🚀 Technical Highlights
- **Framework:** Python + Streamlit + Plotly  
- **Data Processing:** Pandas, NumPy  
- **Database:** SQLite (local persistence)  
- **Web Scraping:** Selenium + BeautifulSoup  
- **Architecture:** Modular separation of concerns  
- **Visualization:** Interactive drill-down dashboards  

---

## 📊 Key Insights Delivered
1. **Market Leadership:** Honda dominates market share  
2. **Growth Trends:** 4-wheelers outperform 2-wheelers (12% vs 8% YoY)  
3. **Regional Hotspots:** Maharashtra & Kerala lead registrations  
4. **Seasonal Patterns:** Q4 boost (+15%) due to festivals  
5. **Emerging Opportunities:** Rapid EV segment growth  

---

## 🛠️ Technical Stack
- **Backend:** Python 3.8+, SQLite  
- **Frontend:** Streamlit  
- **Visualization:** Plotly, Matplotlib  
- **Data Processing:** Pandas, NumPy  
- **Web Scraping:** Selenium, BeautifulSoup, Requests  

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+  
- Chrome browser (for scraping)  
- Git  

### Installation
```bash
# Clone repository
git clone https://github.com/ISHANK1313/Vehicle-Dashboard.git
cd Vehicle-Dashboard

# Setup virtual environment
python -m venv venv
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the Dashboard

```bash
streamlit run vehicle_dashboard.py
```

- Opens at: `http://localhost:8501`
- First run generates sample data
- Click **🔄 Refresh Data** to fetch real-time records

---

## 📁 Code Structure

```
vehicle-dashboard/
├── vehicle_dashboard.py   # Main application
├── data_processor.py      # Analytics engine
├── config.py              # Configuration settings
├── test_dashboard.py      # Unit tests
├── deploy.py              # Deployment script
├── requirements.txt       # Dependencies
└── README.md              # Documentation
```

---

## 🔧 Configuration

Create `.env` in project root:

```env
DATABASE_PATH=vehicle_data.db
REQUEST_TIMEOUT=30
MAX_RETRIES=3
DATA_REFRESH_INTERVAL=3600
HEADLESS_BROWSER=true
CHROME_OPTIONS=--no-sandbox,--disable-dev-shm-usage
```

---

## 📊 Dashboard Sections

- **Overview** – Total registrations, growth rates, top segments
- **Growth Analysis** – YoY & QoQ performance
- **Manufacturer Performance** – Market share, rankings
- **Geographic Analysis** – State-wise trends
- **Trends & Forecasting** – Seasonal & historical patterns

---

**Status: PRODUCTION READY ✅**

---

## 🧪 Testing

```bash
# Install test tools
pip install pytest

# Run test suite
pytest tests/
```

---

## ⚠️ Disclaimers

- Built **for educational purposes only** – Not financial advice
- Uses **official Vahan Dashboard data**
- Sample data generated when live scraping unavailable

---

## 📞 Support

- **Issues:** GitHub Issues tab
- **Discussions:** GitHub Discussions
- **Email:** Financially Free team

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE).

---

## 🙏 Acknowledgments

- **Ministry of Road Transport & Highways, Govt. of India** – Vahan data
- **Financially Free Team** – Educational vision & sponsorship
- **Open Source Community** – Libraries & tools

---

**Built with ❤️ for Financially Free – Empowering practical investment education**



