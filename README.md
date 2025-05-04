# Social Media Marketing Analytics Capstone

This repository contains the complete implementation of an end-to-end social media analytics and predictive modeling system, developed as a data science capstone project. It supports automated data extraction, transformation, modeling, and dashboarding for marketing agencies managing multiple platforms.

---

## Project Structure

```
Code/
├── API-Extraction/
│   ├── email_api_capstone_final.py
│   ├── facebook_paid_ad_code_capstone_final.py
│   ├── fb_insights_code_capstone_final.py
│   ├── ig_insights_code_capstone_final.py
│   ├── linkedin_api_capstone_final.py
│   └── website_api_code_capstone_final.py
├── ETL/
│   └── SocialMediaETLPipeline_code_capstone_complete.py
├── Modeling/
│   ├── predictive_analytics_code_capstone_final_complete.py
│   └── predictive_dashboard_code_capston_final_complete.py
Original data/
├── client_mapping.csv
├── dimensions/
├── facts/
│   ├── vw_daily_metrics.csv
│   ├── vw_mailchimp_campaigns.csv
│   ├── vw_monthly_metrics.csv
│   ├── vw_platform_comparison.csv
│   └── vw_platform_top_content.csv
```

---

## Features

- **Multi-platform integration** (FB, IG, LI, Mailchimp, GA4)
- **ETL pipeline** for metric standardization and unification
- **Random Forest models** for engagement classification and forecasting
- **Dash dashboard** for smart summaries, forecasting charts, and platform comparisons

---

## Setup Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set up environment variables:
   - API keys required for Meta (Facebook/Instagram), LinkedIn, Mailchimp, Google Analytics

3. Run individual scripts as needed
   ```bash
   python Code/ETL/SocialMediaETLPipeline_code_capstone_complete.py
   ```
4. Launch the dashboard:
   ```bash
   python Code/Modeling/predictive_dashboard_code_capston_final_complete.py
   ```

---


## Outputs

- Unified CSV files with harmonized metrics
- Forecasted engagement scores, insights, and cross platform analysis (per client-platform)
- Interactive dashboard with exportable insights