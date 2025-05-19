# Android Malware Detection through Network Traffic Analysis

This repository contains my submission for the DT4003 Maths for Data Science module assessment.

## Assessment Overview

This project investigates network traffic patterns to detect malware in Android applications. As a data scientist for AndroiHypo (a telecommunications company), I analyze network layer features to evaluate two key hypotheses:

1. The probability of having benign traffic given DNS query times > 5 and TCP packets > 40 is at least 9%
2. There is a massive traffic volume bytes difference between benign and malicious traffic types

The analysis includes data cleaning, visualization, and statistical hypothesis testing to evaluate these claims.

## Repository Structure

```
├── analysis_plots/                      # Visualizations for hypothesis testing
│   ├── analysis_results.json            # Statistical results in JSON format
│   ├── dns_query_distribution.png       # Distribution of DNS queries
│   ├── dns_tcp_scatter.png              # Scatter plot of DNS vs TCP
│   ├── tcp_packets_distribution.png     # Distribution of TCP packets
│   ├── volume_bytes_boxplot.png         # Boxplot of traffic volume
│   └── volume_bytes_cdf.png             # Cumulative distribution of traffic volume
│
├── cleaning_visualisations/             # Data cleaning process visualizations
│   ├── cleaning_report.txt              # Text report on data cleaning process
│   ├── missing_values.png               # Visualization of missing values
│   ├── row_counts.png                   # Count of rows before/after cleaning
│   └── traffic_types.png                # Distribution of traffic types
│
├── data/                                # Dataset files
│   ├── android_traffic-not_clean.csv    # Original dataset before cleaning
│   └── android_traffic_clean.csv        # Clean dataset for analysis
│
├── docs/                                # Documentation
│   ├── Android Traffic Analysis - Malware Detection through Network Features.docx   # Report (Word format)
│   ├── Android Traffic Analysis - Malware Detection through Network Features.md     # Report (Markdown)
│   └── DT4003 Maths for Data Science Assessment.md                                 # Assessment specification
│
├── analyse_data.py                      # Python script for data analysis and visualization
└── clean_data.py                        # Python script for data preprocessing
```

## Documentation

- [Assessment Specification](docs/DT4003%20Maths%20for%20Data%20Science%20Assessment.md) - Details the requirements and marking criteria for this assignment
- [Project Report](docs/Android%20Traffic%20Analysis%20-%20Malware%20Detection%20through%20Network%20Features.md) - The complete technical report analyzing the Android traffic data and testing the proposed hypotheses

## Analysis Process

1. **Data Cleaning** (`clean_data.py`): 
   - Processes the raw data from `android_traffic-not_clean.csv`
   - Handles missing values, outliers, and inconsistencies
   - Outputs the cleaned dataset to `android_traffic_clean.csv`
   - Generates visualizations in the `cleaning_visualisations` folder

2. **Data Analysis** (`analyse_data.py`):
   - Performs exploratory data analysis on the cleaned dataset
   - Tests the two hypotheses using statistical methods
   - Creates visualizations stored in the `analysis_plots` folder
   - Outputs analysis results to `analysis_results.json`

## Tools & Technologies

- Python for data processing and analysis
- Pandas for data manipulation
- Matplotlib and Seaborn for visualization
- SciPy and NumPy for statistical analysis

## How to Run

1. Clone this repository
2. Install required dependencies: `pip install -r requirements.txt`
3. Run data cleaning: `python clean_data.py`
4. Run data analysis: `python analyse_data.py`

---

This project was completed for the DT4003 Maths for Data Science module.
