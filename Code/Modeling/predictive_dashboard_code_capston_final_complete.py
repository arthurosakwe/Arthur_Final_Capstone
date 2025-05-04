"""
Capstone Project
Dashboard App
Author: Arthur
Version: 4.0
"""
#import libraries
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import os
import glob
import re
import socket
from xhtml2pdf import pisa
import io


# Initialize  Dash app
app = dash.Dash(__name__, external_stylesheets=['https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css'])
server = app.server
app.title = 'PMM Social Media Dashboard'

#note to add a couple more helper functions to optimize code

#data loading class
class DataLoader:
    def __init__(self, data_dir, model_dir):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.daily_metrics = None
        self.forecasts = {}
        self.correlations = {}
        self.insights = {}
        self.seasonal_patterns = {}
        self.cross_platform_opportunities = {}
        
        # Load all data
        self.load_all_data()
   
    
   #load metrics data and model outputs
    def load_all_data(self):
        try:
            #load metrics data
            self.daily_metrics = pd.read_csv(self.data_dir / 'vw_daily_metrics.csv')
            self.daily_metrics['full_date'] = pd.to_datetime(self.daily_metrics['full_date'])
            
            #load the seasonal patterns
            seasonal_pattern_files = glob.glob(str(self.model_dir / 'forecasts' / 'seasonal_patterns_*.csv'))
            for file_path in seasonal_pattern_files:
                
                file_name = os.path.basename(file_path)
                match = re.search(r'seasonal_patterns_(.+)\.csv', file_name)
                if match:
                    key = match.group(1)
                    self.seasonal_patterns[key] = pd.read_csv(file_path)
            
            #load forecast
            forecast_files = glob.glob(str(self.model_dir / 'forecasts' / 'forecast_*.csv'))
            for file_path in forecast_files:
                
                
                file_name = os.path.basename(file_path)
                match = re.search(r'forecast_rf_(.+)\.csv', file_name)
                if match:
                    key = match.group(1)
                    self.forecasts[key] = pd.read_csv(file_path)
                    #convert date columns - check for multiple possible column names
                    for date_col in ['date', 'week_start_date', 'weekly_date']:
                        if date_col in self.forecasts[key].columns:
                            self.forecasts[key][date_col] = pd.to_datetime(self.forecasts[key][date_col])

                        
            #load cross-platform opportunities
            opportunity_files = glob.glob(str(self.model_dir / 'forecasts' / 'cross_platform_opportunities_*.csv'))
            for file_path in opportunity_files:
                file_name = os.path.basename(file_path)
                match = re.search(r'cross_platform_opportunities_(.+)\.csv', file_name)
                if match:
                    key = match.group(1)
                    self.cross_platform_opportunities[key] = pd.read_csv(file_path)                                   
         
            #load correlations
            correlation_files = glob.glob(str(self.model_dir / 'forecasts' / 'correlation_*.csv'))
            for file_path in correlation_files:
                file_name = os.path.basename(file_path)
                match = re.search(r'correlation_(.+)\.csv', file_name)
                if match:
                    key = match.group(1)
                    self.correlations[key] = pd.read_csv(file_path, index_col=0)            
            return True
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False        

    #get list of clients from vw_daily_metrics file
    def get_clients(self):
        if self.daily_metrics is not None:
            return sorted(self.daily_metrics['Client_Name'].unique())
        return []
    #ggt list of platforms for each client from vw_daily_metrics file
    def get_client_platforms(self, client):
        if self.daily_metrics is not None:
            client_data = self.daily_metrics[self.daily_metrics['Client_Name'] == client]
            if not client_data.empty:
                return sorted(client_data['platform_name'].unique())
        return []
    
    def get_forecast(self, client, platform, metric='total_engagement'):
        client_key = client.replace(" ", "_").lower()
        platform_key = platform.replace(" ", "_").lower()
        key = f"{client_key}_{platform_key}_{metric}"
        if key in self.forecasts:
            return self.forecasts[key]
        # If not found, try matching keys that start with the normalized client and platform
        for k in self.forecasts.keys():
            if k.startswith(f"{client_key}_{platform_key}"):
                return self.forecasts[k]
        return None
        
    def get_correlation(self, client):
        client_key = client.replace(" ", "_").lower()
        if client_key in self.correlations:
            return self.correlations[client_key]
        return None
    
    #filter metrics, exclude frequency and other non-summable metrics
    def filter_metrics_for_calculation(self, data, category=None, name_contains=None):
        #define metrics that should be excluded from sum calculations
        exclude_from_sums = ['Frequency', 'CTR', 'CPC', 'CPM']
        condition = ~data['metric_name'].isin(exclude_from_sums)
        
        #add category filter if provided
        if category:
            condition &= (data['metric_category'] == category)
    
        #add name contains filter if provided
        if name_contains:
            condition &= data['metric_name'].str.contains(name_contains, case=False, na=False)
        return data[condition]
    
    #get seasonal patterns for a client
    def get_seasonal_patterns(self, client):
        client_key = client.replace(" ", "_").lower()
        if client_key in self.seasonal_patterns:
            return self.seasonal_patterns[client_key]
        return None
    
    #get historical data for a client and platform, excluding non-summable metrics
    def get_historical_data(self, client, platform, metric_category='Engagement'):
        if self.daily_metrics is None:
            return None
            
        #exclude these metrics
        exclude_from_sums = ['Frequency', 'CTR', 'CPC', 'CPM']
        
        filtered_data = self.daily_metrics[
            (self.daily_metrics['Client_Name'] == client) &
            (self.daily_metrics['platform_name'] == platform) &
            (self.daily_metrics['metric_category'] == metric_category) &
            (~self.daily_metrics['metric_name'].isin(exclude_from_sums))
        ].copy()
        
        if filtered_data.empty:
            return None
            
        daily_data = filtered_data.groupby('full_date')['value'].sum().reset_index()
        return daily_data
    
    #key takeawys for smart summry  - 3 prioritized for now
    def extract_key_takeaways(self, client):
        takeaways = []
        
        #forecast trend takeaway
        platforms = self.get_client_platforms(client)
        if platforms:
            forecast_data = self.get_forecast(client, platforms[0])
            if forecast_data is not None and not forecast_data.empty:
                
                
                #detect forecast column
                forecast_col = None
                if 'weekly_forecast' in forecast_data.columns:
                    forecast_col = 'weekly_forecast'
                elif 'forecast' in forecast_data.columns:
                    forecast_col = 'forecast'
                
                if forecast_col:
                    forecast_start = forecast_data[forecast_col].iloc[0]
                    forecast_end = forecast_data[forecast_col].iloc[-1]
                
                #make sure forecasts are valid
                if not pd.isna(forecast_start) and not pd.isna(forecast_end):
                    change_pct = (forecast_end - forecast_start) / (forecast_start + 1e-10) * 100
                    
                    direction = "rise" if change_pct > 0 else "decline"
                    takeaways.append({
                        #icon,title and desc
                        'icon': 'fa-chart-line',
                        'title': f"Engagement on {platforms[0]} expected to {direction}",
                        'description': f"{abs(change_pct):.1f}% {direction} in engagement predicted next month"
                    })

        
        #add seasonal pattern takeaway
        seasonal_data = self.get_seasonal_patterns(client)
        if seasonal_data is not None:
            #find best day
            best_day = seasonal_data.groupby('day')['value'].mean().idxmax()
            takeaways.append({
                #icon title and desc
                'icon': 'fa-calendar',
                'title': f"Best posting day: {best_day}",
                'description': f"Schedule key content on {best_day}s for maximum engagement"
            })

        #add takeaway if trend data limited
        if len(takeaways) < 3:
            takeaways.append({
                #set icon, title and description
                'icon': 'fa-bullseye',
                'title': "Engagement optimization opportunity",
                #make description optimistic but honest to encourage action
                'description': "Limited trend data available; consider experimenting with posting strategies across platforms to uncover performance gains"
            })
        
        return takeaways
#TODO: Shift to cloud based    
#primary paths for Windows
data_dir = Path(r"C:\Users\Arthu\Documents\Capstone\1_ETL\3. Processed Data")
model_dir = Path(r"C:\Users\Arthu\Documents\Capstone\3_Modeling\Results")

#primary path when on Mac
alt_data_dir = Path(r"/Users/arthurosakwe/Downloads/Capstone/1_ETL/3. Processed Data")
alt_model_dir = Path(r"/Users/arthurosakwe/Downloads/Capstone/3_Modeling/Results")

if not data_dir.exists():
    data_dir = alt_data_dir

if not model_dir.exists():
    model_dir = alt_model_dir
   
#Initialize the DataLoader
data_loader = DataLoader(data_dir, model_dir)

#get list of clients from daily metrics file
clients = data_loader.get_clients()

    
#Define app layout
app.layout = html.Div([
#Header
    html.Div([
        html.H1("PMM Social Media Analytics Dashboard", className="app-header"),
        html.Div([
            html.I(className="fas fa-chart-line me-2"),
            html.Span("Insights & Forecasts")
        ], className="header-subtitle")
    ], className="header-container"),
    
    html.Div([
    html.Button("Download Report as PDF", id="download-pdf-button", className="download-btn"),
    dcc.Download(id="pdf-download")
    ], style={"textAlign": "right", "marginBottom": "20px"}),

    #order
    #Client filter
    html.Div([
        html.Label("Select Client:"),
        dcc.Dropdown(
            id='client-dropdown',
            options=[{'label': client, 'value': client} for client in clients],
            value=clients[0] if clients else None,
            clearable=False,
            className="dropdown"
        )
    ], className="filter-container"),
      
    #platform filter
    html.Div([
        html.Label("Select Platform:"),
        dcc.Dropdown(
            id='platform-dropdown',
            options=[],
            value=None,
            clearable=False,
            className="dropdown")], className="filter-container"),
    
    #Key Metrics Section
    html.Div([
        html.H2([html.I(className="fas fa-chart-bar me-2"), "Key Metrics Overview"]),
        html.Div(id="key-metrics-cards", className="cards-container")
    ], className="section"),


    #Smart Summary (Hero Section)
    html.Div([
        html.H2([html.I(className="fas fa-brain me-2"), "Smart Summary"]),
        html.Div(id="smart-summary-cards", className="cards-container-smart")
    ], className="section"),

    
    #Tabs for other sections
    dcc.Tabs(id="dashboard-tabs", value="historical", className="tabs", children=[
        #1. Historical Insights
        dcc.Tab(label="Historical Insights", value="historical", className="tab", selected_className="tab-selected", children=[
            html.Div([html.Div(id="historical-performance-content", className="tab-content")])]),
        
        # 2. Platform Performance Tab
        dcc.Tab(label="Forecast", value="performance", className="tab", selected_className="tab-selected", children=[
            html.Div([
                html.Div(id="platform-performance-content", className="tab-content")])]),
        # 3. Cross-Platform Opportunity Tab
        dcc.Tab(label="Cross-Platform Opportunity", value="cross-platform", className="tab", selected_className="tab-selected", children=[
            html.Div(id="cross-platform-content", className="tab-content")]),
        
        # 4. Best Time to Post Tab
        dcc.Tab(label="Best Day to Post", value="timing", className="tab", selected_className="tab-selected", children=[
            html.Div(id="timing-content", className="tab-content")])])])

#Callbacks
@app.callback(
    Output('platform-dropdown', 'options'),
    Output('platform-dropdown', 'value'),
    Input('client-dropdown', 'value')
)
def update_platform_options(client):
    if not client:
        return [], None
    platforms = data_loader.get_client_platforms(client)
    options = [{'label': platform, 'value': platform} for platform in platforms]
    return options, platforms[0] if platforms else None

@app.callback(
    Output('smart-summary-cards', 'children'),
    Input('client-dropdown', 'value')
)
def update_smart_summary(client):
    if not client:
        return html.Div("Please select a client")
    
    takeaways = data_loader.extract_key_takeaways(client)
    
    cards = []
    for takeaway in takeaways:
        cards.append(html.Div([
            html.Div([
                html.I(className=f"fas {takeaway['icon']}")
            ], className="card-icon"),
            html.Div([
                html.H3(takeaway['title']),
                html.P(takeaway['description'])
            ], className="card-content")
        ], className="insight-card"))
    
    return cards

@app.callback(
    Output('cross-platform-content', 'children'),
    Input('client-dropdown', 'value')
)


def update_cross_platform_content(client):
    if not client:
        return html.Div("Please select a client")
    
    client_key = client.replace(" ", "_").lower()
    opportunities = data_loader.cross_platform_opportunities.get(client_key)
    
    if opportunities is None or opportunities.empty:
        return html.Div("No cross-platform data available for this client yet!")
    
    #Create cross platform table
    table_header = [
        html.Thead(html.Tr([
            html.Th("Target Platform"),
            html.Th("Best Predictors"),
            html.Th("R² Score"),
            html.Th("Recommended Action")
        ]))
    ]
    
    rows = []
    for _, opp in opportunities.iterrows():
        #set color based on R^2
        if opp['r2'] > 0.7:
            color = "#2ecc71"  #strong - green
        elif opp['r2'] > 0.5:
            color = "#f39c12"  #moderate - orange
        else:
            color = "#7f8c8d"  #weak - gray
      
        rows.append(html.Tr([
            html.Td(opp['target']),
            html.Td(opp['predictors']),
            html.Td(f"{opp['r2']:.2f}", style={"color": color, "font-weight": "bold"}),
            html.Td(opp['action'])
        ]))
    
    table_body = [html.Tbody(rows)]
    
    return html.Div([
        html.Div([
            html.I(className="fas fa-link me-2"),
            html.Span("Cross-Platform Opportunities")
        ], className="section-title"),
        
        html.Table(table_header + table_body, className="data-table"),
        
        html.Div([
            html.P([
                "This table shows which platforms predict engagement on others. ",
                html.Strong("Higher R² scores"), 
                " indicate stronger predictive relationships that can be leveraged for cross-promotion."
            ], className="help-text")
        ], className="help-container")
    ])


@app.callback(
    Output('platform-performance-content', 'children'),
    Input('client-dropdown', 'value'),
    Input('platform-dropdown', 'value')
)

def update_platform_performance(client, platform):
    if not client or not platform:
        return html.Div("Please select a client and platform")
    
    forecast_data = data_loader.get_forecast(client, platform)
    if forecast_data is None:
        return html.Div("No forecast data available for this platform yet!")

    historical_data = data_loader.get_historical_data(client, platform)
    
    #create figure
    fig = go.Figure()
    
    #add the historical data if available
    if historical_data is not None:
        fig.add_trace(go.Scatter(
            x=historical_data['full_date'],
            y=historical_data['value'],
            name='Historical Engagement',
            line=dict(color='#3498db', width=3)
        ))
    
    #detect forecast column
    forecast_y_col = None
    if 'weekly_forecast' in forecast_data.columns:
        forecast_y_col = 'weekly_forecast'
        forecast_x_col = 'week_start_date'
    elif 'forecast' in forecast_data.columns:
        forecast_y_col = 'forecast'
        forecast_x_col = 'date'
    else:
        return html.Div("Forecast data missing expected columns.")
    
    #add forecast
    fig.add_trace(go.Scatter(
        x=forecast_data[forecast_x_col],
        y=forecast_data[forecast_y_col],
        name='Forecast',
        line=dict(color='#e74c3c', width=3)))
    
    #calculate trend
    if len(forecast_data) > 0:
        forecast_start = forecast_data[forecast_y_col].iloc[0]
        forecast_end = forecast_data[forecast_y_col].iloc[-1]
        change_pct = (forecast_end - forecast_start) / (forecast_start + 1e-10) * 100
        
        if abs(change_pct) < 0.5:
            trend = "Stable"
            color = "#7f8c8d"
        elif change_pct > 0:
            trend = "Increasing"
            color = "#2ecc71"
        else:
            trend = "Decreasing"
            color = "#e74c3c"
            
        trend_text = f"{trend} Trend: {abs(change_pct):.1f}%"
    else:
        trend_text = "Trend data unavailable"
        color = "#7f8c8d"
    
    #get the R^2 
    r2_value = None
    client_key = client.replace(" ", "_").lower()
        
    #look for r^2 in  cross-platform data 
    if client_key in data_loader.cross_platform_opportunities:
        opp_data = data_loader.cross_platform_opportunities[client_key]
        if not opp_data.empty:
            matching_rows = opp_data[opp_data['target'] == platform]
            if not matching_rows.empty:
                r2_value = matching_rows['r2'].iloc[0]
    
    #set r2 display color based on value
    if r2_value is not None:
        if r2_value > 0.7:
            r2_color = "#2ecc71"  # Strong - green
        elif r2_value > 0.5:
            r2_color = "#f39c12"  # Moderate - orange
        else:
            r2_color = "#7f8c8d"  # Weak - gray
        
        r2_text = f"R² Score: {r2_value:.2f}"
    else:
        r2_text = "R² Score: N/A"
        r2_color = "#7f8c8d"  # Gray for unavailable
    
    # define layout
    fig.update_layout(
        title={
            'text': f"{platform} Engagement",
            'x': 0.5,  # This centers the title
            'xanchor': 'center',
            'yanchor': 'top'
        },
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        template="plotly_white"
    )
    return html.Div([
        html.Div([
            html.Div([
                html.H3("Trend"),
                html.Div(trend_text, className="metric-value", style={"color": color})
            ], className="metric-card"),
            html.Div([
                html.H3("Forecast Period"),
                html.Div("Next 30 Days", className="metric-value")
            ], className="metric-card"),
            # Add R² card here
            html.Div([
                html.H3("Model Accuracy"),
                html.Div(r2_text, className="metric-value", style={"color": r2_color})
            ], className="metric-card") 
        ], className="metrics-row"),
        
        dcc.Graph(figure=fig, config={'displayModeBar': False}),
        
        html.Div([
            html.P([
                f"The model shows a {abs(change_pct):.1f}% {trend.lower()} trend for {platform} engagement over the next 30 days. ",
                f"Model accuracy: {r2_text}" if r2_value is not None else ""
            ], className="insight-text")
        ], className="chart-insight")
    ])

@app.callback(
    Output('key-metrics-cards', 'children'),
    Input('client-dropdown', 'value'),
    Input('platform-dropdown', 'value')
)
def update_key_metrics(client, platform):
    if not client or not platform:
        return html.Div("Please select a client and platform")

    if data_loader.daily_metrics is None:
        return html.Div("No key metrics available for this client yet")

    df = data_loader.daily_metrics.copy()
    df['month'] = df['full_date'].dt.to_period('M')

    # Filter by selected client and platform
    filtered = df[(df['Client_Name'] == client) & (df['platform_name'] == platform)]

    if filtered.empty:
        return html.Div("No data available for this client and platform.")

    # Identify months
    latest_month = filtered['month'].max()
    previous_month = latest_month - 1

    current = filtered[filtered['month'] == latest_month]
    previous = filtered[filtered['month'] == previous_month]


    def calc_sum(df, keywords, exclude=[]):
        return df[
            (df['metric_category'].str.contains('|'.join(keywords), case=False, na=False) |
             df['metric_name'].str.contains('|'.join(keywords), case=False, na=False)) &
            (~df['metric_name'].isin(exclude))
        ]['value'].sum()

    def safe_pct_change(curr, prev):
        if prev > 0:
            return ((curr - prev) / prev) * 100
        return 0 if curr == 0 else 100

    # Compute metrics
    metrics = {
        'total_reach': {
            'curr': calc_sum(current, ['reach'], exclude=['Frequency']),
            'prev': calc_sum(previous, ['reach'], exclude=['Frequency'])
        },
        'total_engagement': {
            'curr': calc_sum(current, ['engage', 'like', 'comment', 'share'], exclude=['Engagement Rate', 'Engagement Rate (%)']),
            'prev': calc_sum(previous, ['engage', 'like', 'comment', 'share'], exclude=['Engagement Rate', 'Engagement Rate (%)'])
        },
        'total_impressions': {
            'curr': calc_sum(current, ['impression', 'view']),
            'prev': calc_sum(previous, ['impression', 'view'])
        },
        'total_clicks': {
            'curr': calc_sum(current, ['click', 'link']),
            'prev': calc_sum(previous, ['click', 'link'])
        },
        'total_posts': {
            'curr': current['inferred_post_flag'].sum() if 'inferred_post_flag' in current else 0,
            'prev': previous['inferred_post_flag'].sum() if 'inferred_post_flag' in previous else 0
        }
    }

    percent_changes = {k: safe_pct_change(v['curr'], v['prev']) for k, v in metrics.items()}

    # Card display
    def create_metric_card(title, value, metric_key):
        percent_change = percent_changes[metric_key]
        return html.Div([
            html.H3(title, style={"fontSize": "1rem", "color": "#7f8c8d", "fontWeight": "normal", "marginBottom": "15px", "textAlign": "center"}),
            html.Div(f"{int(value):,}", style={"fontSize": "2.5rem", "fontWeight": "bold", "textAlign": "center", "marginBottom": "10px"}),
            html.Div([
                html.Div([
                    html.Span("↑ " if percent_change > 0 else "↓ " if percent_change < 0 else "", 
                              style={"color": "#4caf50" if percent_change > 0 else "#f44336" if percent_change < 0 else "#7f8c8d"}),
                    html.Span(f"{abs(percent_change):.1f}%", 
                              style={"color": "#4caf50" if percent_change > 0 else "#f44336" if percent_change < 0 else "#7f8c8d"}),
                    html.Span(" FROM LAST PERIOD", style={"fontSize": "0.7rem", "color": "#95a5a6", "marginLeft": "3px"})
                ], style={
                    "display": "flex", "alignItems": "center", "justifyContent": "center",
                    "backgroundColor": "#f1f8e9" if percent_change > 0 else "#ffebee" if percent_change < 0 else "#f5f5f5",
                    "padding": "5px 10px", "borderRadius": "4px", "fontSize": "0.9rem", "fontWeight": "bold"
                })
            ], style={"display": "flex", "justifyContent": "center"})
        ], style={
            "backgroundColor": "#fff", "borderRadius": "8px",
            "boxShadow": "0 2px 10px rgba(0,0,0,0.1)", "padding": "20px"
        })

    # Titles
    metric_titles = {
        'total_reach': 'Total Reach',
        'total_engagement': 'Total Engagement',
        'total_impressions': 'Total Impressions',
        'total_clicks': 'Total Ad Links Clicked',
        'total_posts': 'Total Posts'
    }

    cards = []
    for metric, title in metric_titles.items():
        cards.append(create_metric_card(title, metrics[metric]['curr'], metric))

    return html.Div(
        cards,
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
            "gap": "20px",
            "justifyContent": "center",
            "alignItems": "start",
            "padding": "10px"
        }
    )


@app.callback(
    Output('historical-performance-content', 'children'),
    Input('client-dropdown', 'value'),
    Input('platform-dropdown', 'value'))

def update_historical_performance(client, platform):
    if not client or not platform:
        return html.Div("Please select a client and platform")
    
    historical_data = data_loader.get_historical_data(client, platform)
    
    if historical_data is None:
        return html.Div("No historical data available yet!")
  
    #extract month and year from the dataset
    historical_data['year'] = historical_data['full_date'].dt.year
    historical_data['month'] = historical_data['full_date'].dt.month
    historical_data['day'] = historical_data['full_date'].dt.day
    
    #find most recent month in the data
    most_recent_year = historical_data['year'].max()
    most_recent_month = historical_data.loc[historical_data['year'] == most_recent_year, 'month'].max()
    
    #get previous month
    if most_recent_month == 1:
        previous_month = 12
        previous_year = most_recent_year - 1
    else:
        previous_month = most_recent_month - 1
        previous_year = most_recent_year
    
    #filter data for current and previous months - future iteration will do quarter-over-quarter reviews
    current_month_data = historical_data[
        (historical_data['year'] == most_recent_year) & 
        (historical_data['month'] == most_recent_month)
    ].copy()
    
    previous_month_data = historical_data[
        (historical_data['year'] == previous_year) & 
        (historical_data['month'] == previous_month)
    ].copy()
    
    #check if we have enough data
    if current_month_data.empty:
        return html.Div("No data available for the current month.")
    
    #create figure
    fig = go.Figure()
    
    #add line for all historical data for context
    fig.add_trace(go.Scatter(
        x=historical_data['full_date'],
        y=historical_data['value'],
        mode='lines',
        line=dict(color='#d1d1d1', width=1),
        name="All Historical Data"
    ))
    
    #add current month data
    fig.add_trace(go.Scatter(
        x=current_month_data['full_date'],
        y=current_month_data['value'],
        mode='lines+markers',
        line=dict(color='#3498db', width=3),
        name=f"{datetime(most_recent_year, most_recent_month, 1).strftime('%B %Y')}"
    ))
    
    #Add previous month data with day alignment
    if not previous_month_data.empty:
        #create a version of previous month data aligned by day of month for comparison
        aligned_prev_month = previous_month_data.copy()
        
        #create date series that uses current monts year/month
        # but previous month days, so they stack visually by day of month
        current_month_data = current_month_data.sort_values('day')
        aligned_prev_month = aligned_prev_month.sort_values('day')
        
        #Limit to same number of days for fair comparison
        min_days = min(len(current_month_data), len(aligned_prev_month))
        current_month_data = current_month_data.head(min_days)
        aligned_prev_month = aligned_prev_month.head(min_days)
        
        #ad trace with aligned dates (but show actual dates in hover)
        fig.add_trace(go.Scatter(
            x=current_month_data['full_date'],
            y=aligned_prev_month['value'],
            mode='lines+markers',
            line=dict(color='#e74c3c', width=3, dash='dash'),
            name=f"{datetime(previous_year, previous_month, 1).strftime('%B %Y')}",
            hovertemplate='Day %{customdata}<br>Value: %{y}<extra></extra>',
            customdata=aligned_prev_month['day']))
        
        #calculate the month-over-month change- future dev to look at previous quarter.
        current_total = current_month_data['value'].sum()
        previous_total = aligned_prev_month['value'].sum()
        
        if previous_total > 0:
            percent_change = ((current_total - previous_total) / previous_total)*100
            change_text = f"Change: {percent_change:.1f}% from previous month"
            change_color = "#4caf50" if percent_change > 0 else "#f44336"
        else:
            change_text = "No previous month data for comparison"
            change_color = "#7f8c8d"
        if current_total == previous_total:
            change_text = "No change from previous month"
            change_color = "#7f8c8d"
    else:
        change_text = "No previous month data for comparison"
        change_color = "#7f8c8d"
    
    # Update layout#d
    #set x-axis to focus only on the current month - future dev to look at previous quarter.
    start_of_month = datetime(most_recent_year, most_recent_month, 1)
    if most_recent_month == 12:
        start_of_next_month = datetime(most_recent_year + 1, 1, 1)
    else:
        start_of_next_month = datetime(most_recent_year, most_recent_month + 1, 1)
    
    fig.update_layout(
        xaxis=dict(
            range=[start_of_month, start_of_next_month - timedelta(days=1)],
            tickformat="%b %d",
            tickmode="auto",
        ))
    
    #Summary card for the comparison
    comparison_card = html.Div([
        html.H3("Month-over-Month Comparison"),
        html.Div([
            html.Div(change_text, style={"color": change_color, "fontWeight": "bold"})
        ], className="comparison-stats")
    ], className="metric-card")

    #return completed visualization with comparison card
    return html.Div([
        comparison_card,
        dcc.Graph(figure=fig) ])

#best day to post callback - further dev when datetime added
@app.callback(
    Output('timing-content', 'children'),
    Input('client-dropdown', 'value')
)
def update_best_day_to_post(client):
    if not client:
        return html.Div("Please select a client")

    seasonal_data = data_loader.get_seasonal_patterns(client)
    
    if seasonal_data is None or seasonal_data.empty:
        return html.Div("No seasonal pattern data available yet")
    
    #plot engagement by day of week
    fig = px.bar(
        seasonal_data,
        x='day',
        y='value',
        title=f'Best Day to Post for {client}',
        color='value',
        #add custom for PMM
        color_continuous_scale=[
        [0, '#d5e8d4'],  # Light green
        [0.5, '#a8d5ba'],  # Soft green
        [0.8, '#6ab187'],   # Medium green
        [1, '#5c7359']   # Dark green
    ]
    )
    
    fig.update_layout(
        yaxis_title="Predicted Engagement Score",
        xaxis_title="Day of the Week",
        title_x=0.5,
        template="plotly_white",
        height=400
    )
    
    return dcc.Graph(figure=fig)

#app call to download pdf
@app.callback(
    Output('pdf-download', 'data'),
    Input('download-pdf-button', 'n_clicks'),
    State('client-dropdown', 'value'),
    State('platform-dropdown', 'value'),
    prevent_initial_call=True
)
def generate_pdf(n_clicks, selected_client, selected_platform):
    if not selected_client or not selected_platform:
        return None

    df = data_loader.daily_metrics.copy()
    df['month'] = df['full_date'].dt.to_period('M')

    filtered = df[(df['Client_Name'] == selected_client) & (df['platform_name'] == selected_platform)]
    if filtered.empty:
        metrics_html = "<p>No data available for the selected platform.</p>"
    else:
        latest_month = filtered['month'].max()
        current = filtered[filtered['month'] == latest_month]


        def calc_sum(df, keywords, exclude=[]):
            return df[
                (df['metric_category'].str.contains('|'.join(keywords), case=False, na=False) |
                 df['metric_name'].str.contains('|'.join(keywords), case=False, na=False)) &
                (~df['metric_name'].isin(exclude))
            ]['value'].sum()

        total_reach = calc_sum(current, ['reach'], exclude=['Frequency'])
        total_engagement = calc_sum(current, ['engage', 'like', 'comment', 'share'], exclude=['Engagement Rate'])
        total_impressions = calc_sum(current, ['impression', 'view'])
        total_clicks = calc_sum(current, ['click', 'link'])
        total_posts = current['inferred_post_flag'].sum() if 'inferred_post_flag' in current else 0

        metrics_html = f"""
        <h2>Key Metrics Overview ({selected_platform})</h2>
        <ul>
            <li><strong>Total Reach:</strong> {total_reach:,.0f}</li>
            <li><strong>Total Engagement:</strong> {total_engagement:,.0f}</li>
            <li><strong>Total Impressions:</strong> {total_impressions:,.0f}</li>
            <li><strong>Total Ad Links Clicked:</strong> {total_clicks:,.0f}</li>
            <li><strong>Total Posts:</strong> {total_posts:,.0f}</li>
        </ul>"""

    # Smart Summary
    smart_summary = data_loader.extract_key_takeaways(selected_client)
    smart_html = "<h2>Smart Summary</h2><ul>"
    for takeaway in smart_summary:
        smart_html += f"<li><strong>{takeaway['title']}:</strong> {takeaway['description']}</li>"
    smart_html += "</ul>"

    # Placeholder for Historical Insights
    historical_html = "<h2>Historical Insights</h2><p>Historical engagement trends are visualized inside this dashboard. (Graph snapshots are not exported yet.)</p>"

    # Merge HTML
    full_html = f"""
    <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; font-size: 12px; }}
                h1 {{ text-align: center; }}
                h2 {{ margin-top: 20px; }}
                ul {{ list-style-type: disc; margin-left: 20px; }}
            </style>
        </head>
        <body>
            <h1>Social Media Report for {selected_client}</h1>
            {metrics_html}
            {smart_html}
            {historical_html}
        </body>
    </html>
    """

    # Generate PDF
    pdf_buffer = io.BytesIO()
    pisa_status = pisa.CreatePDF(io.StringIO(full_html), dest=pdf_buffer)
    if pisa_status.err:
        return None

    pdf_buffer.seek(0)
    filename = f"{selected_client}_{selected_platform}_Social_Media_Report.pdf"
    return dcc.send_bytes(pdf_buffer.getvalue(), filename=filename)



    #CSS styles section
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        
        <!-- Base Layout & Variables -->
        <style>
             /* Global styles */
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #ffffff; /* white */
                color: #4f6865; /* green text */
                    }
            
            /* Common  backgrounds */
            .section, .tab, .filter-container, .tab-content, .tab-filter, 
            .insight-card, .metric-card {
                background-color: #ffffff;
            }
            
            .app-container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                }
            
             /* Header */
            .header-container {
                background-color: #4f6865; /* green */
                color: #f7d9d0; /* PMM pink - almost bubble gum pink? */
                padding: 20px;
                border-radius: 8px;
                
                margin-bottom: 20px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
             }
            
            .app-header {
                margin: 0;
                font-size: 2rem;
                font-weight: 600;
                }
            
            .header-subtitle {
                margin-top: 8px;
                font-size: 1.1rem;
                opacity: 0.9;
            }
            
            /* Footer */
            .footer {
                display: flex;
                justify-content: space-between;
                padding: 15px 0;
                margin-top: 20px;
                border-top: 1px solid #eee;
                color: #7f8c8d;
            }
            
            .footer-text {
                margin: 0;
                font-size: 0.9rem;
            }
         </style>
        
          <!--UI ELEMENTS -->
         <style>
            /* Sections */
            .section {
                padding: 20px;
                border-radius: 8px;
                 border: 1px solid #e0e6e4;
                margin-bottom: 30px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
             }
            
            .section h2 {
                margin-top: 0;
                margin-bottom: 20px;
                color: #4f6865;
                font-size: 1.5rem;
                font-weight: 600;
                display: flex;
                align-items: center;
            }
            
            .section-title {
                font-size: 1.3rem;
                font-weight: 600;
                margin-bottom: 15px;
                color: #4f6865;
                display: flex;
                align-items: center;
            }
            
            /* Filter Elements */
            .filter-container {
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 20px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            }
            
            .dropdown {
                margin-top: 5px;
                }
            
            
            
            /* Label styling */
            .filter-container label {
                font-size: 1rem;
                font-weight: 600;
                color: #4f6865;
                margin-bottom: 8px;
                display: block;
            }
            
            /* Dropdown styling */
            .filter-container .Select-control, 
            .filter-container select {
                width: 100%;
                padding: 7px 10px;
                font-size: 1rem;
                border: 1px solid #4f6865;
                border-radius: 6px;
                color: #4f6865;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
                transition: border-color 0.2s, box-shadow 0.2s;
            }
            
            
            /* Cards & Containers */
            .cards-container {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 20px;
                justify-content: center;
                align-items: start;
                padding: 10px;
            }
            .insight-card {
                display: flex;
                border-radius: 8px;
                padding: 15px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
                border-left: 4px solid #4f6865;
                transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s;
                color: #4f6865; 
            }
            .card-icon {
                font-size: 2rem;
                margin-right: 15px;
                color: #4f6865;
                display: flex;
                align-items: center;
                justify-content: center;
                min-width: 50px;
            }
            
            .card-content h3 {
                margin: 0 0 8px 0;
                font-size: 1.1rem;
                color: #4f6865;
                font-weight: 600;
            }
            
            .card-content p {
                margin: 0;
                color: #7f8c8d; /* light grayish */
                font-size: 0.9rem;
            }
            
            /* Metrics */
            .metrics-row {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 15px;
                margin-bottom: 20px;
            }
            
            .metric-card {
                border-radius: 8px;
                padding: 15px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
                text-align: center;
            }
            
            .metric-card h3 {
                margin: 0 0 10px 0;
                font-size: 1rem;
                color: #4f6865;
                font-weight: 600;
              }
            
             .metric-value {
                font-size: 1.2rem;
                font-weight: 600;
            }
            
              /* Buttons */
            .download-btn {
                background-color: #f7d9d0;
                color: #4f6865;
                padding: 10px 15px;
                border-radius: 4px;
                cursor: pointer;
                margin-bottom: 15px;
                display: flex;
                align-items: center;
                font-weight: 600;
            }
        </style>
        
        <!-- Interactive cards etc -->
        <style>
            /* Tabs */
            .tabs {
                display: flex;
                justify-content: center;
                gap: 40px; /* adds space between tabs */
                border-bottom: 2px solid #4f6865;
                margin-bottom: 20px;
            }
            
            .tab {
                padding: 10px 20px;
                color: #4f6865;
                border-right: 1px solid #eee;
                cursor: pointer;
                font-size: 1rem;
                transition: background-color 0.3s, color 0.3s;
            }
            
            .tab:last-child {
                border-right: none;
            }
            
            .tab-filter {
                padding: 15px;
                border-radius: 8px 8px 0 0;
                border-bottom: 1px solid #eee;
            }
            
            .tab-content {
                padding: 20px;
                border-radius: 0 0 8px 8px;
                min-height: 400px;
            }
            
            /* Hover States */
            .download-btn:hover {
                background-color: #e6c5bc;
            }
            
            .tab:hover {
                background-color: #f7d9d0;
            }
            
            .insight-card:hover {
                transform: translateY(-4px);
                border-left: 4px solid #f7d9d0;
                box-shadow: 0 6px 12px rgba(0, 0, 0, 0.08);
            }
            
            .insight-card:hover .card-icon {
                color: #f7d9d0; 
            }
            
            .filter-container .Select-control:hover, 
            .filter-container select:hover,
            .filter-container .Select-control:focus, 
            .filter-container select:focus {
                border-color: #f7d9d0;
                box-shadow: 0 0 0 3px rgba(247, 217, 208, 0.4); /* pink glow thingy */
                outline: none;
            }
            
            .tab-selected {
                background-color: #f7d9d0;
                color: #4f6865;
                font-weight: 600;
                border-top: 3px solid #4f6865;
                border-bottom: none;
                outline: none;
            }
            
            .tab:focus {
                border-color: transparent;
            }
            
            .tab--selected {
                border-bottom: none !important;
                box-shadow: none !important;
                outline: none !important;
            }
        </style>
        
        <style>
            @media (max-width: 768px) {
                .cards-container {
                    grid-template-columns: 1fr;
                }
                
                .cards-container.centered-cards {
                    justify-content: center;
                    display: flex;
                    flex-wrap: wrap;
                    gap: 15px;
                    max-width: 1000px;
                    margin: 0 auto;
                }
            
                .metrics-row {
                    grid-template-columns: 1fr;
                }
            }
        </style>

    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''
#look for port
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

if __name__ == '__main__':    
    port = find_free_port()
    host = "127.0.0.1"  
    dashboard_url = f"http://{host}:{port}"
    print("Dashboard is running at:", dashboard_url)
    app.run(debug=True, port=port)