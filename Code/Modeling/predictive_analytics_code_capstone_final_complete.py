"""
Capstone Project
Data Modeling
Author: Arthur Osakwe
Version: 4.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import logging
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV


#Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("predictive_modeling_log.log"),
        logging.StreamHandler()])
logger = logging.getLogger("predictive_modeling")

#Add class for the predictive modeling functions
class SocialMediaPredictiveModeling:    
    
    #initialize the pipeline
    def __init__(self, data_dir, output_dir):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.minimum_r2_threshold = 0.4 #TODO: review this

        
        #create output directories 
        self.create_directories()
        
        #initialize containers
        self.daily_metrics = None
        self.platform_comparison = None
 
        self.time_series_models = {}
        self.regression_models = {}
        self.feature_importance = {}
        self.model_diagnostics = []
        self.correlations = {}
        self.cross_platform_opportunities = {}
        
        
        self.load_data()
        
    #createouptut directories
    def create_directories(self):
        os.makedirs(self.output_dir, exist_ok=True)

        subdirs = ['models', 'plots', 'forecasts']
        for subdir in subdirs:
            os.makedirs(self.output_dir / subdir, exist_ok=True)
    #load views from csv sources
    def load_data(self):
        file_map = {'daily_metrics': 'vw_daily_metrics.csv',
            'platform_comparison': 'vw_platform_comparison.csv',
            'mailchimp_campaigns': 'vw_mailchimp_campaigns.csv'}

        for attr, filename in file_map.items():
            file_path = self.data_dir / filename
            if file_path.exists():
                setattr(self, attr, pd.read_csv(file_path))
                logger.info(f"Loaded source file {filename} into {attr}")
            else:
                setattr(self, attr, None)
                logger.error(f"Missing file: {filename} in {self.data_dir}")

        ##TO DO: datetime unavailable atm need to revisit ETL code to add date only column and datetime column - submit as is for now for capstone and revise later due to time constraints 
        if self.daily_metrics is not None and 'full_date' in self.daily_metrics.columns:
            self.daily_metrics['full_date'] = pd.to_datetime(self.daily_metrics['full_date'])

        return True
    
    #Build pivot table to transform data to ML ready format
    def create_platform_pivot_table(self, engagement_data):
        #check for sufficient data
        if engagement_data.empty:
            logger.warning("No engagement data available for pivot table creation")
            return pd.DataFrame()
            
        #pivot table
        pivot_data = engagement_data.pivot_table(index='full_date',
            columns=['platform_name', 'metric_name'],
            values='value',
            aggfunc='sum')
        
        #log any sparse columns(Nan mostly)
        if not pivot_data.empty:
            sparsity = pivot_data.isna().sum() / len(pivot_data)
            sparse_cols = sparsity[sparsity > 0.5].index.tolist()
            
            if sparse_cols:
                logger.warning(f"Sparse data detected in pivot table. Columns with >50% missing: {len(sparse_cols)}")
                
        #flatten indexes
        pivot_data.columns = [f"{platform}_{metric}" for platform, metric in pivot_data.columns]
        
        #reset index - get date as the col
        result = pivot_data.reset_index()
        
        #handle missing data fill with zeros if relevant
        zero_metrics = ['post_count', 'comment_count', 'like_count', 'share_count', 'click_count']
        
        for col in result.columns:
            if any(zm in col.lower() for zm in zero_metrics):
                # Fill counts with 0
                result[col] = result[col].fillna(0)
        
        return result
    
    #Generate lag features, calendar attributes, rolling stats, and zero indicators.
    def create_features_for_weekly_model(self, weekly_series):
        df = weekly_series.copy()
        
        #add key calendar features for seasonality
        df['week_of_year'] = df.index.isocalendar().week
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        
        #add indicator for zero values in previous weeks
        #determine if  prev inactivity predict future inactivity
        df['prev_week_zero'] = (df['value'].shift(1).fillna(0) == 0).astype(int)
        
        #add lag features to learn from past avlues
        for lag in [1, 2, 3, 4, 8, 12]:  # Weekly lags (previous weeks, months, quarter)
            if len(df) > lag:
                df[f'lag_{lag}'] = df['value'].shift(lag)
        
        #add rolling stats to smooth short term volatility 
        #detect low activity periods
        if len(df) > 4:
            df['rolling_mean_4'] = df['value'].rolling(window=4, min_periods=1).mean()
            df['rolling_std_4'] = df['value'].rolling(window=4, min_periods=1).std().fillna(0)
            df['zero_ratio_4'] = df['value'].rolling(window=4, min_periods=1).apply(lambda x: (x == 0).mean(), raw=True).fillna(1)
        
        #detect longer term patterns
        if len(df) > 12:
            df['rolling_mean_12'] = df['value'].rolling(window=12, min_periods=1).mean()
            df['rolling_std_12'] = df['value'].rolling(window=12, min_periods=1).std().fillna(0)
        
        #check for seasonality (same week last year)
        if len(df) > 52:
            df['same_week_last_year'] = df['value'].shift(52)
        
        #fill missing values
        df = df.fillna(0)
        
        return df
     
    #update time series features during forcasting
    def update_weekly_forecast_features(self, features, new_value, prediction_history, step_idx, next_date):
        features = features.copy()
    
        #update time-based features
        features['week_of_year'] = next_date.isocalendar()[1]  # Week of year
        features['month'] = next_date.month
        features['quarter'] = (next_date.month - 1) // 3 + 1
    
        #update zero indicator
        features['prev_week_zero'] = int(new_value == 0)
    
        #update lag features
        features['lag_1'] = new_value
    
        if step_idx >= 1 and 'lag_2' in features:
            features['lag_2'] = prediction_history[step_idx - 1]
        if step_idx >= 2 and 'lag_3' in features:
            features['lag_3'] = prediction_history[step_idx - 2]
        if step_idx >= 3 and 'lag_4' in features:
            features['lag_4'] = prediction_history[step_idx - 3]
        if step_idx >= 7 and 'lag_8' in features:
            features['lag_8'] = prediction_history[step_idx - 7]
        if step_idx >= 11 and 'lag_12' in features:
            features['lag_12'] = prediction_history[step_idx - 11]
    
        #update rolling statistics if we have enough predictions
        if step_idx >= 3 and 'rolling_mean_4' in features and 'rolling_std_4' in features:
            recent_values = prediction_history[max(0, step_idx - 3):step_idx + 1]
            features['rolling_mean_4'] = sum(recent_values) / len(recent_values)
            if len(recent_values) > 1:
                features['rolling_std_4'] = np.std(recent_values)
            else:
                features['rolling_std_4'] = 0
            #update zero ratio
            features['zero_ratio_4'] = sum(v == 0 for v in recent_values) / len(recent_values)
    
        if step_idx >= 11 and 'rolling_mean_12' in features and 'rolling_std_12' in features:
            recent_values = prediction_history[max(0, step_idx - 11):step_idx + 1]
            features['rolling_mean_12'] = sum(recent_values) / len(recent_values)
            features['rolling_std_12'] = np.std(recent_values) if len(recent_values) > 1 else 0
    
        return features
    
    #detect viral engagement spikes
    def add_spike_flag(self, time_series, window=7, threshold=2.0):
        df = time_series.to_frame(name='value')
        
        #rolling mean and rolling std
        rolling_mean = df['value'].rolling(window=window, min_periods=1).mean()
        
        rolling_std = df['value'].rolling(window=window, min_periods=1).std().fillna(0)
        
        #define spike condition
        spike_condition = df['value'] > (rolling_mean + threshold * rolling_std)
        #add spike_flag
        df['spike_flag'] = spike_condition.astype(int)
        
        return df
    
    #cross platform analysis
    def analyze_cross_platform_effects(self, client_name):
        if self.daily_metrics is None or self.daily_metrics.empty:
            logger.warning("No daily metrics data available")
            return None
            
        #filter data for the client
        client_data = self.daily_metrics[self.daily_metrics['Client_Name'] == client_name].copy()
        
        #filter for engagement metrics
        engagement_data = client_data[client_data['metric_category'] == 'Engagement']
        
        if engagement_data.empty:
            logger.warning(f"No engagement data for client {client_name}")
            return None
        
        #create a pivot table with dates as index, platforms as columns
        engagement_pivot = engagement_data.pivot_table(
            index='full_date',
            columns='platform_name',
            values='value',
            aggfunc='sum'
        ).fillna(0)
        
        #calculate correlation matrix
        corr_matrix = engagement_pivot.corr()
        
        #save correlation matrix
        corr_path = self.output_dir / 'forecasts' / f'correlation_{client_name.replace(" ", "_").lower()}.csv'
        corr_matrix.to_csv(corr_path)
        
        #store in the correlations dictionary for later use
        client_key = client_name.replace(" ", "_").lower()
        if not hasattr(self, 'correlations'):
            self.correlations = {}
        self.correlations[client_key] = corr_matrix
        
        logger.info(f"Cross-platform correlation matrix calculated for {client_name}")
                   
        return corr_matrix
   
    #calculate seasonal patterns from metrics
    def calculate_seasonal_patterns(self, client_name):
        if self.daily_metrics is None or self.daily_metrics.empty:
            logger.warning(f"No daily metrics data available for {client_name}")
            return None
            
        #filter data
        client_data = self.daily_metrics[self.daily_metrics['Client_Name'] == client_name].copy()
        engagement_data = client_data[client_data['metric_category'] == 'Engagement']
        
        if engagement_data.empty:
            logger.warning(f"No engagement data for client {client_name}")
            return None
        
        #extract day of week from date(future iteration review for datetime)
        engagement_data.loc[:, 'day'] = engagement_data['full_date'].dt.day_name()
        
        #group by day only
        seasonal_patterns = engagement_data.groupby(['day'])['value'].mean().reset_index()

        #manually reorder the days
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        seasonal_patterns = seasonal_patterns.set_index('day').reindex(days_order).reset_index()

        #save result to csv
        output_path = self.output_dir / 'forecasts' / f'seasonal_patterns_{client_name.replace(" ", "_").lower()}.csv'
        seasonal_patterns.to_csv(output_path, index=False)
        
        logger.info(f"Seasonal patterns calculated and saved for {client_name}")
        
        return seasonal_patterns  
    
    def calculate_summary_metrics(self):
        if self.daily_metrics is None:
            logger.warning("No daily metrics available.")
            return None

        summary = pd.DataFrame({'Client_Name': self.daily_metrics['Client_Name'].unique()})

        #definitions: metric categories and their rules
        metric_configs = {
            'total_reach': {
                'filters': lambda df: ((df['metric_category'] == 'Reach') | (df['metric_name'].str.contains('reach', case=False, na=False))),
                'exclude': ['Frequency'] },
            'total_engagement': {
                'filters': lambda df: (
                    (df['metric_category'] == 'Engagement') |
                    df['metric_name'].str.contains('like|comment|share|interact|engage', case=False, na=False)),
                'exclude': ['Engagement Rate', 'Engagement Rate (%)'] },
            'total_impressions': {
                'filters': lambda df: (
                    (df['metric_category'] == 'Impressions') |
                    df['metric_name'].str.contains('view|impression', case=False, na=False)),
                'exclude': []},

            'total_clicks': {
                'filters': lambda df: (
                    (df['metric_category'] == 'Clicks') |
                    df['metric_name'].str.contains('click|link', case=False, na=False)),
                'exclude': []}}

        #clculate each summary metric
        for metric_name, config in metric_configs.items():
            data = self.daily_metrics[config['filters'](self.daily_metrics)]
            if config['exclude']:
                data = data[~data['metric_name'].isin(config['exclude'])]

            if not data.empty:
                metric_sum = data.groupby('Client_Name')['value'].sum()
                summary[metric_name] = summary['Client_Name'].map(metric_sum).fillna(0).astype(int)
            else:
                summary[metric_name] = 0
                logger.warning(f"No data found for {metric_name}.")
                
        #add total_posts using inferred_post_flag
        if 'inferred_post_flag' in self.daily_metrics.columns:
            inferred_posts = self.daily_metrics[self.daily_metrics['inferred_post_flag'] == 1]
            posts_by_client = inferred_posts.groupby('Client_Name')['inferred_post_flag'].count()
            summary['total_posts'] = summary['Client_Name'].map(posts_by_client).fillna(0).astype(int)  
                
        #add recent platform breakdowns if available
        if self.platform_comparison is not None:
            recent_platform_data = self.platform_comparison.sort_values(['year', 'month'], ascending=False)
            for platform in recent_platform_data['platform_name'].unique():
                platform_data = recent_platform_data[recent_platform_data['platform_name'] == platform]
                platform_pivot = platform_data.pivot_table(index='Client_Name', columns='metric_category', values='total_value', aggfunc='sum')
                platform_pivot.columns = [f"{platform}_{col}" for col in platform_pivot.columns]
                summary = summary.merge(platform_pivot.reset_index(), on='Client_Name', how='left')

        #add Mailchimp metrics if available
        if hasattr(self, 'mailchimp_campaigns') and self.mailchimp_campaigns is not None:
            recent_mailchimp = self.mailchimp_campaigns.sort_values(['year', 'month'], ascending=False)
            mailchimp_pivot = recent_mailchimp.pivot_table(index='Client_Name', columns='metric_name', values='value', aggfunc='sum')
            mailchimp_pivot.columns = [f"mailchimp_{col}" for col in mailchimp_pivot.columns]
            summary = summary.merge(mailchimp_pivot.reset_index(), on='Client_Name', how='left')

        #Fill missing with 0
        summary.fillna(0, inplace=True)
        for col in summary.columns:
            if col != 'Client_Name' and pd.api.types.is_numeric_dtype(summary[col]):
                summary[col] = summary[col].astype(int)

        #dashboard totals
        dashboard_keys = ['total_reach', 'total_engagement', 'total_impressions', 'total_clicks', 'total_posts']
        grand_totals = {k: summary[k].sum() for k in dashboard_keys}

        formatted_totals = {k: (f"{v/1_000_000:.1f}M" if v >= 1_000_000 else f"{v/1_000:.1f}K" if v >= 1000 else str(v)) for k, v in grand_totals.items()}

        #save outputs
        summary.to_csv(self.output_dir / 'forecasts' / 'summary_metrics_by_client.csv', index=False)
        pd.DataFrame([grand_totals]).to_csv(self.output_dir / 'forecasts' / 'dashboard_metrics.csv', index=False)

        self.summary_metrics = summary
        self.dashboard_metrics = formatted_totals

        logger.info("Summary metrics calculated and saved successfully.")
        return summary
    
    #Return the best performing model
    #hopefully this helps fix the low R2
    def tune_random_forest(self, X_train, y_train):

        param_grid = {'n_estimators': [100, 300],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
            'max_features': ['sqrt', 'log2']}
    
        rf = RandomForestRegressor(random_state=42)
        
        grid_search = GridSearchCV(
            rf,
            param_grid,
            cv=3, #3-fold cross-validation
            scoring='r2', #optimize for R2
            n_jobs=-1,    #use all CPU cores
            verbose=1   #show progress
        )
    
        grid_search.fit(X_train, y_train)
    
        return grid_search.best_estimator_
    
    def calculate_regression_metrics(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        return {'rmse': np.sqrt(mse),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)}
    
    #### function to predict engagement  ###
    #create classification model to predict if engagement will occur
    #then create regression model to predict by how much, when it occurs
    #update - using weekly data to smooth out variance a little
    def predict_engagement_two_part_model(self, client_name, platform, days=30):
        weeks = days // 7
        
        if self.daily_metrics is None or self.daily_metrics.empty:
            logger.warning("No daily metrics data available")
            return None
    
        #filter data 
        client_data = self.daily_metrics[
            (self.daily_metrics['Client_Name'] == client_name) &
            (self.daily_metrics['platform_name'] == platform) &
            (self.daily_metrics['metric_category'] == 'Engagement')
        ].copy()
    
        if client_data.empty:
            logger.warning(f"No engagement data for client {client_name} and platform {platform}")
            return None
    
        #group by week
        client_data['week_start'] = client_data['full_date'] - pd.to_timedelta(client_data['full_date'].dt.dayofweek, unit='D')
        weekly_engagement = client_data.groupby('week_start')['value'].sum().reset_index()
            
        #set week_start as index
        weekly_engagement.set_index('week_start', inplace=True)
        weekly_engagement = weekly_engagement.asfreq('W-MON').fillna(0)  # Ensure no missing weeks
        
        #add feature df for weekly data
        features_df = self.create_features_for_weekly_model(weekly_engagement)
        
        #class should have binary target(did engagement occor or not)
        features_df.loc[:, 'has_engagement'] = (features_df['value'] > 0).astype(int)
        
        #split data for both models
        feature_cols = [col for col in features_df.columns if col not in ['value', 'has_engagement']]
        
        #create train/test splits
        train_size = int(len(features_df) * 0.8)
        train_df = features_df.iloc[:train_size]
        test_df = features_df.iloc[train_size:]
        
        #Train the classification model to predict if engagement will occur
        X_train_cls = train_df[feature_cols]
        y_train_cls = train_df['has_engagement']
        
        if y_train_cls.nunique() == 1:
            logger.warning(f"Single-class training data for {client_name} - {platform}, engagement always {y_train_cls.iloc[0]}")
            # Use a simple predictor that always returns the single class
            class_value = y_train_cls.iloc[0]
            classifier = DummyClassifier(strategy="constant", constant=class_value)
        else:
            classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        
        classifier.fit(X_train_cls, y_train_cls)
        
        #Train regression model to predict engagement amount( on non-zero samples)
        non_zero_train = train_df[train_df['value'] > 0]
        if len(non_zero_train) < 5:  # Need minimum samples for regression (reduced from 10 since we're using weekly data)
            logger.warning(f"Insufficient non-zero weekly samples for {client_name} - {platform}: {len(non_zero_train)} points")
            return None
        
        X_train_reg = non_zero_train[feature_cols]
        y_train_reg = non_zero_train['value']
        
        #apply tuner func to fine tune model
        regressor = self.tune_random_forest(X_train_reg, y_train_reg)
        
        #save the fine-tuned regressor model
        title = f"{client_name}_{platform}_two_part_weekly_regressor"
        title = title.replace(" ", "_").lower()
        
        model_info = {
            'model': regressor,
            'feature_names': X_train_reg.columns.tolist() if hasattr(X_train_reg, 'columns') else None
        }
        
        model_path = self.output_dir / 'models' / f'{title}.pkl'
        
        with open(model_path, 'wb') as f:
            joblib.dump(model_info, f)
        
        #evaluate models if test data available
        if len(test_df) > 0:
            # Evaluate classification model
            X_test_cls = test_df[feature_cols]
            y_test_cls = test_df['has_engagement']
            cls_preds = classifier.predict(X_test_cls)
            cls_accuracy = accuracy_score(y_test_cls, cls_preds)
            
            #evaluate regression model on non-zero samples
            non_zero_test = test_df[test_df['value'] > 0]
            if len(non_zero_test) > 0:
                X_test_reg = non_zero_test[feature_cols]
                y_test_reg = non_zero_test['value']
                reg_preds = regressor.predict(X_test_reg)
                r2 = r2_score(y_test_reg, reg_preds)
                mae = mean_absolute_error(y_test_reg, reg_preds)
                
                logger.info(f"Two-part weekly model for {client_name} - {platform}: "
                            f"Classification accuracy: {cls_accuracy:.4f}, "
                            f"Regression R²: {r2:.4f}, MAE: {mae:.4f}")
            else:
                logger.info(f"Two-part weekly model for {client_name} - {platform}: "
                            f"Classification accuracy: {cls_accuracy:.4f}, "
                            f"No non-zero samples in test set for regression evaluation")
        
        #generate forecast directly for future weeks
        last_date = weekly_engagement.index[-1]
        future_dates = [last_date + pd.Timedelta(weeks=i+1) for i in range(weeks)]
        
        #start with the latest features
        current_features = features_df.iloc[-1][feature_cols].to_dict()
        
        weekly_predictions = []
        weekly_probabilities = []
        
        for i, next_date in enumerate(future_dates):
            #first predict if engagement will occur
            try:
                probas = classifier.predict_proba(pd.DataFrame([current_features], columns=feature_cols))[0]
                has_engagement_prob = probas[1] if len(probas) > 1 else (1.0 if hasattr(classifier, 'classes_') and classifier.classes_[0] == 1 else 0.0)
                has_engagement = has_engagement_prob >= 0.5
            except (IndexError, AttributeError):
                has_engagement = bool(classifier.predict(pd.DataFrame([current_features], columns=feature_cols))[0])
                has_engagement_prob = 1.0 if has_engagement else 0.0
        
            #if engagement predicted, use regression model to predict amount
            if has_engagement:
                pred_value = regressor.predict(pd.DataFrame([current_features], columns=feature_cols))[0]
            else:
                pred_value = 0.0
        
            weekly_predictions.append(pred_value)
            weekly_probabilities.append(has_engagement_prob)
        
            #update all weekly features (time + lag + rolling)
            current_features = self.update_weekly_forecast_features(
                features=current_features,
                new_value=pred_value,
                prediction_history=weekly_predictions,
                step_idx=i,
                next_date=next_date
            )

        
        #build weekly forecast DataFrame
        forecast_df = pd.DataFrame({
            'week_start_date': future_dates,
            'weekly_forecast': weekly_predictions,
            'engagement_probability': weekly_probabilities
        })
    
        #save forecast to file
        forecast_path = self.output_dir / 'forecasts' / f'forecast_rf_weekly_{client_name.replace(" ", "_").lower()}_{platform.replace(" ", "_").lower()}_total_engagement.csv'
        forecast_df.to_csv(forecast_path, index=False)
        
        
        #after calculating cls_accuracy, r2, mae, etc.
        if len(non_zero_test) > 0:
            diagnostic_entry = {
                'Client': client_name,
                'Platform': platform,
                'Train Samples': len(X_train_reg),
                'Test Samples': len(X_test_reg),
                'Train MAE': mean_absolute_error(y_train_reg, regressor.predict(X_train_reg)),
                'Test MAE': mean_absolute_error(y_test_reg, reg_preds),
                'Train R²': r2_score(y_train_reg, regressor.predict(X_train_reg)),
                'Test R²': r2_score(y_test_reg, reg_preds),
                'Model Saved?': 'Yes' if r2 >= self.minimum_r2_threshold else 'No'
            }
            self.model_diagnostics.append(diagnostic_entry)
        else:
            diagnostic_entry = {
                'Client': client_name,
                'Platform': platform,
                'Train Samples': len(X_train_reg),
                'Test Samples': 0,
                'Train MAE': mean_absolute_error(y_train_reg, regressor.predict(X_train_reg)),
                'Test MAE': None,
                'Train R²': r2_score(y_train_reg, regressor.predict(X_train_reg)),
                'Test R²': None,
                'Model Saved?': 'No'
            }
            self.model_diagnostics.append(diagnostic_entry)
       
        logger.info(f"Two-part weekly model forecast generated for {client_name} - {platform}")
        
        return forecast_df
    
    #get sm platforms used by clients
    def get_client_platforms(self, client):
        if self.daily_metrics is not None:

            client_data = self.daily_metrics[self.daily_metrics['Client_Name'] == client]
            if not client_data.empty:
                return sorted(client_data['platform_name'].unique())
        return []

    #precompute and save cross-platform opportunities for all clients, and print R² even if low. 
    def precompute_all_cross_platform_opportunities(self):
        if self.daily_metrics is None:
            logger.warning("Daily metrics not loaded. Cannot generate cross-platform opportunities.")
            return
    
        #mke sure output directory exists
        os.makedirs(self.output_dir / 'forecasts', exist_ok=True)
    
        #get all clients
        clients = self.daily_metrics['Client_Name'].unique()
        total_opportunities = 0
    
        for client in clients:
            #get correlation data is calculated
            correlation_data = self.analyze_cross_platform_effects(client)
            platforms = self.get_client_platforms(client)
    
            opportunities = []
    
            #get correllation data
            if correlation_data is None or correlation_data.empty:
                logger.warning(f"No correlation data available for {client}")
            elif len(platforms) <= 1:
                logger.info(f"Only one platform found for {client}, cannot generate cross-platform opportunities")
            else:
                #build opportunities
                for target_platform in platforms:
                    if target_platform not in correlation_data.columns:
                        continue
    
                    predictors = []
                    scores = []
    
                    for other_platform in platforms:
                        if other_platform != target_platform and other_platform in correlation_data.index:
                            try:
                                corr_value = correlation_data.loc[other_platform, target_platform]
                                if not np.isnan(corr_value):
                                    predictors.append(other_platform)
                                    scores.append(abs(corr_value))
                            except (KeyError, ValueError) as e:
                                logger.warning(f"Error accessing correlation for {other_platform} -> {target_platform}: {str(e)}")
                                continue
    
                    if predictors:
                        #sort them by correlation strength
                        sorted_indices = np.argsort(scores)[::-1]
                        top_predictors = [predictors[i] for i in sorted_indices[:2] if i < len(predictors)]
    
                        if not top_predictors:
                            continue
    
                        # Calculate R²
                        r2_score_val = scores[sorted_indices[0]] ** 2 if sorted_indices.size > 0 else 0
    
                        # Determine action based on R²
                        if r2_score_val > 0.7:
                            action = "Run joint campaigns"
                        elif r2_score_val > 0.5:
                            action = "Cross-promote content"
                        elif r2_score_val > 0.3:
                            action = "Test coordinated posting"
                        else:
                            action = "Monitor relationship"
    
                        opportunities.append({
                            'target': target_platform,
                            'predictors': ", ".join(top_predictors),
                            'r2': r2_score_val,
                            'action': action
                        })
    
            #save opportunities even if empty (to know we processed it)
            output_path = self.output_dir / 'forecasts' / f'cross_platform_opportunities_{client.replace(" ", "_").lower()}.csv'
    
            if opportunities:
                pd.DataFrame(opportunities).to_csv(output_path, index=False)
                total_opportunities += len(opportunities)
            else:
                #empty df with the expected columns
                empty_df = pd.DataFrame(columns=['target', 'predictors', 'r2', 'action'])
                empty_df.to_csv(output_path, index=False)
    
            logger.info(f"Generated {len(opportunities)} cross-platform opportunities for {client}")

    #Run main model pipeline
    def run_predictive_modeling(self, client_name=None):
        try:
            #get clients to process
            clients = [client_name] if client_name else self.daily_metrics['Client_Name'].unique().tolist()
            #process each client
            for client in clients:
                logger.info(f"Running predictive modeling for client: {client}")
                #get platforms for this client
                platforms = self.daily_metrics[self.daily_metrics['Client_Name'] == client]['platform_name'].unique()
                #process each platform
                for platform in platforms:
                    self.predict_engagement_two_part_model(client, platform)
                #analyze cross-platform and seasonal patterns
                self.analyze_cross_platform_effects(client)
                self.calculate_seasonal_patterns(client)
            #calculate and save summary metrics + pre-compute cpo's
            self.precompute_all_cross_platform_opportunities()
            self.calculate_summary_metrics()
            #save model log
            if self.model_diagnostics:
                diagnostics_df = pd.DataFrame(self.model_diagnostics)
                diagnostics_path = self.output_dir / 'forecasts' / 'model_diagnostics_log.csv'
                diagnostics_df.to_csv(diagnostics_path, index=False)
                logger.info(f"Saved model diagnostics log to {diagnostics_path}")

            logger.info("Predictive modeling completed successfully")
            return True
        #error if modeling fails
        except Exception as e:
            logger.error(f"Error in predictive modeling: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

if __name__ == "__main__":
    #set paths for data and output
    data_dir = Path(r"C:\Users\Arthu\Documents\Capstone\1_ETL\3. Processed Data")
    output_dir = "results"
    
    #run predictive modeling with reduced plot generation
    modeling = SocialMediaPredictiveModeling(
        data_dir=data_dir, 
        output_dir=output_dir    )
    
    #run for all clients or specify a client name
    modeling.run_predictive_modeling()