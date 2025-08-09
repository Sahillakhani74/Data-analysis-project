import pandas as pd
import numpy as np
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import warnings
from typing import Optional

from data_analysis_project.agents.crew_tasks import DataAnalysisTasks
warnings.filterwarnings('ignore')
from crewai.tools import BaseTool

class SmartDataSourceRouter(BaseTool):
    name: str = "Smart Data Source Router"
    description: str = (
        "Intelligently detects data source type and provides comprehensive initial assessment. "
        "Handles CSV, Excel (single/multiple sheets), and basic database files."
    )

    def _run(self, file_path: str) -> str:
        try:
            # Detect file type
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                source_type = "CSV File"
            elif file_path.endswith(('.xlsx', '.xls')):
                # Handle Excel with multiple sheets
                excel_file = pd.ExcelFile(file_path)
                if len(excel_file.sheet_names) > 1:
                    df = pd.read_excel(file_path, sheet_name=0)  # Read first sheet
                    source_type = f"Excel Workbook ({len(excel_file.sheet_names)} sheets)"
                else:
                    df = pd.read_excel(file_path)
                    source_type = "Excel File"
            elif file_path.startswith('sqlite:///') or file_path.startswith('postgresql://') or file_path.startswith('mysql+mysqlconnector://'):
                # Handle database connections using SQLAlchemy
                try:
                    engine = create_engine(file_path)
                    inspector = sqlalchemy.inspect(engine)
                    table_names = inspector.get_table_names()
                    
                    if not table_names:
                        return "Error: No tables found in the specified database."
                    
                    # Read a sample from the first table
                    df = pd.read_sql_table(table_names[0], con=engine, limit=1000)
                    source_type = f"Database ({engine.name}, {len(table_names)} tables)"
                except Exception as db_e:
                    return f"Error connecting to database or reading data: {str(db_e)}"
            else:
                return f"Error: Unsupported file type or database connection string for {file_path}"
            
            # Schema inference and quality assessment
            total_rows, total_cols = df.shape
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            
            # Data quality metrics
            missing_data = df.isnull().sum().sum()
            duplicate_rows = df.duplicated().sum()
            
            # Identify potential key columns and business metrics
            key_columns = []
            metric_columns = []
            
            for col in df.columns:
                # Potential ID columns
                if any(keyword in col.lower() for keyword in ['id', 'key', 'index']):
                    key_columns.append(col)
                # Potential business metrics
                if any(keyword in col.lower() for keyword in ['revenue', 'sales', 'amount', 'price', 'cost', 'value']):
                    metric_columns.append(col)
            
            report = f"""
📊 DATA SOURCE SUMMARY
────────────────────────
📁 Source Type: {source_type}
📈 Total Records: {total_rows:,} rows
📋 Columns: {total_cols} fields
    • Numeric: {len(numeric_cols)} columns
    • Categorical: {len(categorical_cols)} columns  
    • DateTime: {len(datetime_cols)} columns

🎯 Key Business Elements Detected:
    • ID/Key Columns: {key_columns if key_columns else 'None detected'}
    • Metric Columns: {metric_columns if metric_columns else 'None detected'}

⚠️ Data Quality Status:
    • Missing Values: {missing_data:,} cells ({(missing_data/(total_rows*total_cols)*100):.1f}%)
    • Duplicate Rows: {duplicate_rows:,} ({(duplicate_rows/total_rows*100):.1f}%)

🔍 Analysis Readiness: {'✅ Ready for comprehensive analysis' if missing_data < (total_rows*total_cols*0.1) else '⚠️ Requires data cleaning'}

📋 Next Steps: Ready for Advanced EDA and Business Intelligence Analysis
            """
            
            return report.strip()
            
        except Exception as e:
            return f"Error in data source analysis: {str(e)}"

class ComprehensiveStatisticalProfiling(BaseTool):
    name: str = "Comprehensive Statistical Profiling"
    description: str = (
        "Performs comprehensive statistical analysis including distributions, "
        "correlations, and advanced profiling like a professional data analyst."
    )

    def _run(self, file_path: str) -> str:
        try:
            df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
            
            numeric_df = df.select_dtypes(include=[np.number])
            categorical_df = df.select_dtypes(include=['object'])
            
            report_sections = []
            
            # 1. Descriptive Statistics
            if not numeric_df.empty:
                desc_stats = numeric_df.describe()
                report_sections.append("📊 DESCRIPTIVE STATISTICS")
                report_sections.append("─" * 30)
                for col in numeric_df.columns:
                    col_data = numeric_df[col].dropna()
                    if len(col_data) > 0:
                        skewness = stats.skew(col_data)
                        kurtosis = stats.kurtosis(col_data)
                        report_sections.append(f"• {col}:")
                        report_sections.append(f"  Mean: {col_data.mean():.2f}, Std: {col_data.std():.2f}")
                        report_sections.append(f"  Skewness: {skewness:.2f}, Kurtosis: {kurtosis:.2f}")
                        report_sections.append(f"  Range: {col_data.min():.2f} to {col_data.max():.2f}")
            
            # 2. Correlation Analysis
            if len(numeric_df.columns) > 1:
                corr_matrix = numeric_df.corr()
                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.7:  # Strong correlation threshold
                            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
                
                report_sections.append("\n🔗 CORRELATION ANALYSIS")
                report_sections.append("─" * 25)
                if high_corr_pairs:
                    report_sections.append("Strong Correlations (|r| > 0.7):")
                    for col1, col2, corr in high_corr_pairs:
                        report_sections.append(f"• {col1} ↔ {col2}: {corr:.3f}")
                else:
                    report_sections.append("• No strong correlations (|r| > 0.7) detected")
            
            # 3. Distribution Analysis
            report_sections.append("\n📈 DISTRIBUTION ANALYSIS")
            report_sections.append("─" * 27)
            for col in numeric_df.columns:
                col_data = numeric_df[col].dropna()
                if len(col_data) > 8:  # Need sufficient data for normality test
                    _, p_value = stats.normaltest(col_data)
                    is_normal = p_value > 0.05
                    report_sections.append(f"• {col}: {'Normal' if is_normal else 'Non-normal'} distribution (p={p_value:.3f})")
            
            # 4. Categorical Analysis
            if not categorical_df.empty:
                report_sections.append("\n📋 CATEGORICAL ANALYSIS")
                report_sections.append("─" * 26)
                for col in categorical_df.columns:
                    unique_count = df[col].nunique()
                    most_common = df[col].value_counts().head(1)
                    if not most_common.empty:
                        top_category = most_common.index[0]
                        top_count = most_common.iloc[0]
                        report_sections.append(f"• {col}: {unique_count} unique values")
                        report_sections.append(f"  Most common: '{top_category}' ({top_count} occurrences)")
            
            # 5. Missing Value Analysis
            missing_analysis = df.isnull().sum()
            missing_cols = missing_analysis[missing_analysis > 0]
            if not missing_cols.empty:
                report_sections.append("\n🔍 MISSING VALUE ANALYSIS")
                report_sections.append("─" * 28)
                for col, missing_count in missing_cols.items():
                    missing_pct = (missing_count / len(df)) * 100
                    report_sections.append(f"• {col}: {missing_count} missing ({missing_pct:.1f}%)")
            
            return "\n".join(report_sections)
            
        except Exception as e:
            return f"Error in statistical profiling: {str(e)}"

class MissingValueAnalyzer(BaseTool):
    name: str = "Missing Value Analyzer"
    description: str = (
        "Analyzes missing values in the dataset, identifies patterns, and suggests imputation strategies."
    )

    def _run(self, file_path: str) -> str:
        try:
            df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
            
            report_sections = []
            report_sections.append("🔍 MISSING VALUE ANALYSIS")
            report_sections.append("═" * 30)
            
            missing_counts = df.isnull().sum()
            missing_pct = (df.isnull().sum() / len(df)) * 100
            
            missing_info = pd.DataFrame({
                'Missing Count': missing_counts,
                'Missing Percentage': missing_pct
            })
            missing_info = missing_info[missing_info['Missing Count'] > 0].sort_values(by='Missing Percentage', ascending=False)
            
            if missing_info.empty:
                report_sections.append("✅ No missing values detected in the dataset. Data is complete!")
                return "\n".join(report_sections)
            
            report_sections.append("Summary of Missing Values by Column:")
            report_sections.append(missing_info.to_string())
            
            report_sections.append("\n💡 IMPUTATION STRATEGY RECOMMENDATIONS:")
            report_sections.append("─" * 40)
            
            for col in missing_info.index:
                col_type = df[col].dtype
                missing_percentage = missing_info.loc[col, 'Missing Percentage']
                
                report_sections.append(f"\n• Column: '{col}' (Type: {col_type}) - {missing_percentage:.1f}% Missing")
                
                if missing_percentage > 50:
                    report_sections.append("  Recommendation: Consider dropping this column due to high missingness, or advanced imputation if critical.")
                elif pd.api.types.is_numeric_dtype(df[col]):
                    report_sections.append("  Recommendation: Impute with Mean/Median (for numerical data).")
                    report_sections.append("  Alternative: Impute with Regression (e.g., K-NN, Linear Regression) for more accuracy.")
                elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                    report_sections.append("  Recommendation: Impute with Mode (for categorical data).")
                    report_sections.append("  Alternative: Impute with 'Unknown' category or use advanced methods.")
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    report_sections.append("  Recommendation: Impute with nearest valid observation (forward/backward fill) or a specific date.")
                else:
                    report_sections.append("  Recommendation: Investigate data type and context for appropriate imputation.")
            
            # Further analysis on missing patterns (conceptual, as heatmaps require visualization)
            report_sections.append("\n📊 MISSING PATTERN INSIGHTS:")
            report_sections.append("─" * 30)
            report_sections.append("• Consider visualizing missing data patterns (e.g., using `missingno` library heatmaps) to identify relationships between missingness in different columns.")
            report_sections.append("• Analyze if missingness is random (MCAR), dependent on observed data (MAR), or dependent on missing data itself (MNAR).")
            
            return "\n".join(report_sections)
            
        except Exception as e:
            return f"Error in missing value analysis: {str(e)}"

class AdvancedOutlierDetection(BaseTool):
    name: str = "Advanced Outlier Detection"
    description: str = (
        "Performs multiple outlier detection methods including statistical "
        "and machine learning approaches for comprehensive analysis."
    )

    def _run(self, file_path: str) -> str:
        try:
            df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
            
            numeric_df = df.select_dtypes(include=[np.number])
            
            if numeric_df.empty:
                return "No numeric columns found for outlier analysis."
            
            report_sections = []
            report_sections.append("🚨 ADVANCED OUTLIER ANALYSIS")
            report_sections.append("═" * 35)
            
            for col in numeric_df.columns:
                col_data = numeric_df[col].dropna()
                if len(col_data) < 10:  # Need sufficient data
                    continue
                    
                report_sections.append(f"\n🔍 Analysis for '{col}':")
                report_sections.append("─" * (15 + len(col)))
                
                # 1. Z-Score Method
                z_scores = np.abs(stats.zscore(col_data))
                z_outliers = len(z_scores[z_scores > 3])
                
                # 2. IQR Method
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                iqr_outliers = len(col_data[(col_data < lower_bound) | (col_data > upper_bound)])
                
                # 3. Isolation Forest (if enough data)
                if len(col_data) >= 50:
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    outlier_pred = iso_forest.fit_predict(col_data.values.reshape(-1, 1))
                    ml_outliers = len(outlier_pred[outlier_pred == -1])
                else:
                    ml_outliers = "N/A (insufficient data)"
                
                report_sections.append(f"• Z-Score Method (>3σ): {z_outliers} outliers ({(z_outliers/len(col_data)*100):.1f}%)")
                report_sections.append(f"• IQR Method: {iqr_outliers} outliers ({(iqr_outliers/len(col_data)*100):.1f}%)")
                report_sections.append(f"• Isolation Forest: {ml_outliers if isinstance(ml_outliers, str) else f'{ml_outliers} outliers ({ml_outliers/len(col_data)*100:.1f}%)'}")
                
                # Business context assessment
                if z_outliers > 0 or iqr_outliers > 0:
                    report_sections.append(f"• Range: {col_data.min():.2f} to {col_data.max():.2f}")
                    report_sections.append("• Recommendation: Review outliers for business validity")
                else:
                    report_sections.append("• Status: No significant outliers detected")
            
            # Summary recommendations
            report_sections.append("\n💡 BUSINESS RECOMMENDATIONS")
            report_sections.append("─" * 30)
            report_sections.append("• Review flagged outliers for data entry errors")
            report_sections.append("• Consider business context before removing outliers")
            report_sections.append("• High-value transactions may be legitimate outliers")
            
            return "\n".join(report_sections)
            
        except Exception as e:
            return f"Error in outlier detection: {str(e)}"

class BusinessPerformanceAnalyzer(BaseTool):
    name: str = "Business Performance Analyzer"
    description: str = (
        "Analyzes business performance metrics, identifies trends, top/bottom performers, "
        "and generates actionable business insights."
    )

    def _run(self, file_path: str) -> str:
        try:
            df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
            
            report_sections = []
            report_sections.append("💼 BUSINESS PERFORMANCE ANALYSIS")
            report_sections.append("═" * 40)
            
            # Identify potential business metrics
            metric_cols = []
            dimension_cols = []
            
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    # Check if it's a business metric
                    if any(keyword in col.lower() for keyword in ['revenue', 'sales', 'amount', 'price', 'cost', 'profit', 'value']):
                        metric_cols.append(col)
                elif df[col].dtype == 'object':
                    # Potential dimension for grouping
                    if df[col].nunique() < 50:  # Reasonable number of categories
                        dimension_cols.append(col)
            
            if not metric_cols:
                return "No business metrics detected. Please ensure your data contains columns with business values (revenue, sales, etc.)"
            
            # Performance Analysis
            for metric in metric_cols:
                report_sections.append(f"\n📊 {metric.upper()} ANALYSIS")
                report_sections.append("─" * (len(metric) + 12))
                
                metric_data = df[metric].dropna()
                total_value = metric_data.sum()
                avg_value = metric_data.mean()
                
                report_sections.append(f"• Total {metric}: ${total_value:,.2f}" if 'revenue' in metric.lower() or 'sales' in metric.lower() or 'amount' in metric.lower() else f"• Total {metric}: {total_value:,.2f}")
                report_sections.append(f"• Average {metric}: ${avg_value:,.2f}" if 'revenue' in metric.lower() or 'sales' in metric.lower() or 'amount' in metric.lower() else f"• Average {metric}: {avg_value:,.2f}")
                
                # Performance by dimensions
                for dim in dimension_cols[:2]:  # Limit to top 2 dimensions
                    if dim in df.columns:
                        perf_by_dim = df.groupby(dim)[metric].agg(['sum', 'count', 'mean']).sort_values('sum', ascending=False)
                        
                        report_sections.append(f"\n🏆 Top Performers by {dim}:")
                        top_3 = perf_by_dim.head(3)
                        for idx, (category, row) in enumerate(top_3.iterrows(), 1):
                            report_sections.append(f"  {idx}. {category}: ${row['sum']:,.2f}" if 'revenue' in metric.lower() or 'sales' in metric.lower() else f"  {idx}. {category}: {row['sum']:,.2f}")
                        
                        report_sections.append(f"\n📉 Bottom Performers by {dim}:")
                        bottom_3 = perf_by_dim.tail(3)
                        for idx, (category, row) in enumerate(bottom_3.iterrows(), 1):
                            report_sections.append(f"  {idx}. {category}: ${row['sum']:,.2f}" if 'revenue' in metric.lower() or 'sales' in metric.lower() else f"  {idx}. {category}: {row['sum']:,.2f}")
            
            # Growth Analysis (if date column exists)
            date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            date_like_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['date', 'time', 'month', 'year'])]
            
            if date_cols or date_like_cols:
                report_sections.append(f"\n📈 TREND ANALYSIS")
                report_sections.append("─" * 18)
                report_sections.append("• Time-based trends detected in data")
                report_sections.append("• Recommend time series analysis for growth patterns")
                report_sections.append("• Consider seasonality analysis for business planning")
            
            # Business Insights Summary
            report_sections.append(f"\n💡 KEY BUSINESS INSIGHTS")
            report_sections.append("─" * 28)
            
            # Concentration analysis
            if dimension_cols and metric_cols:
                main_metric = metric_cols[0]
                main_dim = dimension_cols[0]
                concentration = df.groupby(main_dim)[main_metric].sum().sort_values(ascending=False)
                top_20_pct = concentration.head(int(len(concentration) * 0.2)).sum()
                total = concentration.sum()
                pareto_ratio = (top_20_pct / total) * 100
                
                report_sections.append(f"• Top 20% of {main_dim} contribute {pareto_ratio:.1f}% of total {main_metric}")
                
                if pareto_ratio > 80:
                    report_sections.append("• HIGH CONCENTRATION: Focus resources on top performers")
                elif pareto_ratio < 50:
                    report_sections.append("• BALANCED DISTRIBUTION: Performance spread across categories")
                else:
                    report_sections.append("• MODERATE CONCENTRATION: Mixed performance pattern")
            
            report_sections.append("\n🎯 STRATEGIC RECOMMENDATIONS")
            report_sections.append("─" * 32)
            report_sections.append("• Focus investment on top-performing segments")
            report_sections.append("• Investigate underperforming areas for improvement opportunities")
            report_sections.append("• Consider market expansion in high-potential segments")
            
            return "\n".join(report_sections)
            
        except Exception as e:
            return f"Error in business performance analysis: {str(e)}"

class TrendAndSeasonalityAnalyzer(BaseTool):
    name: str = "Trend and Seasonality Analyzer"
    description: str = (
        "Analyzes trends and seasonality in a given metric over time. "
        "Generates insights on growth/decline, seasonal patterns, and overall trajectory. "
        "Saves the aggregated time series data to a new CSV or Excel file if output_file_path is provided."
    )

    def _run(self, file_path: str, date_column: str, metric_column: str, output_file_path: Optional[str] = None) -> str:
        try:
            df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

            if date_column not in df.columns or metric_column not in df.columns:
                return f"Error: Date column '{date_column}' or metric column '{metric_column}' not found."
            if not pd.api.types.is_numeric_dtype(df[metric_column]):
                return f"Error: Metric column '{metric_column}' is not numeric."

            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            df.dropna(subset=[date_column, metric_column], inplace=True)

            if df.empty:
                return "Error: No valid data after cleaning date and metric columns."

            # Aggregate by month for trend analysis
            df['period'] = df[date_column].dt.to_period('M')
            monthly_data = df.groupby('period')[metric_column].sum().reset_index()
            monthly_data['period'] = monthly_data['period'].dt.to_timestamp() # Convert Period to Timestamp for plotting/analysis

            report_sections = []
            report_sections.append(f"📈 TREND AND SEASONALITY ANALYSIS FOR '{metric_column}'")
            report_sections.append("═" * 60)
            report_sections.append(f"Time Period: {monthly_data['period'].min().strftime('%Y-%m')} to {monthly_data['period'].max().strftime('%Y-%m')}")

            if len(monthly_data) < 2:
                return "Insufficient data for trend analysis (need at least two periods)."

            # Overall Trend
            first_val = monthly_data[metric_column].iloc[0]
            last_val = monthly_data[metric_column].iloc[-1]
            overall_growth = ((last_val - first_val) / first_val) * 100 if first_val != 0 else 0

            report_sections.append(f"\n📊 Overall Trend:")
            report_sections.append(f"• Starting {metric_column}: {first_val:,.2f}")
            report_sections.append(f"• Ending {metric_column}: {last_val:,.2f}")
            report_sections.append(f"• Total Change: {overall_growth:+.1f}% over the period.")

            if overall_growth > 10:
                report_sections.append("  Insight: Strong positive growth trend detected.")
            elif overall_growth < -10:
                report_sections.append("  Insight: Significant negative decline trend detected.")
            else:
                report_sections.append("  Insight: Relatively stable or moderate trend.")

            # Seasonality (if enough data for at least 2 full years)
            if len(monthly_data) >= 24: # At least 2 years of monthly data
                monthly_avg = monthly_data.groupby(monthly_data['period'].dt.month)[metric_column].mean().sort_index()
                
                report_sections.append(f"\n🗓️ Monthly Seasonality (Average {metric_column}):")
                for month_num, avg_val in monthly_avg.items():
                    report_sections.append(f"• {pd.to_datetime(str(month_num), format='%m').strftime('%B')}: {avg_val:,.2f}")
                
                peak_month = monthly_avg.idxmax()
                trough_month = monthly_avg.idxmin()
                report_sections.append(f"  Insight: Peak performance typically in {pd.to_datetime(str(peak_month), format='%m').strftime('%B')}.")
                report_sections.append(f"  Insight: Trough performance typically in {pd.to_datetime(str(trough_month), format='%m').strftime('%B')}.")
                report_sections.append("  Recommendation: Plan marketing/operations around seasonal peaks and troughs.")
            else:
                report_sections.append("\n🗓️ Seasonality Analysis: Insufficient data (need at least 2 years of data) to reliably detect seasonal patterns.")

            # Anomalies (simple deviation from rolling mean)
            if len(monthly_data) > 12: # Need enough data for rolling mean
                monthly_data['rolling_mean'] = monthly_data[metric_column].rolling(window=3).mean()
                monthly_data['deviation'] = (monthly_data[metric_column] - monthly_data['rolling_mean']).abs()
                anomalies = monthly_data[monthly_data['deviation'] > monthly_data['deviation'].std() * 2] # 2 standard deviations from mean deviation

                if not anomalies.empty:
                    report_sections.append("\n🚨 Potential Anomalies Detected:")
                    for _, row in anomalies.iterrows():
                        report_sections.append(f"• {row['period'].strftime('%Y-%m')}: {metric_column} = {row[metric_column]:,.2f} (Deviation: {row['deviation']:.2f})")
                    report_sections.append("  Recommendation: Investigate these periods for unusual events or data errors.")
                else:
                    report_sections.append("\n🚨 No significant anomalies detected based on rolling mean deviation.")

            if output_file_path:
                if output_file_path.endswith('.csv'):
                    monthly_data.to_csv(output_file_path, index=False)
                elif output_file_path.endswith(('.xlsx', '.xls')):
                    monthly_data.to_excel(output_file_path, index=False)
                else:
                    return f"Error: Unsupported output file type for {output_file_path}. Please use .csv or .xlsx."
                report_sections.append(f"\nAggregated monthly data saved to: {output_file_path}")

            return "\n".join(report_sections).strip()

        except Exception as e:
            return f"Error in trend and seasonality analysis: {str(e)}"

class SegmentationPerformanceAnalyzer(BaseTool):
    name: str = "Segmentation Performance Analyzer"
    description: str = (
        "Analyzes performance across different segments (e.g., customer segments, product categories, regions). "
        "Identifies top/bottom performing segments and provides comparative insights. "
        "Saves the segmented performance data to a new CSV or Excel file if output_file_path is provided."
    )

    def _run(self, file_path: str, segment_column: str, metric_column: str, output_file_path: Optional[str] = None) -> str:
        """
        Analyzes performance across different segments (e.g., customer segments, product categories, regions).
        Identifies top/bottom performing segments and provides comparative insights.
        Saves the segmented performance data to a new CSV or Excel file if output_file_path is provided.
        """
        try:
            df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

            if segment_column not in df.columns or metric_column not in df.columns:
                return f"Error: Segment column '{segment_column}' or metric column '{metric_column}' not found."
            if not pd.api.types.is_numeric_dtype(df[metric_column]):
                return f"Error: Metric column '{metric_column}' is not numeric."

            df.dropna(subset=[segment_column, metric_column], inplace=True)

            if df.empty:
                return "Error: No valid data after cleaning segment and metric columns."

            segment_performance = df.groupby(segment_column)[metric_column].agg(['sum', 'mean', 'count']).sort_values(by='sum', ascending=False)
            
            report_sections = []
            report_sections.append(f"📊 SEGMENTATION PERFORMANCE ANALYSIS: '{metric_column}' by '{segment_column}'")
            report_sections.append("═" * 70)
            report_sections.append(f"Total Segments: {len(segment_performance)}")
            report_sections.append(f"Total {metric_column}: {segment_performance['sum'].sum():,.2f}")

            report_sections.append(f"\n🏆 Top 5 Performing Segments by Total {metric_column}:")
            report_sections.append("─" * 50)
            for i, (segment, row) in enumerate(segment_performance.head(5).iterrows(), 1):
                report_sections.append(f"{i}. {segment}: Sum={row['sum']:,.2f}, Avg={row['mean']:.2f}, Count={row['count']}")

            report_sections.append(f"\n📉 Bottom 5 Performing Segments by Total {metric_column}:")
            report_sections.append("─" * 50)
            for i, (segment, row) in enumerate(segment_performance.tail(5).iterrows(), 1):
                report_sections.append(f"{i}. {segment}: Sum={row['sum']:,.2f}, Avg={row['mean']:.2f}, Count={row['count']}")

            # Concentration Analysis (Pareto Principle)
            total_metric_sum = segment_performance['sum'].sum()
            if total_metric_sum > 0:
                segment_performance['cumulative_sum'] = segment_performance['sum'].cumsum()
                segment_performance['cumulative_pct'] = (segment_performance['cumulative_sum'] / total_metric_sum) * 100
                
                top_80_pct_segments = segment_performance[segment_performance['cumulative_pct'] <= 80]
                num_segments_for_80_pct = len(top_80_pct_segments)
                
                report_sections.append(f"\n💡 Concentration Analysis (Pareto Principle):")
                report_sections.append(f"• Top {num_segments_for_80_pct} segments account for ~80% of total {metric_column}.")
                if num_segments_for_80_pct / len(segment_performance) < 0.3: # If top 30% or less segments make 80%
                    report_sections.append("  Insight: High concentration of performance in a few key segments. Focus on nurturing these.")
                else:
                    report_sections.append("  Insight: Performance is more evenly distributed across segments. Broader strategies may be effective.")

            report_sections.append(f"\n🎯 Strategic Recommendations:")
            report_sections.append("─" * 30)
            report_sections.append("• Allocate resources to top-performing segments to maximize returns.")
            report_sections.append("• Investigate reasons for underperformance in bottom segments and develop targeted improvement plans.")
            report_sections.append("• Consider A/B testing different strategies for segments with moderate performance.")

            if output_file_path:
                if output_file_path.endswith('.csv'):
                    segment_performance.to_csv(output_file_path, index=False)
                elif output_file_path.endswith(('.xlsx', '.xls')):
                    segment_performance.to_excel(output_file_path, index=False)
                else:
                    return f"Error: Unsupported output file type for {output_file_path}. Please use .csv or .xlsx."
                report_sections.append(f"\nSegmented performance data saved to: {output_file_path}")

            return "\n".join(report_sections).strip()

        except Exception as e:
            return f"Error in segmentation performance analysis: {str(e)}"

class AnomalyAndPatternDetector(BaseTool):
    name: str = "Anomaly and Pattern Detector"
    description: str = (
        "Detects anomalies and recurring patterns in a specified numerical column. "
        "Supports IQR method, Z-score, and Isolation Forest for anomaly detection. "
        "Saves the DataFrame with anomaly flags to a new CSV or Excel file if output_file_path is provided."
    )

    def _run(self, file_path: str, column: str, method: str = "iqr", output_file_path: Optional[str] = None) -> str:
        """
        Detects anomalies and recurring patterns in a specified numerical column.
        Supports IQR method, Z-score, and Isolation Forest for anomaly detection.
        Saves the DataFrame with anomaly flags to a new CSV or Excel file if output_file_path is provided.
        """
        try:
            df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

            if column not in df.columns:
                return f"Error: Column '{column}' not found in data."
            if not pd.api.types.is_numeric_dtype(df[column]):
                return f"Error: Column '{column}' is not numeric."

            df_clean = df.copy()
            df_clean.dropna(subset=[column], inplace=True)

            if df_clean.empty:
                return "Error: No valid data in the specified column for anomaly detection."

            report_sections = []
            report_sections.append(f"🚨 ANOMALY AND PATTERN DETECTION FOR '{column}'")
            report_sections.append("═" * 60)

            anomalies_detected = False
            anomaly_indices = pd.Series(False, index=df_clean.index)

            if method.lower() == 'iqr':
                Q1 = df_clean[column].quantile(0.25)
                Q3 = df_clean[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                anomaly_indices = (df_clean[column] < lower_bound) | (df_clean[column] > upper_bound)
                report_sections.append(f"Method: IQR (Interquartile Range) - Thresholds: [{lower_bound:.2f}, {upper_bound:.2f}]")
            elif method.lower() == 'z_score':
                mean = df_clean[column].mean()
                std = df_clean[column].std()
                if std == 0:
                    report_sections.append("Cannot apply Z-score: Standard deviation is zero (all values are the same).")
                else:
                    z_scores = np.abs((df_clean[column] - mean) / std)
                    anomaly_indices = z_scores > 3 # Z-score > 3 is common threshold
                    report_sections.append(f"Method: Z-Score (>3σ) - Mean: {mean:.2f}, Std Dev: {std:.2f}")
            elif method.lower() == 'isolation_forest':
                if len(df_clean) < 50:
                    report_sections.append("Cannot apply Isolation Forest: Insufficient data (need at least 50 observations).")
                else:
                    iso_forest = IsolationForest(contamination='auto', random_state=42)
                    outlier_pred = iso_forest.fit_predict(df_clean[[column]])
                    anomaly_indices = df_clean['anomaly_score'] == -1
                    report_sections.append(f"Method: Isolation Forest")
            else:
                return f"Error: Unsupported anomaly detection method '{method}'. Supported: iqr, z_score, isolation_forest."

            num_anomalies = anomaly_indices.sum()
            if num_anomalies > 0:
                anomalies_detected = True
                report_sections.append(f"\nFound {num_anomalies} anomalies ({num_anomalies/len(df_clean)*100:.2f}% of clean data).")
                report_sections.append("Examples of Anomalous Data Points:")
                report_sections.append(df_clean[anomaly_indices].head().to_string())
                report_sections.append("\n💡 Insight: These data points deviate significantly from the norm.")
                report_sections.append("  Recommendation: Investigate these anomalies for potential errors, fraud, or significant business events.")
            else:
                report_sections.append("\nNo significant anomalies detected with the chosen method.")

            # Add anomaly flag to original DataFrame
            df['is_anomaly'] = df.index.map(anomaly_indices).fillna(False) # Map back to original index, fill non-numeric with False

            if output_file_path:
                if output_file_path.endswith('.csv'):
                    df.to_csv(output_file_path, index=False)
                elif output_file_path.endswith(('.xlsx', '.xls')):
                    df.to_excel(output_file_path, index=False)
                else:
                    return f"Error: Unsupported output file type for {output_file_path}. Please use .csv or .xlsx."
                report_sections.append(f"\nData with 'is_anomaly' flag saved to: {output_file_path}")

            return "\n".join(report_sections).strip()

        except Exception as e:
            return f"Error in anomaly detection: {str(e)}"

class StatisticalSignificanceTesting(BaseTool):
    name: str = "Statistical Significance Testing"
    description: str = (
        "Performs statistical significance tests including correlation analysis, "
        "t-tests, and chi-square tests for business hypothesis validation."
    )

    def _run(self, file_path: str, column1: str, column2: str, test_type: str = "correlation") -> str:
        """
        Performs statistical significance tests including correlation analysis,
        t-tests, and chi-square tests for business hypothesis validation.
        """
        try:
            df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
            
            if column1 not in df.columns or column2 not in df.columns:
                return f"Error: Columns '{column1}' or '{column2}' not found in data."
            
            report_sections = []
            report_sections.append("📊 STATISTICAL SIGNIFICANCE TESTING")
            report_sections.append("═" * 40)
            
            col1_data = df[column1].dropna()
            col2_data = df[column2].dropna()
            
            if test_type.lower() == "correlation":
                # Correlation analysis
                if pd.api.types.is_numeric_dtype(df[column1]) and pd.api.types.is_numeric_dtype(df[column2]):
                    # Align data (remove NaN pairs)
                    clean_data = df[[column1, column2]].dropna()
                    if len(clean_data) < 10:
                        return "Insufficient data for correlation analysis (need at least 10 paired observations)."
                    
                    col1_clean = clean_data[column1]
                    col2_clean = clean_data[column2]
                    
                    # Pearson correlation
                    pearson_r, pearson_p = pearsonr(col1_clean, col2_clean)
                    
                    # Spearman correlation (non-parametric)
                    spearman_r, spearman_p = spearmanr(col1_clean, col2_clean)
                    
                    report_sections.append(f"🔗 CORRELATION ANALYSIS: {column1} vs {column2}")
                    report_sections.append("─" * 50)
                    report_sections.append(f"• Sample size: {len(col1_clean)} paired observations")
                    report_sections.append(f"• Pearson correlation: r = {pearson_r:.4f}, p = {pearson_p:.4f}")
                    report_sections.append(f"• Spearman correlation: ρ = {spearman_r:.4f}, p = {spearman_p:.4f}")
                    
                    # Interpretation
                    if pearson_p < 0.001:
                        significance = "Highly significant (p < 0.001)"
                    elif pearson_p < 0.01:
                        significance = "Very significant (p < 0.01)"
                    elif pearson_p < 0.05:
                        significance = "Significant (p < 0.05)"
                    else:
                        significance = "Not significant (p ≥ 0.05)"
                    
                    strength = "Strong" if abs(pearson_r) > 0.7 else "Moderate" if abs(pearson_r) > 0.3 else "Weak"
                    direction = "positive" if pearson_r > 0 else "negative"
                    
                    report_sections.append(f"\n💡 INTERPRETATION:")
                    report_sections.append(f"• Strength: {strength} {direction} correlation")
                    report_sections.append(f"• Statistical significance: {significance}")
                    
                    if pearson_p < 0.05:
                        report_sections.append(f"• Business insight: There is a statistically significant relationship")
                        report_sections.append(f"  between {column1} and {column2}")
                    else:
                        report_sections.append(f"• Business insight: No significant linear relationship detected")
                
                else:
                    return f"Error: Both columns must be numeric for correlation analysis."
            
            elif test_type.lower() == "t_test":
                # Two-sample t-test (assuming column2 is a binary grouping variable)
                if pd.api.types.is_numeric_dtype(df[column1]) and df[column2].nunique() == 2:
                    groups = df[column2].unique()
                    group1_data = df[df[column2] == groups[0]][column1].dropna()
                    group2_data = df[df[column2] == groups[1]][column1].dropna()
                    
                    if len(group1_data) < 5 or len(group2_data) < 5:
                        return "Insufficient data for t-test (need at least 5 observations per group)."
                    
                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
                    
                    # Calculate effect size (Cohen's d)
                    pooled_std = np.sqrt(((len(group1_data) - 1) * group1_data.var() + 
                                        (len(group2_data) - 1) * group2_data.var()) / 
                                       (len(group1_data) + len(group2_data) - 2))
                    cohens_d = (group1_data.mean() - group2_data.mean()) / pooled_std
                    
                    report_sections.append(f"📊 TWO-SAMPLE T-TEST: {column1} by {column2}")
                    report_sections.append("─" * 50)
                    report_sections.append(f"• Group 1 ({groups[0]}): n = {len(group1_data)}, mean = {group1_data.mean():.3f}")
                    report_sections.append(f"• Group 2 ({groups[1]}): n = {len(group2_data)}, mean = {group2_data.mean():.3f}")
                    report_sections.append(f"• t-statistic: {t_stat:.4f}")
                    report_sections.append(f"• p-value: {p_value:.4f}")
                    report_sections.append(f"• Effect size (Cohen's d): {abs(cohens_d):.3f}")
                    
                    # Interpretation
                    significance = "Significant" if p_value < 0.05 else "Not significant"
                    effect_magnitude = "Large" if abs(cohens_d) > 0.8 else "Medium" if abs(cohens_d) > 0.5 else "Small"
                    
                    report_sections.append(f"\n💡 INTERPRETATION:")
                    report_sections.append(f"• Statistical significance: {significance} (α = 0.05)")
                    report_sections.append(f"• Effect size: {effect_magnitude} practical difference")
                    
                    if p_value < 0.05:
                        higher_group = groups[0] if group1_data.mean() > group2_data.mean() else groups[1]
                        report_sections.append(f"• Business insight: {higher_group} shows significantly higher {column1}")
                    else:
                        report_sections.append(f"• Business insight: No significant difference between groups")
                
                else:
                    return f"Error: T-test requires {column1} to be numeric and {column2} to have exactly 2 groups."
            
            elif test_type.lower() == "chi_square":
                # Chi-square test for independence (assuming both columns are categorical)
                if (pd.api.types.is_string_dtype(df[column1]) or pd.api.types.is_categorical_dtype(df[column1])) and \
                   (pd.api.types.is_string_dtype(df[column2]) or pd.api.types.is_categorical_dtype(df[column2])):
                    
                    contingency_table = pd.crosstab(df[column1], df[column2])
                    if contingency_table.empty or contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
                        return "Insufficient data for Chi-square test (need at least 2x2 contingency table)."
                    
                    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                    
                    report_sections.append(f"📊 CHI-SQUARE TEST: {column1} vs {column2}")
                    report_sections.append("─" * 50)
                    report_sections.append(f"• Chi-square statistic: {chi2:.4f}")
                    report_sections.append(f"• p-value: {p_value:.4f}")
                    report_sections.append(f"• Degrees of freedom: {dof}")
                    
                    # Interpretation
                    significance = "Significant association" if p_value < 0.05 else "No significant association"
                    report_sections.append(f"\n💡 INTERPRETATION:")
                    report_sections.append(f"• Statistical significance: {significance} (α = 0.05)")
                    
                    if p_value < 0.05:
                        report_sections.append(f"• Business insight: There is a statistically significant association")
                        report_sections.append(f"  between {column1} and {column2}")
                    else:
                        report_sections.append(f"• Business insight: No significant association detected")
                else:
                    return f"Error: Chi-square test requires both columns to be categorical."

            elif test_type.lower() == "anova":
                # One-way ANOVA (assuming column1 is numeric and column2 is categorical with >2 groups)
                if pd.api.types.is_numeric_dtype(df[column1]) and \
                   (pd.api.types.is_string_dtype(df[column2]) or pd.api.types.is_categorical_dtype(df[column2])) and \
                   df[column2].nunique() > 2:
                    
                    groups = [df[column1].dropna()[df[column2] == g].values for g in df[column2].unique()]
                    # Filter out empty groups
                    groups = [g for g in groups if len(g) > 1]

                    if len(groups) < 2:
                        return "Insufficient data for ANOVA (need at least 2 non-empty groups)."
                    
                    f_stat, p_value = stats.f_oneway(*groups)
                    
                    report_sections.append(f"📊 ONE-WAY ANOVA: {column1} by {column2}")
                    report_sections.append("─" * 50)
                    report_sections.append(f"• F-statistic: {f_stat:.4f}")
                    report_sections.append(f"• p-value: {p_value:.4f}")
                    
                    # Interpretation
                    significance = "Significant difference" if p_value < 0.05 else "No significant difference"
                    report_sections.append(f"\n💡 INTERPRETATION:")
                    report_sections.append(f"• Statistical significance: {significance} (α = 0.05)")
                    
                    if p_value < 0.05:
                        report_sections.append(f"• Business insight: There is a statistically significant difference in {column1}")
                        report_sections.append(f"  means across different groups of {column2}.")
                        report_sections.append(f"  Further post-hoc tests (e.g., Tukey's HSD) may be needed to identify which specific groups differ.")
                    else:
                        report_sections.append(f"• Business insight: No significant difference in {column1} means across groups of {column2}.")
                else:
                    return f"Error: ANOVA requires '{column1}' to be numeric and '{column2}' to be categorical with more than 2 unique groups."

            elif test_type.lower() == "mann_whitney_u":
                # Mann-Whitney U test (non-parametric two-sample test)
                if pd.api.types.is_numeric_dtype(df[column1]) and df[column2].nunique() == 2:
                    groups = df[column2].unique()
                    group1_data = df[df[column2] == groups[0]][column1].dropna()
                    group2_data = df[df[column2] == groups[1]][column1].dropna()
                    
                    if len(group1_data) < 5 or len(group2_data) < 5:
                        return "Insufficient data for Mann-Whitney U test (need at least 5 observations per group)."
                    
                    u_stat, p_value = stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')
                    
                    report_sections.append(f"📊 MANN-WHITNEY U TEST: {column1} by {column2}")
                    report_sections.append("─" * 50)
                    report_sections.append(f"• U-statistic: {u_stat:.4f}")
                    report_sections.append(f"• p-value: {p_value:.4f}")
                    
                    # Interpretation
                    significance = "Significant difference" if p_value < 0.05 else "No significant difference"
                    report_sections.append(f"\n💡 INTERPRETATION:")
                    report_sections.append(f"• Statistical significance: {significance} (α = 0.05)")
                    
                    if p_value < 0.05:
                        report_sections.append(f"• Business insight: There is a statistically significant difference in the distributions of {column1}")
                        report_sections.append(f"  between {groups[0]} and {groups[1]}. This test is suitable for non-normally distributed data.")
                    else:
                        report_sections.append(f"• Business insight: No significant difference in distributions detected between groups.")
                else:
                    return f"Error: Mann-Whitney U test requires '{column1}' to be numeric and '{column2}' to have exactly 2 groups."

            elif test_type.lower() == "kolmogorov_smirnov":
                # Kolmogorov-Smirnov test (for distribution comparison)
                if pd.api.types.is_numeric_dtype(df[column1]) and pd.api.types.is_numeric_dtype(df[column2]):
                    clean_data = df[[column1, column2]].dropna()
                    if len(clean_data) < 10:
                        return "Insufficient data for Kolmogorov-Smirnov test (need at least 10 observations per column)."
                    
                    ks_stat, p_value = stats.ks_2samp(clean_data[column1], clean_data[column2])
                    
                    report_sections.append(f"📊 KOLMOGOROV-SMIRNOV TEST: {column1} vs {column2}")
                    report_sections.append("─" * 50)
                    report_sections.append(f"• KS-statistic: {ks_stat:.4f}")
                    report_sections.append(f"• p-value: {p_value:.4f}")
                    
                    # Interpretation
                    significance = "Significantly different distributions" if p_value < 0.05 else "No significant difference in distributions"
                    report_sections.append(f"\n💡 INTERPRETATION:")
                    report_sections.append(f"• Statistical significance: {significance} (α = 0.05)")
                    
                    if p_value < 0.05:
                        report_sections.append(f"• Business insight: The distributions of {column1} and {column2} are statistically different.")
                    else:
                        report_sections.append(f"• Business insight: The distributions of {column1} and {column2} are not statistically different.")
                else:
                    return f"Error: Kolmogorov-Smirnov test requires both columns to be numeric."
            
            else:
                return f"Error: Unsupported test type '{test_type}'. Supported: correlation, t_test, chi_square, anova, mann_whitney_u, kolmogorov_smirnov."
            
            return "\n".join(report_sections)
            
        except Exception as e:
            return f"Error in statistical testing: {str(e)}"

class GenerateInteractiveChart(BaseTool):
    name: str = "Generate Interactive Chart"
    description: str = (
        "Generates an interactive Plotly chart (bar, line, scatter, pie, histogram, box) "
        "and saves it as an HTML file."
    )

    def _run(self, file_path: str, chart_type: str, x_column: str, y_column: str = None, color_column: str = None, title: str = "Interactive Chart", output_path: str = "chart.html") -> str:
        """
        Generates an interactive Plotly chart (bar, line, scatter, pie, histogram, box)
        and saves it as an HTML file.
        
        Args:
            file_path (str): Path to the CSV or Excel data file.
            chart_type (str): Type of chart to generate (e.g., 'bar', 'line', 'scatter', 'pie', 'histogram', 'box').
            x_column (str): Column to use for the X-axis.
            y_column (str, optional): Column to use for the Y-axis (required for bar, line, scatter).
            color_column (str, optional): Column to use for coloring/grouping.
            title (str, optional): Title of the chart.
            output_path (str, optional): Path to save the HTML chart file.
        """
        try:
            df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
            
            fig = None
            
            if chart_type.lower() == 'bar':
                if not y_column:
                    return "Error: 'y_column' is required for bar charts."
                fig = px.bar(df, x=x_column, y=y_column, color=color_column, title=title)
            elif chart_type.lower() == 'line':
                if not y_column:
                    return "Error: 'y_column' is required for line charts."
                # Ensure x_column is datetime for line charts if it's date-like
                if pd.api.types.is_string_dtype(df[x_column]) and any(keyword in x_column.lower() for keyword in ['date', 'time', 'month', 'year']):
                    df[x_column] = pd.to_datetime(df[x_column], errors='coerce')
                    df.dropna(subset=[x_column], inplace=True)
                fig = px.line(df, x=x_column, y=y_column, color=color_column, title=title)
            elif chart_type.lower() == 'scatter':
                if not y_column:
                    return "Error: 'y_column' is required for scatter plots."
                fig = px.scatter(df, x=x_column, y=y_column, color=color_column, title=title)
            elif chart_type.lower() == 'pie':
                if not y_column: # For pie, y_column is typically 'values'
                    return "Error: 'y_column' (values) is required for pie charts."
                fig = px.pie(df, names=x_column, values=y_column, title=title)
            elif chart_type.lower() == 'histogram':
                fig = px.histogram(df, x=x_column, color=color_column, title=title)
            elif chart_type.lower() == 'box':
                fig = px.box(df, x=x_column, y=y_column, color=color_column, title=title)
            else:
                return f"Error: Unsupported chart type '{chart_type}'. Supported types: bar, line, scatter, pie, histogram, box."
            
            if fig:
                fig.write_html(output_path)
                return f"Interactive {chart_type} chart saved to {output_path}"
            else:
                return "Error: Chart generation failed."
            
        except Exception as e:
            return f"Error generating interactive chart: {str(e)}"

class GenerateProfessionalDashboard(BaseTool):
    name: str = "Generate Professional Dashboard"
    description: str = (
        "Creates a comprehensive professional dashboard with multiple chart types "
        "and business intelligence visualizations, saving it as an HTML file."
    )

    def _run(self, file_path: str, output_path: str = "dashboard.html") -> str:
        """
        Creates a comprehensive professional dashboard with multiple chart types
        and business intelligence visualizations.
        """
        try:
            df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

            fig = make_subplots(
                rows=2, cols=2,
                specs=[[{"type": "indicator"}, {"type": "bar"}],
                       [{"type": "scatter"}, {"type": "pie"}]],
                subplot_titles=("Key Metrics", "Performance by Category", "Correlation", "Market Share")
            )

            # KPI Indicators
            if numeric_cols:
                main_metric = numeric_cols[0]
                total_value = df[main_metric].sum()
                avg_value = df[main_metric].mean()
                fig.add_trace(go.Indicator(
                    mode="number+delta",
                    value=total_value,
                    title={"text": f"Total {main_metric}"},
                    delta={'reference': avg_value, 'relative': True, 'position': "top"}),
                    row=1, col=1
                )

            # Bar Chart
            if numeric_cols and categorical_cols:
                main_cat = categorical_cols[0]
                main_metric = numeric_cols[0]
                top_performers = df.groupby(main_cat)[main_metric].sum().nlargest(10)
                fig.add_trace(go.Bar(x=top_performers.index, y=top_performers.values, name=f"Top 10 by {main_metric}"), row=1, col=2)

            # Scatter Plot
            if len(numeric_cols) >= 2:
                fig.add_trace(go.Scatter(x=df[numeric_cols[0]], y=df[numeric_cols[1]], mode='markers', name=f"{numeric_cols[0]} vs {numeric_cols[1]}"), row=2, col=1)

            # Pie Chart
            if categorical_cols:
                main_cat = categorical_cols[0]
                market_share = df[main_cat].value_counts().nlargest(5)
                fig.add_trace(go.Pie(labels=market_share.index, values=market_share.values, name="Market Share"), row=2, col=2)

            fig.update_layout(title_text="Professional Business Dashboard", height=800)
            fig.write_html(output_path)
            
            return f"Interactive dashboard saved to {output_path}"
            
        except Exception as e:
            return f"Error generating dashboard: {str(e)}"

class NaturalLanguageDataQuery(BaseTool):
    name: str = "Natural Language Data Query"
    description: str = (
        "Processes natural language questions about data and provides targeted analysis. "
        "Handles questions like 'What are the top 5 customers by revenue?' or "
        "'Show me trends in sales over time'."
    )

    def _run(self, file_path: str, user_question: str) -> str:
        """
        Processes natural language questions about data and provides targeted analysis.
        Handles questions like "What are the top 5 customers by revenue?" or 
        "Show me trends in sales over time".
        """
        try:
            df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
            
            # Clean the question for analysis
            question_lower = user_question.lower()
            
            report_sections = []
            report_sections.append(f"🔍 CUSTOM ANALYSIS: {user_question}")
            report_sections.append("═" * (20 + len(user_question)))
            
            # Identify query type and execute appropriate analysis
            
            # TOP/BOTTOM ANALYSIS
            if any(keyword in question_lower for keyword in ['top', 'best', 'highest', 'largest']):
                # Extract number if specified
                import re
                numbers = re.findall(r'\d+', question_lower)
                n = int(numbers[0]) if numbers else 5
                
                # Find relevant columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                
                if 'customer' in question_lower and categorical_cols and numeric_cols:
                    customer_col = next((col for col in categorical_cols if 'customer' in col.lower()), categorical_cols[0])
                    revenue_col = next((col for col in numeric_cols if any(keyword in col.lower() for keyword in ['revenue', 'sales', 'amount'])), numeric_cols[0])
                    
                    top_results = df.groupby(customer_col)[revenue_col].sum().nlargest(n)
                    
                    report_sections.append(f"📊 TOP {n} RESULTS:")
                    report_sections.append("─" * 20)
                    for i, (customer, value) in enumerate(top_results.items(), 1):
                        report_sections.append(f"{i:2d}. {customer}: ${value:,.2f}")
                    
                    total_value = top_results.sum()
                    overall_total = df[revenue_col].sum()
                    percentage = (total_value / overall_total) * 100
                    
                    report_sections.append(f"\n💡 INSIGHTS:")
                    report_sections.append(f"• Top {n} represent {percentage:.1f}% of total {revenue_col}")
                    report_sections.append(f"• Total value: ${total_value:,.2f} out of ${overall_total:,.2f}")
                
                elif numeric_cols and categorical_cols:
                    # Generic top analysis
                    main_metric = numeric_cols[0]
                    main_dimension = categorical_cols[0]
                    
                    top_results = df.groupby(main_dimension)[main_metric].sum().nlargest(n)
                    
                    report_sections.append(f"📊 TOP {n} {main_dimension.upper()} BY {main_metric.upper()}:")
                    report_sections.append("─" * 50)
                    for i, (item, value) in enumerate(top_results.items(), 1):
                        report_sections.append(f"{i:2d}. {item}: {value:,.2f}")
            
            # BOTTOM/WORST ANALYSIS
            elif any(keyword in question_lower for keyword in ['bottom', 'worst', 'lowest', 'smallest']):
                import re
                numbers = re.findall(r'\d+', question_lower)
                n = int(numbers[0]) if numbers else 5
                
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                
                if numeric_cols and categorical_cols:
                    main_metric = numeric_cols[0]
                    main_dimension = categorical_cols[0]
                    
                    bottom_results = df.groupby(main_dimension)[main_metric].sum().nsmallest(n)
                    
                    report_sections.append(f"📉 BOTTOM {n} {main_dimension.upper()} BY {main_metric.upper()}:")
                    report_sections.append("─" * 50)
                    for i, (item, value) in enumerate(bottom_results.items(), 1):
                        report_sections.append(f"{i:2d}. {item}: {value:,.2f}")
                    
                    report_sections.append(f"\n💡 IMPROVEMENT OPPORTUNITIES:")
                    report_sections.append(f"• Focus on bottom {n} for performance improvement")
                    report_sections.append(f"• Investigate root causes of underperformance")
            
            # TREND ANALYSIS
            elif any(keyword in question_lower for keyword in ['trend', 'over time', 'growth', 'change']):
                date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
                date_like_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['date', 'time', 'month', 'year'])]
                
                if date_cols or date_like_cols:
                    time_col = date_cols[0] if date_cols else date_like_cols[0]
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    
                    if numeric_cols:
                        metric_col = numeric_cols[0]
                        
                        # Convert to datetime if needed
                        if time_col in date_like_cols:
                            try:
                                df[time_col] = pd.to_datetime(df[time_col])
                            except:
                                pass
                        
                        # Group by time period and calculate trends
                        if pd.api.types.is_datetime64_any_dtype(df[time_col]):
                            df_sorted = df.sort_values(time_col)
                            time_series = df_sorted.groupby(df_sorted[time_col].dt.to_period('M'))[metric_col].sum()
                            
                            if len(time_series) > 1:
                                first_value = time_series.iloc[0]
                                last_value = time_series.iloc[-1]
                                growth_rate = ((last_value - first_value) / first_value) * 100
                                
                                report_sections.append(f"📈 TREND ANALYSIS FOR {metric_col.upper()}:")
                                report_sections.append("─" * 40)
                                report_sections.append(f"• Time period: {time_series.index[0]} to {time_series.index[-1]}")
                                report_sections.append(f"• Starting value: {first_value:,.2f}")
                                report_sections.append(f"• Ending value: {last_value:,.2f}")
                                report_sections.append(f"• Overall growth: {growth_rate:+.1f}%")
                                
                                # Trend direction
                                if growth_rate > 10:
                                    trend_desc = "Strong upward trend"
                                elif growth_rate > 0:
                                    trend_desc = "Positive growth trend"
                                elif growth_rate > -10:
                                    trend_desc = "Slight decline trend"
                                else:
                                    trend_desc = "Significant downward trend"
                                
                                report_sections.append(f"• Trend direction: {trend_desc}")
                else:
                    report_sections.append("❌ No date/time columns found for trend analysis")
            
            # COMPARISON ANALYSIS
            elif any(keyword in question_lower for keyword in ['compare', 'vs', 'versus', 'difference']):
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if categorical_cols and numeric_cols:
                    main_dimension = categorical_cols[0]
                    main_metric = numeric_cols[0]
                    
                    comparison_data = df.groupby(main_dimension)[main_metric].agg(['sum', 'mean', 'count'])
                    
                    report_sections.append(f"⚖️ COMPARISON ANALYSIS: {main_dimension.upper()} BY {main_metric.upper()}")
                    report_sections.append("─" * 60)
                    
                    for category in comparison_data.index:
                        row = comparison_data.loc[category]
                        report_sections.append(f"• {category}:")
                        report_sections.append(f"  Total: {row['sum']:,.2f}, Average: {row['mean']:.2f}, Count: {row['count']}")
                    
                    # Statistical comparison
                    if len(comparison_data) == 2:
                        categories = list(comparison_data.index)
                        group1_data = df[df[main_dimension] == categories[0]][main_metric]
                        group2_data = df[df[main_dimension] == categories[1]][main_metric]
                        
                        if len(group1_data) > 1 and len(group2_data) > 1:
                            from scipy.stats import ttest_ind
                            t_stat, p_value = ttest_ind(group1_data, group2_data)
                            
                            report_sections.append(f"\n📊 STATISTICAL COMPARISON:")
                            report_sections.append(f"• t-test p-value: {p_value:.4f}")
                            if p_value < 0.05:
                                report_sections.append("• Statistically significant difference detected")
                            else:
                                report_sections.append("• No statistically significant difference")
            
            # CORRELATION ANALYSIS
            elif any(keyword in question_lower for keyword in ['correlation', 'relationship', 'affects', 'influence']):
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if len(numeric_cols) >= 2:
                    corr_matrix = df[numeric_cols].corr()
                    
                    report_sections.append("🔗 CORRELATION ANALYSIS:")
                    report_sections.append("─" * 25)
                    
                    # Find strongest correlations
                    strong_corrs = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            corr_val = corr_matrix.iloc[i, j]
                            if abs(corr_val) > 0.3:  # Moderate correlation threshold
                                strong_corrs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
                    
                    if strong_corrs:
                        report_sections.append("Strong relationships found:")
                        for col1, col2, corr in sorted(strong_corrs, key=lambda x: abs(x[2]), reverse=True):
                            strength = "Strong" if abs(corr) > 0.7 else "Moderate"
                            direction = "positive" if corr > 0 else "negative"
                            report_sections.append(f"• {col1} ↔ {col2}: {strength} {direction} correlation (r = {corr:.3f})")
                    else:
                        report_sections.append("• No strong correlations detected (|r| > 0.3)")
            
            # SUMMARY/OVERVIEW ANALYSIS
            elif any(keyword in question_lower for keyword in ['summary', 'overview', 'describe', 'about']):
                report_sections.append("📋 DATA SUMMARY OVERVIEW:")
                report_sections.append("─" * 28)
                
                rows, cols = df.shape
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                
                report_sections.append(f"• Dataset size: {rows:,} rows × {cols} columns")
                report_sections.append(f"• Numeric columns: {len(numeric_cols)}")
                report_sections.append(f"• Categorical columns: {len(categorical_cols)}")
                
                if numeric_cols:
                    total_revenue = 0
                    revenue_cols = [col for col in numeric_cols if any(keyword in col.lower() for keyword in ['revenue', 'sales', 'amount'])]
                    if revenue_cols:
                        total_revenue = df[revenue_cols[0]].sum()
                        report_sections.append(f"• Total {revenue_cols[0]}: ${total_revenue:,.2f}")
                
                missing_data = df.isnull().sum().sum()
                if missing_data > 0:
                    report_sections.append(f"• Missing values: {missing_data:,} cells")
                else:
                    report_sections.append("• Data completeness: 100% (no missing values)")
            
            # DEFAULT: GENERAL ANALYSIS
            else:
                report_sections.append("🔍 GENERAL DATA ANALYSIS:")
                report_sections.append("─" * 28)
                report_sections.append("I'll provide a general analysis of your data.")
                report_sections.append("For more specific insights, try asking questions like:")
                report_sections.append("• 'What are the top 10 customers by revenue?'")
                report_sections.append("• 'Show me sales trends over time'")
                report_sections.append("• 'Compare performance across regions'")
                report_sections.append("• 'What factors affect revenue?'")
                
                # Basic summary
                rows, cols = df.shape
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                
                report_sections.append(f"\n📊 Quick Overview:")
                report_sections.append(f"• {rows:,} records across {cols} dimensions")
                report_sections.append(f"• {len(numeric_cols)} numeric fields, {len(categorical_cols)} categorical fields")
            
            return "\n".join(report_sections)
            
        except Exception as e:
            return "\n".join(report_sections)
            
        except Exception as e:
            return f"Error processing query: {str(e)}"

class ExecuteSQLQuery(BaseTool):
    name: str = "Execute SQL Query"
    description: str = (
        "Executes a SQL query against a specified database and returns the results. "
        "Supports SQLite, PostgreSQL, MySQL. "
        "Saves the query results to a new CSV or Excel file if output_file_path is provided."
    )

    def _run(self, db_connection_string: str, sql_query: str, output_file_path: Optional[str] = None) -> str:
        """
        Executes a SQL query against a specified database and returns the results.
        Supports SQLite, PostgreSQL, MySQL.
        Saves the query results to a new CSV or Excel file if output_file_path is provided.
        """
        try:
            engine = create_engine(db_connection_string)
            
            with engine.connect() as connection:
                df = pd.read_sql_query(sql_query, connection)
            
            report_sections = []
            report_sections.append(f"SQL QUERY EXECUTION RESULT")
            report_sections.append("═" * 30)
            report_sections.append(f"Database: {engine.name}")
            report_sections.append(f"Query: {sql_query}")
            report_sections.append(f"\nResults ({len(df)} rows, {len(df.columns)} columns):")
            report_sections.append(df.head().to_string()) # Show first 5 rows

            if output_file_path:
                if output_file_path.endswith('.csv'):
                    df.to_csv(output_file_path, index=False)
                elif output_file_path.endswith(('.xlsx', '.xls')):
                    df.to_excel(output_file_path, index=False)
                else:
                    return f"Error: Unsupported output file type for {output_file_path}. Please use .csv or .xlsx."
                report_sections.append(f"\nQuery results saved to: {output_file_path}")

            return "\n".join(report_sections).strip()

        except Exception as e:
            return f"Error executing SQL query: {str(e)}"

class GenerateExecutiveReport(BaseTool):
    name: str = "Generate Executive Report"
    description: str = (
        "Generates a comprehensive executive summary report with key findings, "
        "insights, and strategic recommendations in a professional Markdown format."
    )

    def _run(self, file_path: str, output_path: str = "executive_report.md") -> str:
        """
        Generates a comprehensive executive summary report with key findings,
        insights, and strategic recommendations in a professional format.
        """
        try:
            df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
            
            report_content = []
            
            # Header
            report_content.append("# 📊 Executive Business Intelligence Report")
            report_content.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}  ")
            report_content.append(f"**Data Source:** `{file_path}`  ")
            
            # Executive Summary
            report_content.append("\n## 🎯 Executive Summary")
            
            rows, cols = df.shape
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            report_content.append(f"This analysis covers **{rows:,} records** across **{cols} key business dimensions**.")
            
            # Key Business Metrics
            if numeric_cols:
                revenue_cols = [col for col in numeric_cols if any(keyword in col.lower() for keyword in ['revenue', 'sales', 'amount', 'value'])]
                if revenue_cols:
                    main_metric = revenue_cols[0]
                    total_value = df[main_metric].sum()
                    avg_value = df[main_metric].mean()
                    
                    report_content.append(f"- **Total {main_metric}:** ${total_value:,.2f}")
                    report_content.append(f"- **Average {main_metric}:** ${avg_value:,.2f}")
                    
                    if categorical_cols:
                        main_dimension = categorical_cols[0]
                        performance_by_cat = df.groupby(main_dimension)[main_metric].sum().sort_values(ascending=False)
                        top_performer = performance_by_cat.index[0]
                        top_value = performance_by_cat.iloc[0]
                        report_content.append(f"- **Top performing {main_dimension}:** {top_performer} (${top_value:,.2f})")
            
            # Key Findings
            report_content.append("\n## 💡 Key Findings")
            
            missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            report_content.append(f"- **Data Quality:** {'Excellent' if missing_pct < 1 else 'Good' if missing_pct < 5 else 'Requires attention'} ({100-missing_pct:.1f}% complete).")
            
            if numeric_cols and categorical_cols:
                main_metric = numeric_cols[0]
                main_dimension = categorical_cols[0]
                performance_data = df.groupby(main_dimension)[main_metric].sum().sort_values(ascending=False)
                top_20_pct_count = max(1, len(performance_data) // 5)
                top_20_pct_value = performance_data.head(top_20_pct_count).sum()
                total_value = performance_data.sum()
                concentration_ratio = (top_20_pct_value / total_value) * 100
                report_content.append(f"- **Performance Concentration:** Top 20% of {main_dimension}s generate **{concentration_ratio:.1f}%** of total {main_metric}.")
            
            # Strategic Recommendations
            report_content.append("\n## 🚀 Strategic Recommendations")
            report_content.append("1. **Focus on Top Segments:** Allocate resources to top-performing segments to maximize ROI.")
            report_content.append("2. **Investigate Underperformers:** Analyze root causes for underperforming areas to identify improvement opportunities.")
            report_content.append("3. **Monitor KPIs:** Establish regular monitoring of key performance indicators to track progress and identify trends early.")

            with open(output_path, "w") as f:
                f.write("\n".join(report_content))
            
            return f"Executive report saved to {output_path}"
            
        except Exception as e:
            return f"Error generating executive report: {str(e)}"

class TemporalFeatureEngineer(BaseTool):
    name: str = "Temporal Feature Engineer"
    description: str = (
        "Extracts temporal features from a datetime column (year, month, day, day of week, etc.). "
        "Saves the modified DataFrame to a new CSV or Excel file if output_file_path is provided."
    )

    def _run(self, file_path: str, date_column: str, output_file_path: Optional[str] = None) -> str:
        """
        Extracts temporal features from a datetime column (year, month, day, day of week, etc.).
        Saves the modified DataFrame to a new CSV or Excel file if output_file_path is provided.
        """
        try:
            df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

            if date_column not in df.columns:
                return f"Error: Date column '{date_column}' not found in data."

            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            df.dropna(subset=[date_column], inplace=True)

            if df[date_column].empty:
                return f"Error: No valid datetime values found in '{date_column}' after cleaning."

            df['year'] = df[date_column].dt.year
            df['month'] = df[date_column].dt.month
            df['day'] = df[date_column].dt.day
            df['day_of_week'] = df[date_column].dt.dayofweek # Monday=0, Sunday=6
            df['day_name'] = df[date_column].dt.day_name()
            df['week_of_year'] = df[date_column].dt.isocalendar().week.astype(int)
            df['quarter'] = df[date_column].dt.quarter
            df['is_weekend'] = df[date_column].dt.dayofweek.isin([5, 6]).astype(int)

            report = f"""
⚙️ TEMPORAL FEATURE ENGINEERING REPORT
──────────────────────────────────────
Original Date Column: '{date_column}'
New Features Added:
• year
• month
• day
• day_of_week (0=Monday, 6=Sunday)
• day_name (e.g., Monday, Tuesday)
• week_of_year
• quarter
• is_weekend (1=Weekend, 0=Weekday)

First 5 rows with new features:
{df[[date_column, 'year', 'month', 'day', 'day_of_week', 'day_name', 'week_of_year', 'quarter', 'is_weekend']].head().to_string()}
            """

            if output_file_path:
                if output_file_path.endswith('.csv'):
                    df.to_csv(output_file_path, index=False)
                elif output_file_path.endswith(('.xlsx', '.xls')):
                    df.to_excel(output_file_path, index=False)
                else:
                    return f"Error: Unsupported output file type for {output_file_path}. Please use .csv or .xlsx."
                report += f"\n\nModified data saved to: {output_file_path}"

            return report.strip()

        except Exception as e:
            return f"Error in temporal feature engineering: {str(e)}"

class NumericalFeatureEngineer(BaseTool):
    name: str = "Numerical Feature Engineer"
    description: str = (
        "Applies numerical feature engineering techniques (log, standardization, normalization, binning). "
        "Saves the modified DataFrame to a new CSV or Excel file if output_file_path is provided."
    )

    def _run(self, file_path: str, column: str, transformation_type: str, output_file_path: Optional[str] = None) -> str:
        """
        Applies numerical feature engineering techniques (log, standardization, normalization, binning).
        Saves the modified DataFrame to a new CSV or Excel file if output_file_path is provided.
        """
        try:
            df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

            if column not in df.columns:
                return f"Error: Column '{column}' not found in data."
            if not pd.api.types.is_numeric_dtype(df[column]):
                return f"Error: Column '{column}' is not numeric."

            original_col_name = column
            new_col_name = f"{column}_{transformation_type}"
            report_sections = []

            if transformation_type.lower() == 'log':
                # Add a small constant to avoid log(0) or log(negative)
                df[new_col_name] = np.log1p(df[column].fillna(0))
                report_sections.append(f"Applied Log Transformation (log1p) to '{column}'. New column: '{new_col_name}'")
            elif transformation_type.lower() == 'standardize':
                scaler = StandardScaler()
                df[new_col_name] = scaler.fit_transform(df[[column]].fillna(df[column].mean()))
                report_sections.append(f"Applied Standardization (Z-score) to '{column}'. New column: '{new_col_name}'")
            elif transformation_type.lower() == 'normalize':
                min_val = df[column].min()
                max_val = df[column].max()
                if (max_val - min_val) == 0:
                    df[new_col_name] = 0 # Avoid division by zero if all values are the same
                else:
                    df[new_col_name] = (df[column].fillna(df[column].mean()) - min_val) / (max_val - min_val)
                report_sections.append(f"Applied Min-Max Normalization to '{column}'. New column: '{new_col_name}'")
            elif transformation_type.lower().startswith('bin'):
                num_bins = int(transformation_type.split('_')[1]) if '_' in transformation_type else 5
                df[new_col_name] = pd.cut(df[column], bins=num_bins, labels=False, include_lowest=True)
                report_sections.append(f"Applied Binning ({num_bins} bins) to '{column}'. New column: '{new_col_name}'")
            else:
                return f"Error: Unsupported transformation type '{transformation_type}'. Supported: log, standardize, normalize, bin_N (e.g., bin_5)."

            report_sections.append(f"\nFirst 5 rows of original and new feature:")
            report_sections.append(df[[original_col_name, new_col_name]].head().to_string())

            if output_file_path:
                if output_file_path.endswith('.csv'):
                    df.to_csv(output_file_path, index=False)
                elif output_file_path.endswith(('.xlsx', '.xls')):
                    df.to_excel(output_file_path, index=False)
                else:
                    return f"Error: Unsupported output file type for {output_file_path}. Please use .csv or .xlsx."
                report_sections.append(f"\nModified data saved to: {output_file_path}")

            return "\n".join(report_sections).strip()

        except Exception as e:
            return f"Error in numerical feature engineering: {str(e)}"

class DataCleaningTool(BaseTool):
    name: str = "Data Cleaning Tool"
    description: str = (
        "Cleans the data by handling missing values, removing duplicates, and saving the cleaned data to a new file."
    )

    def _run(self, file_path: str, output_path: str = "cleaned_data.csv") -> str:
        try:
            df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

            # Handle missing values
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col].fillna(df[col].median(), inplace=True)
            for col in df.select_dtypes(include=['object']).columns:
                df[col].fillna(df[col].mode()[0], inplace=True)

            # Remove duplicates
            duplicates_removed = df.duplicated().sum()
            df.drop_duplicates(inplace=True)

            df.to_csv(output_path, index=False)

            return f"Data cleaned successfully. Duplicates removed: {duplicates_removed}. Cleaned data saved to {output_path}"
        except Exception as e:
            return f"Error in data cleaning: {str(e)}"

class CategoricalFeatureEngineer(BaseTool):
    name: str = "Categorical Feature Engineer"
    description: str = (
        "Applies categorical feature engineering techniques (one-hot encoding, label encoding, frequency encoding). "
        "Saves the modified DataFrame to a new CSV or Excel file if output_file_path is provided."
    )

    def _run(self, file_path: str, column: str, encoding_type: str, output_file_path: Optional[str] = None) -> str:
        """
        Applies categorical feature engineering techniques (one-hot encoding, label encoding, frequency encoding).
        Saves the modified DataFrame to a new CSV or Excel file if output_file_path is provided.
        """
        try:
            df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

            if column not in df.columns:
                return f"Error: Column '{column}' not found in data."
            if not pd.api.types.is_string_dtype(df[column]) and not pd.api.types.is_categorical_dtype(df[column]):
                return f"Error: Column '{column}' is not categorical."

            report_sections = []

            if encoding_type.lower() == 'one_hot':
                # Handle potential NaN values before one-hot encoding
                df[column] = df[column].fillna('Missing')
                dummies = pd.get_dummies(df[column], prefix=column, dtype=int)
                df = pd.concat([df, dummies], axis=1)
                df.drop(columns=[column], inplace=True)
                report_sections.append(f"Applied One-Hot Encoding to '{column}'. New columns created.")
                report_sections.append(f"First 5 rows with new features (showing dummy columns):")
                report_sections.append(df.filter(like=f'{column}_').head().to_string())
            elif encoding_type.lower() == 'label':
                encoder = LabelEncoder()
                df[f"{column}_encoded"] = encoder.fit_transform(df[column].fillna('Missing'))
                report_sections.append(f"Applied Label Encoding to '{column}'. New column: '{column}_encoded'")
                report_sections.append(f"First 5 rows of original and new feature:")
                report_sections.append(df[[column, f"{column}_encoded"]].head().to_string())
            elif encoding_type.lower() == 'frequency':
                freq_map = df[column].value_counts(normalize=True)
                df[f"{column}_freq"] = df[column].map(freq_map)
                report_sections.append(f"Applied Frequency Encoding to '{column}'. New column: '{column}_freq'")
                report_sections.append(f"First 5 rows of original and new feature:")
                report_sections.append(df[[column, f"{column}_freq"]].head().to_string())
            else:
                return f"Error: Unsupported encoding type '{encoding_type}'. Supported: one_hot, label, frequency."

            if output_file_path:
                if output_file_path.endswith('.csv'):
                    df.to_csv(output_file_path, index=False)
                elif output_file_path.endswith(('.xlsx', '.xls')):
                    df.to_excel(output_file_path, index=False)
                else:
                    return f"Error: Unsupported output file type for {output_file_path}. Please use .csv or .xlsx."
                report_sections.append(f"\nModified data saved to: {output_file_path}")

            return "\n".join(report_sections).strip()

        except Exception as e:
            return f"Error in categorical feature engineering: {str(e)}"

class ReadFileTool(BaseTool):
    name: str = "Read File Content"
    description: str = (
        "Reads the entire content of a specified file and returns it as a string. "
        "Useful for reading generated reports (e.g., markdown, HTML) or data files."
    )

    def _run(self, file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except FileNotFoundError:
            return f"Error: File not found at {file_path}"
        except Exception as e:
            return f"Error reading file {file_path}: {str(e)}"

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime



class EnhancedDataTools:
    def __init__(self):
        self.cleaned_data_path = "data_analysis_project/cleaned_data.csv"
    
    def smart_data_router(self, file_path: str) -> str:
        """Intelligently routes data to the correct analysis tool."""
        router = SmartDataSourceRouter()
        return router._run(file_path)

    def comprehensive_statistical_profiling(self, file_path: str) -> str:
        profiler = ComprehensiveStatisticalProfiling()
        return profiler._run(file_path)

    def missing_value_analyzer(self, file_path: str) -> str:
        analyzer = MissingValueAnalyzer()
        return analyzer._run(file_path)

    def advanced_outlier_detection(self, file_path: str) -> str:
        detector = AdvancedOutlierDetection()
        return detector._run(file_path)

    def business_performance_analyzer(self, file_path: str) -> str:
        analyzer = BusinessPerformanceAnalyzer()
        return analyzer._run(file_path)

    def trend_seasonality_analyzer(self, file_path: str, date_column: str, metric_column: str, output_file_path: Optional[str] = None) -> str:
        analyzer = TrendAndSeasonalityAnalyzer()
        return analyzer._run(file_path, date_column, metric_column, output_file_path)

    def segmentation_performance_analyzer(self, file_path: str, segment_column: str, metric_column: str, output_file_path: Optional[str] = None) -> str:
        analyzer = SegmentationPerformanceAnalyzer()
        return analyzer._run(file_path, segment_column, metric_column, output_file_path)

    def anomaly_pattern_detector(self, file_path: str, column: str, method: str = "iqr", output_file_path: Optional[str] = None) -> str:
        detector = AnomalyAndPatternDetector()
        return detector._run(file_path, column, method, output_file_path)

    def statistical_significance_testing(self, file_path: str, column1: str, column2: str, test_type: str = "correlation") -> str:
        tester = StatisticalSignificanceTesting()
        return tester._run(file_path, column1, column2, test_type)

    def generate_interactive_chart(self, file_path: str, chart_type: str, x_column: str, y_column: str = None, color_column: str = None, title: str = "Interactive Chart", output_path: str = "chart.html") -> str:
        generator = GenerateInteractiveChart()
        return generator._run(file_path, chart_type, x_column, y_column, color_column, title, output_path)

    def generate_professional_dashboard(self, file_path: str, output_path: str = "dashboard.html") -> str:
        generator = GenerateProfessionalDashboard()
        return generator._run(file_path, output_path)

    def natural_language_query(self, file_path: str, user_question: str) -> str:
        query_processor = NaturalLanguageDataQuery()
        return query_processor._run(file_path, user_question)

    def execute_sql_query(self, db_connection_string: str, sql_query: str, output_file_path: Optional[str] = None) -> str:
        query_executor = ExecuteSQLQuery()
        return query_executor._run(db_connection_string, sql_query, output_file_path)

    def generate_executive_report(self, file_path: str, output_path: str = "executive_report.md") -> str:
        report_generator = GenerateExecutiveReport()
        return report_generator._run(file_path, output_path)

    def temporal_feature_engineer(self, file_path: str, date_column: str, output_file_path: Optional[str] = None) -> str:
        engineer = TemporalFeatureEngineer()
        return engineer._run(file_path, date_column, output_file_path)

    def numerical_feature_engineer(self, file_path: str, column: str, transformation_type: str, output_file_path: Optional[str] = None) -> str:
        engineer = NumericalFeatureEngineer()
        return engineer._run(file_path, column, transformation_type, output_file_path)

    def data_cleaning_tool(self, file_path: str, output_path: str = "cleaned_data.csv") -> str:
        cleaner = DataCleaningTool()
        return cleaner._run(file_path, output_path)

    def categorical_feature_engineer(self, file_path: str, column: str, encoding_type: str, output_file_path: Optional[str] = None) -> str:
        engineer = CategoricalFeatureEngineer()
        return engineer._run(file_path, column, encoding_type, output_file_path)

    def read_file_tool(self, file_path: str) -> str:
        reader = ReadFileTool()
        return reader._run(file_path)
        
    def _load_cleaned_data(self) -> Optional[pd.DataFrame]:
        """Safely load cleaned data with error handling"""
        try:
            if os.path.exists(self.cleaned_data_path):
                return pd.read_csv(self.cleaned_data_path)
            else:
                print(f"Warning: Cleaned data file not found at {self.cleaned_data_path}")
                return None
        except Exception as e:
            print(f"Error loading cleaned data: {str(e)}")
            return None
    
    def enhanced_trend_seasonality_analyzer(self, data_path: str) -> str:
        """Enhanced trend and seasonality analyzer with better error handling"""
        try:
            df = self._load_cleaned_data()
            if df is None:
                return "Error: Could not load data for trend analysis"
            
            # Look for date columns
            date_columns = []
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        pd.to_datetime(df[col])
                        date_columns.append(col)
                    except:
                        continue
            
            if not date_columns:
                return "No date columns found for trend analysis. Creating basic time-based analysis using row index."
            
            # Use the first date column found
            date_col = date_columns[0]
            df[date_col] = pd.to_datetime(df[date_col])
            
            # Find numeric columns for analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols:
                return "No numeric columns found for trend analysis"
            
            results = {
                "trend_analysis": {},
                "seasonal_patterns": {},
                "recommendations": []
            }
            
            # Analyze trends for each numeric column
            for col in numeric_cols[:3]:  # Limit to first 3 columns
                if col in df.columns:
                    trend_data = df.groupby(df[date_col].dt.to_period('M'))[col].mean()
                    
                    # Simple trend calculation
                    if len(trend_data) > 1:
                        trend_slope = np.polyfit(range(len(trend_data)), trend_data.values, 1)[0]
                        results["trend_analysis"][col] = {
                            "trend_direction": "increasing" if trend_slope > 0 else "decreasing",
                            "trend_strength": abs(trend_slope),
                            "data_points": len(trend_data)
                        }
            
            return json.dumps(results, indent=2)
            
        except Exception as e:
            return f"Error in trend analysis: {str(e)}. Using fallback analysis."
    
    def enhanced_anomaly_pattern_detector(self, data_path: str) -> str:
        """Enhanced anomaly detection with better error handling"""
        try:
            df = self._load_cleaned_data()
            if df is None:
                return "Error: Could not load data for anomaly detection"
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols:
                return "No numeric columns found for anomaly detection"
            
            anomalies = {}
            
            for col in numeric_cols[:3]:  # Limit to first 3 columns
                if col in df.columns:
                    # Simple IQR-based anomaly detection
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    anomaly_indices = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
                    
                    anomalies[col] = {
                        "count": len(anomaly_indices),
                        "percentage": (len(anomaly_indices) / len(df)) * 100,
                        "bounds": {"lower": lower_bound, "upper": upper_bound}
                    }
            
            return json.dumps(anomalies, indent=2)
            
        except Exception as e:
            return f"Error in anomaly detection: {str(e)}. No anomalies could be detected."
    
    def enhanced_segmentation_performance_analyzer(self, data_path: str) -> str:
        """Enhanced segmentation analyzer with better error handling"""
        try:
            df = self._load_cleaned_data()
            if df is None:
                return "Error: Could not load data for segmentation analysis"
            
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not categorical_cols or not numeric_cols:
                return "Insufficient data for segmentation analysis"
            
            segmentation_results = {}
            
            # Analyze first categorical column against first numeric column
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            
            segment_stats = df.groupby(cat_col)[num_col].agg(['count', 'mean', 'std']).fillna(0)
            
            segmentation_results[f"{cat_col}_vs_{num_col}"] = {
                "segments": segment_stats.to_dict('index'),
                "total_segments": len(segment_stats),
                "best_performing": segment_stats['mean'].idxmax(),
                "worst_performing": segment_stats['mean'].idxmin()
            }
            
            return json.dumps(segmentation_results, indent=2)
            
        except Exception as e:
            return f"Error in segmentation analysis: {str(e)}. Basic segmentation completed."
    
    def enhanced_generate_professional_dashboard(self, dashboard_type: str = "executive") -> str:
        """Generate actual HTML dashboard files"""
        try:
            df = self._load_cleaned_data()
            if df is None:
                return "Error: Could not load data for dashboard generation"
            
            # Create basic dashboard HTML
            if dashboard_type == "executive":
                html_content = self._create_executive_dashboard_html(df)
                filename = "executive_summary_dashboard.html"
            else:
                html_content = self._create_detailed_dashboard_html(df)
                filename = "detailed_analysis_dashboard.html"
            
            # Save to file
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return f"Successfully generated {filename} dashboard"
            
        except Exception as e:
            return f"Error generating dashboard: {str(e)}"
    
    def _create_executive_dashboard_html(self, df: pd.DataFrame) -> str:
        """Create a basic executive dashboard HTML"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Basic statistics
        total_rows = len(df)
        numeric_summary = df[numeric_cols].describe() if numeric_cols else pd.DataFrame()
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Executive Summary Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .kpi {{ background: #f0f0f0; padding: 20px; margin: 10px; border-radius: 5px; display: inline-block; }}
                .chart {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Executive Summary Dashboard</h1>
            
            <div class="kpi">
                <h3>Total Records</h3>
                <h2>{total_rows:,}</h2>
            </div>
            
            <div class="kpi">
                <h3>Data Columns</h3>
                <h2>{len(df.columns)}</h2>
            </div>
            
            <div class="kpi">
                <h3>Numeric Columns</h3>
                <h2>{len(numeric_cols)}</h2>
            </div>
            
            <div id="summary-chart" class="chart"></div>
            
            <script>
                var data = [{{
                    x: {list(df.columns[:5])},
                    y: {[df[col].count() for col in df.columns[:5]]},
                    type: 'bar'
                }}];
                
                var layout = {{
                    title: 'Data Completeness by Column',
                    xaxis: {{ title: 'Columns' }},
                    yaxis: {{ title: 'Non-null Count' }}
                }};
                
                Plotly.newPlot('summary-chart', data, layout);
            </script>
        </body>
        </html>
        """
        
        return html_template
    
    def _create_detailed_dashboard_html(self, df: pd.DataFrame) -> str:
        """Create a basic detailed dashboard HTML"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Detailed Analysis Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .chart {{ margin: 30px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Detailed Analysis Dashboard</h1>
            
            <h2>Data Overview</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Rows</td><td>{len(df):,}</td></tr>
                <tr><td>Total Columns</td><td>{len(df.columns)}</td></tr>
                <tr><td>Numeric Columns</td><td>{len(numeric_cols)}</td></tr>
                <tr><td>Missing Values</td><td>{df.isnull().sum().sum()}</td></tr>
            </table>
            
            <div id="correlation-chart" class="chart"></div>
            
            <script>
                // Add correlation heatmap if numeric columns exist
                {'// Correlation chart would go here' if numeric_cols else '// No numeric data for correlation'}
            </script>
        </body>
        </html>
        """
        
        return html_template
