from crewai import Task
from pandas import DataFrame

class DataAnalysisTasks:
    def __init__(self, data_path: str = None, user_query: str = None):
        self.data_path = data_path
        self.user_query = user_query

    def route_data(self, agent):
        return Task(
            description=(
                "Analyze the provided data path or inferred database connection to determine the data source type "
                "(CSV, Excel, Parquet, JSON, or SQL Database). "
                "Infer the schema, perform an initial data quality assessment, and generate a concise data summary report. "
                "Based on this assessment, determine the appropriate processing pathway (file-based or database-based)."
                f"\n\nInput Data Path: {self.data_path}"
            ),
            expected_output=(
                "A detailed 'DATA SOURCE SUMMARY' report in markdown format, including:\n"
                "- Source Type (e.g., 'Excel Workbook', 'CSV File', 'SQL Database')\n"
                "- Total Records (number of rows)\n"
                "- Columns (number of fields)\n"
                "- Key Metrics detected (e.g., 'Sales, Revenue, Customer data')\n"
                "- Initial Quality Issues identified (e.g., '3 columns with missing values', 'No major issues')\n"
                "- Analysis Readiness status (e.g., 'Ready for processing')\n"
                "- The determined processing pathway (e.g., 'FILE_SOURCE' or 'DATABASE_SOURCE')."
                "\n\nExample Output:\n"
                "```\n"
                "📊 DATA SOURCE SUMMARY\n"
                "────────────────────────\n"
                "📁 Source Type: Excel Workbook (3 sheets)\n"
                "📈 Total Records: 45,678 rows\n"
                "📋 Columns: 23 fields\n"
                "🎯 Key Metrics: Sales, Revenue, Customer data detected\n"
                "⚠️  Quality Issues: 3 columns with missing values\n"
                "🔍 Analysis Readiness: Ready for processing\n"
                "➡️ Processing Pathway: FILE_SOURCE\n"
                "```"
            ),
            agent=agent,
            tool_params={"file_path": self.data_path}
        )

    def manage_data_quality(self, agent):
        return Task(
            description=(
                "Based on the 'DATA SOURCE SUMMARY' from the preceding task's output, and the raw data, meticulously identify and handle "
                "data quality issues. This includes:\n"
                "- Detecting and analyzing patterns of missing values, and recommending/applying imputation strategies.\n"
                "- Identifying and removing duplicate records.\n"
                "- Performing data validation and cleaning (e.g., correcting data types, standardizing formats).\n"
                "- Detecting and analyzing outliers using statistical methods (Z-score, IQR) and recommending treatment.\n"
                "After performing these steps, save the cleaned data to 'data_analysis_project/cleaned_data.csv'."
                "The output should be a cleaned and validated DataFrame, along with a 'DATA QUALITY REPORT'."
                f"\n\nData File Path: {self.data_path}"
            ),
            expected_output=(
                "A comprehensive 'DATA QUALITY REPORT' in markdown format, detailing:\n"
                "- Missing Value Analysis: Patterns, counts, and imputation actions taken.\n"
                "- Duplicate Analysis: Number of duplicates found and removed.\n"
                "- Data Validation & Cleaning: Specific corrections made (e.g., type conversions, format standardization).\n"
                "- Outlier Analysis: Identified outliers, their impact, and treatment applied.\n"
                "- A confirmation that the data is now clean and ready for EDA.\n"
                "- Confirmation that the cleaned data has been saved to 'data_analysis_project/cleaned_data.csv'."
                "The final output should also implicitly represent the cleaned DataFrame, ready for the next task."
                "\n\nExample Output:\n"
                "```\n"
                "✅ DATA QUALITY REPORT\n"
                "────────────────────────\n"
                "📊 Missing Values: Handled 15% missing values in 'Revenue' column using median imputation.\n"
                "🗑️ Duplicates: 125 duplicate rows identified and removed.\n"
                "🧹 Cleaning: 'Date' column converted to datetime format. 'Product_ID' standardized to uppercase.\n"
                "🚨 Outliers: 5% of 'Sales' values identified as outliers (IQR method), capped at 99th percentile.\n"
                "✨ Data Status: Data is now clean, consistent, and validated for advanced analysis.\n"
                "💾 Cleaned data saved to: data_analysis_project/cleaned_data.csv\n"
                "```"
            ),
            agent=agent,
            tool_params={
                "file_path": self.data_path,
                "output_path": "data_analysis_project/cleaned_data.csv"
            }
        )

    def perform_eda(self, agent):
        return Task(
            description=(
                "Conduct a comprehensive Exploratory Data Analysis (EDA) on the cleaned dataset. "
                "This involves:\n"
                "- Generating descriptive statistics (mean, median, mode, std dev, quartiles).\n"
                "- Analyzing distributions (skewness, kurtosis, normality tests) for numerical variables.\n"
                "- Creating frequency distributions and unique value counts for categorical variables.\n"
                "- Computing Pearson/Spearman correlation matrices and identifying feature correlations with target variables.\n"
                "- Detecting multicollinearity (VIF analysis) and cross-tabulation analysis.\n"
                "The output should be a detailed 'EDA REPORT' summarizing key findings and statistical insights."
                "The preceding task's 'DATA QUALITY REPORT' provides the context for the cleaned data."
            ),
            expected_output=(
                "A comprehensive 'EDA REPORT' in markdown format, structured with clear headings for each analysis component:\n"
                "- **Data Profiling & Summary Statistics**: Key descriptive stats, unique counts, cardinality.\n"
                "- **Distribution Analysis**: Insights on skewness, kurtosis, normality, and categorical frequencies.\n"
                "- **Correlation & Relationship Analysis**: Key correlations, multicollinearity findings, cross-tab insights.\n"
                "- **Outlier Analysis**: Confirmation of outlier treatment and remaining observations.\n"
                "Each section should include specific numerical findings and interpretations, like Power BI's auto-insights."
                "\n\nExample Output:\n"
                "```\n"
                "🔬 ADVANCED EDA REPORT\n"
                "────────────────────────\n"
                "### Data Profiling & Summary Statistics\n"
                "- 'Revenue': Mean=1500, Median=1200, Std Dev=500. Skewness=0.8 (moderately right-skewed).\n"
                "- 'Customer_Segment': 5 unique values. 'Premium' (40%), 'Standard' (30%), 'Basic' (20%), 'New' (10%).\n"
                "### Correlation & Relationship Analysis\n"
                "- Strong positive correlation between 'Marketing_Spend' and 'Sales' (Pearson r=0.75).\n"
                "- 'Product_Category' shows significant association with 'Customer_Segment' (Chi-square p<0.01).\n"
                "### Distribution Analysis\n"
                "- 'Order_Value' distribution is slightly bimodal, suggesting two distinct customer behaviors.\n"
                "### Outlier Analysis\n"
                "- No significant outliers detected after initial treatment, data distributions are robust.\n"
                "```"
            ),
            agent=agent,
            tool_params={"file_path": "data_analysis_project/cleaned_data.csv"}
        )

    def engineer_features(self, agent):
        return Task(
            description=(
                "Based on the 'EDA REPORT' from the preceding task's output, and the cleaned data located at 'data_analysis_project/cleaned_data.csv', "
                "create additional analytical features to enhance the dataset. "
                "This includes:\n"
                "- **Temporal Features**: Extract year, month, day, day of week, weekend/weekday flags, seasonality indicators from datetime columns.\n"
                "- **Numerical Feature Engineering**: Apply log transformations, standardization/normalization, binning, ratio/rate calculations, moving averages.\n"
                "- **Categorical Feature Engineering**: Implement frequency encoding, target encoding, category grouping, dummy variable creation.\n"
                "- **Business Logic Features**: Calculate customer lifetime value, RFM scores, cohort analysis features, performance rankings.\n"
                "The output should be a 'FEATURE ENGINEERING REPORT' detailing the new features created and their rationale."
            ),
            expected_output=(
                "A 'FEATURE ENGINEERING REPORT' in markdown format, detailing the new features created and their analytical purpose:\n"
                "- **Temporal Features**: List of new date-based columns (e.g., 'Order_Year', 'Order_Month', 'Is_Weekend').\n"
                "- **Numerical Features**: Description of transformations (e.g., 'Log_Revenue', 'Normalized_Spend', 'Sales_Growth_Rate').\n"
                "- **Categorical Features**: Explanation of encoding methods (e.g., 'Product_Category_Encoded', 'Customer_Segment_Dummy').\n"
                "- **Business Logic Features**: Details on calculated business metrics (e.g., 'CLTV', 'RFM_Score', 'Customer_Rank').\n"
                "The final output should also implicitly represent the DataFrame with new features, ready for the next task."
                "\n\nExample Output:\n"
                "```\n"
                "⚙️ FEATURE ENGINEERING REPORT\n"
                "───────────────────────────\n"
                "### Temporal Features\n"
                "- 'Order_Year', 'Order_Month', 'Order_DayOfWeek', 'Is_Weekend' extracted from 'Order_Date'.\n"
                "- 'Time_Since_Last_Purchase' calculated for customer behavior analysis.\n"
                "### Numerical Features\n"
                "- 'Log_Sales' created to normalize skewed sales data.\n"
                "- 'Profit_Margin' calculated as (Revenue - Cost) / Revenue.\n"
                "### Categorical Features\n"
                "- 'Region_Encoded' created using frequency encoding for regional performance comparison.\n"
                "### Business Logic Features\n"
                "- 'Customer_Lifetime_Value' (CLTV) estimated based on historical purchases.\n"
                "- 'RFM_Score' calculated to segment customers by Recency, Frequency, and Monetary value.\n"
                "```"
            ),
            agent=agent
        )

    def generate_business_insights(self, agent):
        return Task(
            description=(
                "Leverage the engineered features from the preceding task's output and the cleaned data located at 'data_analysis_project/cleaned_data.csv' to generate actionable business insights. "
                "Focus on:\n"
                "- **Performance Analysis**: Identify top/bottom performers (products, regions, segments), growth/decline trends, benchmarking.\n"
                "- **Trend & Pattern Detection**: Identify seasonal, cyclical, and growth patterns, change points, recurring patterns.\n"
                "- **Segmentation Analysis**: Provide insights on customer, product, geographic, demographic, and behavioral segments.\n"
                "- **Risk & Anomaly Analysis**: Detect statistical anomalies, business rule violations, performance deviations.\n"
                "- **Predictive Indicators**: Identify leading indicators, early warning signals, growth forecasting signals.\n"
                "The output should be a 'BUSINESS INSIGHTS REPORT' with clear, actionable statements, evidence, impact, and recommendations."
            ),
            expected_output=(
                "A 'BUSINESS INSIGHTS REPORT' in markdown format, structured with 'INSIGHT', 'EVIDENCE', 'IMPACT', and 'RECOMMENDATION' sections for each key finding:\n"
                "```\n"
                "💡 BUSINESS INSIGHTS REPORT\n"
                "───────────────────────────\n"
                "### Insight 1: Strong Q4 Sales Growth in North Region\n"
                "💡 INSIGHT: The North region experienced a significant 25% year-over-year sales growth in Q4, outperforming all other regions.\n"
                "📊 EVIDENCE: Sales data shows North region's Q4 revenue increased from $1.2M (last year) to $1.5M (this year), while other regions averaged 5% growth.\n"
                "🎯 IMPACT: This growth contributes significantly to overall company revenue and indicates successful regional strategies.\n"
                "🚀 RECOMMENDATION: Investigate successful marketing campaigns or sales initiatives in the North region for replication in underperforming areas.\n\n"
                "### Insight 2: Decline in Customer Retention for Product X\n"
                "💡 INSIGHT: Customer retention rate for Product X dropped by 10% in the last quarter, indicating potential dissatisfaction or competitive pressure.\n"
                "📊 EVIDENCE: Cohort analysis shows 3-month retention for Product X users decreased from 70% to 60%.\n"
                "🎯 IMPACT: This decline could lead to significant long-term revenue loss if not addressed.\n"
                "🚀 RECOMMENDATION: Conduct a customer satisfaction survey for Product X users and analyze competitor offerings to identify root causes.\n"
                "```"
            ),
            agent=agent
        )

    def perform_statistical_testing(self, agent):
        return Task(
            description=(
                "Conduct rigorous statistical tests on the cleaned data located at 'data_analysis_project/cleaned_data.csv' to validate the business insights generated from the preceding task's output. "
                "This includes:\n"
                "- **Hypothesis Testing**: Perform T-tests, Chi-square tests, ANOVA, Mann-Whitney U tests as appropriate to confirm statistical significance of observed differences or associations.\n"
                "- **Business Significance Testing**: Interpret A/B test results, calculate confidence intervals, effect sizes, and power analysis to quantify practical business impact.\n"
                "- **Correlation & Causation Analysis**: Test significance of correlations, perform partial correlation analysis, and identify confounding variables.\n"
                "The output should be a 'STATISTICAL VALIDATION REPORT' confirming or refuting insights with statistical evidence."
            ),
            expected_output=(
                "A 'STATISTICAL VALIDATION REPORT' in markdown format, providing statistical evidence for the business insights:\n"
                "- For each insight, state the hypothesis tested, the statistical test used, p-value, confidence intervals, and conclusion (e.g., 'Statistically significant at p<0.05').\n"
                "- Clearly distinguish between statistical significance and practical business significance.\n"
                "- Include effect sizes where relevant to quantify the magnitude of observed effects.\n"
                "\n\nExample Output:\n"
                "```\n"
                "📊 STATISTICAL VALIDATION REPORT\n"
                "───────────────────────────────\n"
                "### Insight 1 Validation: North Region Sales Growth\n"
                "- Hypothesis: Mean Q4 sales in North region are significantly higher than other regions.\n"
                "- Test: One-way ANOVA.\n"
                "- Result: F(3, 120) = 8.5, p = 0.0002. (Statistically significant).\n"
                "- Conclusion: The observed sales growth in the North region is statistically significant and not due to random chance. Effect size (Cohen's d) indicates a large practical impact.\n\n"
                "### Insight 2 Validation: Product X Retention Decline\n"
                "- Hypothesis: Customer retention rate for Product X is significantly lower than the average retention rate for similar products.\n"
                "- Test: Two-sample T-test.\n"
                "- Result: t(250) = -3.2, p = 0.0015. (Statistically significant).\n"
                "- Conclusion: The decline in retention for Product X is statistically significant, confirming a genuine issue. Confidence interval for the difference in means is [-0.12, -0.08], indicating a consistent negative impact.\n"
                "```"
            ),
            agent=agent
        )

    def build_dashboards(self, agent):
        return Task(
            description=(
                "Design and build professional, interactive dashboards based on the generated business insights and statistical validations from the preceding task's output, using the cleaned data located at 'data_analysis_project/cleaned_data.csv'. "
                "This task involves creating:\n"
                "- An **Executive Summary Dashboard**: KPI scorecards, high-level performance metrics, key trend visualizations, alerts, business impact summaries.\n"
                "- A **Detailed Analysis Dashboard**: Interactive charts with drill-down, filter panels, data tables, multi-dimensional views, conditional formatting.\n"
                "Ensure the dashboards adhere to Power BI/Tableau style visualization standards, consistent color schemes, clear typography, logical layout, and are mobile-responsive. "
                "Specify the types of charts used for each key insight."
            ),
            expected_output=(
                "A 'PROFESSIONAL DASHBOARD DESIGN' document in markdown format, detailing the structure and content of both dashboards:\n"
                "- **Executive Summary Dashboard**: List of KPIs, chart types (e.g., 'Gauge for Revenue', 'Line chart for Sales Trend'), and key insights displayed.\n"
                "- **Detailed Analysis Dashboard**: Description of interactive features (filters, drill-downs), chart types for specific analyses (e.g., 'Bar chart for Top 10 Products', 'Scatter plot for Marketing Spend vs. Sales'), and data tables.\n"
                "- Confirmation of adherence to design principles (clean, consistent, responsive).\n"
                "This output should be a blueprint for a visual representation, not the actual interactive dashboard."
                "\n\nExample Output:\n"
                "```\n"
                "📈 PROFESSIONAL DASHBOARD DESIGN\n"
                "───────────────────────────────\n"
                "### Executive Summary Dashboard\n"
                "- **KPIs**: Total Revenue (Gauge), Profit Margin (Gauge), Customer Acquisition Cost (Gauge).\n"
                "- **Key Trends**: Overall Sales Trend (Line Chart), Monthly Active Users (Area Chart).\n"
                "- **Alerts**: Low Stock Items (Table with Conditional Formatting).\n"
                "- **Business Impact**: Revenue vs. Target (Bar Chart).\n\n"
                "### Detailed Analysis Dashboard\n"
                "- **Filters**: Date Range, Product Category, Region, Customer Segment.\n"
                "- **Charts**: \n"
                "  - Top 10 Products by Revenue (Horizontal Bar Chart with drill-down to sub-categories).\n"
                "  - Sales Distribution by Region (Choropleth Map).\n"
                "  - Customer Demographics (Stacked Bar Chart for Age Groups by Segment).\n"
                "  - Marketing Spend vs. Conversion Rate (Scatter Plot).\n"
                "- **Interactivity**: All charts support hover tooltips, drill-down on categories, and dynamic filtering.\n"
                "- **Design**: Clean, modern aesthetic with a consistent corporate color palette. Fully mobile-responsive.\n"
                "```"
            ),
            agent=agent
        )

    def generate_executive_summary(self, agent):
        return Task(
            description=(
                "Generate a comprehensive executive summary report based on the cleaned data. "
                "Use the 'Generate Executive Report' tool with 'data_analysis_project/cleaned_data.csv' as input. "
                "Save the output to 'executive_report.md'."
            ),
            expected_output=(
                "A confirmation that the executive report has been successfully generated and saved to 'executive_report.md'."
            ),
            agent=agent,
        )

    def read_dashboard_files(self, agent):
        return Task(
            description=(
                "Read the HTML content of the generated dashboards. "
                "Use the 'Read File Content' tool to read 'executive_summary_dashboard.html' and 'detailed_analysis_dashboard.html'."
            ),
            expected_output=(
                "The HTML content of both 'executive_summary_dashboard.html' and 'detailed_analysis_dashboard.html'."
            ),
            agent=agent,
        )

    def compile_final_report(self, agent):
        return Task(
            description=(
                "Compile all analytical findings into a structured JSON output. "
                "This involves synthesizing a 'Technical Analysis Report' and a 'Business Intelligence Report' in markdown format, "
                "and then constructing a final JSON object with all the components."
            ),
            expected_output=(
                "A JSON object containing the comprehensive analysis results, including:\n"
                "```json\n"
                "{\n"
                "  \"executive_summary_report\": \"[Markdown content of the executive summary report]\",\n"
                "  \"technical_analysis_report\": \"[Markdown content of the technical analysis report]\",\n"
                "  \"business_intelligence_report\": \"[Markdown content of the business intelligence report]\",\n"
                "  \"executive_dashboard_html\": \"[HTML content of executive_summary_dashboard.html]\",\n"
                "  \"detailed_dashboard_html\": \"[HTML content of detailed_analysis_dashboard.html]\"\n"
                "}\n"
                "```"
            ),
            agent=agent,
        )

    def process_custom_query(self, agent):
        return Task(
            description=(
                "Interpret the user's natural language query for custom analysis, using the comprehensive analysis from the previous tasks as context, and the cleaned data located at 'data_analysis_project/cleaned_data.csv'. "
                "Determine the type of analysis needed (e.g., comparison, trend, ranking, filtering, aggregation, correlation, time-based, segmentation, growth, anomaly). "
                "Translate the query into executable data processing steps (e.g., Pandas operations for files, SQL queries for databases). "
                "Execute the analysis, generate targeted visualizations, and provide specific insights and recommendations. "
                "The output must be clearly labeled as 'CUSTOM ANALYSIS' to distinguish it from the initial comprehensive analysis."
                f"\n\nUser's Custom Query: {self.user_query}"
            ),
            expected_output=(
                "A 'CUSTOM ANALYSIS' report in markdown format, clearly labeled and distinct from the initial comprehensive analysis. "
                "It should include:\n"
                "- The user's original question.\n"
                "- The specific findings from the custom analysis.\n"
                "- A description of the targeted visualizations generated (e.g., 'Bar chart showing top 5 products').\n"
                "- Contextual insights and specific recommendations related to the custom query.\n"
                "\n\nExample Output:\n"
                "```\n"
                "📊 CUSTOM ANALYSIS: Top 10 Customers by Revenue\n"
                "──────────────────────────────────────────────\n"
                "❓ User Question: 'Show me top 10 customers by revenue'\n"
                "📈 Specific Findings:\n"
                "- Customer 'Alpha Corp' is the highest revenue generator with $1.2M.\n"
                "- The top 3 customers account for 40% of total revenue.\n"
                "🎨 Targeted Visualization: Horizontal Bar Chart displaying 'Customer Name' vs 'Total Revenue'.\n"
                "💡 Contextual Insights: These top customers are critical for sustained revenue. Understanding their needs and ensuring satisfaction is paramount.\n"
                " Recommendation: Implement a dedicated account management program for the top 10 customers to foster loyalty and identify upselling opportunities.\n"
                "```"
            ),
            agent=agent
        )
