from crewai import Agent,LLM
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaLLM as Ollama

from .crew_tools import (
    SmartDataSourceRouter,
    ComprehensiveStatisticalProfiling,
    MissingValueAnalyzer,
    AdvancedOutlierDetection,
    BusinessPerformanceAnalyzer,
    TrendAndSeasonalityAnalyzer,
    SegmentationPerformanceAnalyzer,
    AnomalyAndPatternDetector,
    StatisticalSignificanceTesting,
    GenerateInteractiveChart,
    GenerateProfessionalDashboard,
    NaturalLanguageDataQuery,
    ExecuteSQLQuery,
    GenerateExecutiveReport,
    TemporalFeatureEngineer,
    NumericalFeatureEngineer,
    DataCleaningTool,
    CategoricalFeatureEngineer,
    ReadFileTool
)

load_dotenv()

gemini_api_key = os.getenv("GOOGLE_API_KEY")
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") # Default Ollama URL

# Debugging: Print a masked version of the API key to confirm it's loaded
if gemini_api_key:
    print(f"DEBUG: Loaded Gemini API Key: {gemini_api_key[:5]}...{gemini_api_key[-5:]}")
else:
    print("DEBUG: Gemini API Key not loaded from environment.")

# FIXED: Use CrewAI's LLM wrapper with proper model specification
default_llm = LLM(
    model="gemini/gemini-2.5-flash",
    temperature=0.7,
    api_key=gemini_api_key,
)

# Alternative option for Gemini (if the above doesn't work, try this):
# default_llm = LLM(
#     model="google/gemini-2.0-flash-exp",
#     temperature=0.7,
#     api_key=os.environ.get("GOOGLE_API_KEY"),
# )

# Define a secondary LLM for local models like Llama-3.1 or Mistral
# Note: For Ollama with CrewAI, you might need to use LLM wrapper as well
secondary_llm = LLM(
    model="ollama/llama3.1:8b",
    base_url=ollama_base_url,
)

# Alternative: If you want to keep using LangChain's ChatGoogleGenerativeAI directly:
# langchain_gemini_llm = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash-exp",  
#     temperature=0.7,
#     google_api_key=os.environ.get("GOOGLE_API_KEY"),
# )

class DataAnalysisAgents:
    def __init__(self):
        self.smart_data_router = SmartDataSourceRouter()
        self.comprehensive_statistical_profiling = ComprehensiveStatisticalProfiling()
        self.missing_value_analyzer = MissingValueAnalyzer()
        self.advanced_outlier_detection = AdvancedOutlierDetection()
        self.business_performance_analyzer = BusinessPerformanceAnalyzer()
        self.trend_seasonality_analyzer = TrendAndSeasonalityAnalyzer()
        self.segmentation_performance_analyzer = SegmentationPerformanceAnalyzer()
        self.anomaly_pattern_detector = AnomalyAndPatternDetector()
        self.statistical_significance_testing = StatisticalSignificanceTesting()
        self.generate_interactive_chart = GenerateInteractiveChart()
        self.generate_professional_dashboard = GenerateProfessionalDashboard()
        self.natural_language_query = NaturalLanguageDataQuery()
        self.execute_sql_query = ExecuteSQLQuery()
        self.generate_executive_report = GenerateExecutiveReport()
        self.temporal_feature_engineer = TemporalFeatureEngineer()
        self.numerical_feature_engineer = NumericalFeatureEngineer()
        self.data_cleaning_tool = DataCleaningTool()
        self.categorical_feature_engineer = CategoricalFeatureEngineer()
        self.read_file_tool = ReadFileTool()

    def smart_data_router_agent(self):
        return Agent(
            role='Smart Data Router',
            backstory=(
                "You are the initial gatekeeper and intelligent traffic controller of the data analysis platform. "
                "Your expertise lies in swiftly identifying the nature of incoming data (file or database), "
                "inferring its schema, and performing a rapid initial quality assessment. "
                "You ensure that data is correctly categorized and routed to the most appropriate processing pipeline, "
                "setting the stage for all subsequent analysis with precision and foresight."
            ),
            goal=(
                "Intelligently detect data source types (CSV, Excel, Parquet, JSON, SQL DBs), "
                "infer schemas, assess initial data quality, and route data to the correct processing pipeline."
            ),
            verbose=True,
            allow_delegation=False,
            llm=default_llm, # Using default_llm (Gemini)
            tools=[self.smart_data_router]
        )

    def data_quality_manager(self):
        return Agent(
            role='Data Quality Manager',
            backstory=(
                "You are the meticulous guardian of data integrity. With an eagle eye for detail, "
                "you identify and address every imperfection in the dataset, from missing values and duplicates "
                "to inconsistencies and outliers. Your mission is to transform raw, messy data into a pristine, "
                "reliable foundation, ensuring that all subsequent analyses are built on solid, trustworthy ground."
            ),
            goal=(
                "Ensure data integrity and reliability by meticulously handling missing values, duplicates, "
                "inconsistencies, and outliers, preparing the data for robust analysis."
            ),
            verbose=True,
            allow_delegation=False,
            llm=default_llm, # Using default_llm (Gemini)
            tools=[
                self.missing_value_analyzer,
                self.advanced_outlier_detection,
                self.data_cleaning_tool
            ]
        )

    def advanced_eda_engine(self):
        return Agent(
            role='Advanced EDA Engine',
            backstory=(
                "You are the master of uncovering hidden patterns and statistical truths within data. "
                "Equipped with a comprehensive suite of statistical tools, you delve deep into distributions, "
                "correlations, and summary statistics. Your insights are the bedrock for understanding the data's "
                "underlying structure, revealing critical relationships and potential areas of interest "
                "that drive meaningful business intelligence."
            ),
            goal=(
                "Perform comprehensive Exploratory Data Analysis (EDA), including statistical summaries, "
                "distribution analysis, correlation matrices, and outlier detection, to deeply understand the dataset."
            ),
            verbose=True,
            allow_delegation=False,
            llm=default_llm, # Using default_llm (Gemini) to avoid CUDA memory issues
            tools=[
                self.comprehensive_statistical_profiling
            ]
        )

    def feature_engineering_agent(self):
        return Agent(
            role='Feature Engineering Agent',
            backstory=(
                "You are the creative architect of new analytical dimensions. Your expertise lies in transforming "
                "raw data into rich, insightful features that unlock deeper analytical possibilities. "
                "Whether it's extracting temporal patterns, calculating complex business metrics, or encoding "
                "categorical variables, you enhance the dataset's predictive power and analytical depth, "
                "just like a seasoned data scientist."
            ),
            goal=(
                "Create valuable new features from existing data, including temporal, numerical, and categorical "
                "enhancements, to enrich the dataset for advanced analysis and modeling."
            ),
            verbose=True,
            allow_delegation=False,
            llm=default_llm, # Using default_llm (Gemini)
            tools=[
                self.temporal_feature_engineer,
                self.numerical_feature_engineer,
                self.categorical_feature_engineer
            ]
        )

    def business_insights_generator(self):
        return Agent(
            role='Business Insights Generator',
            backstory=(
                "You are the strategic mind that translates complex data into actionable business intelligence. "
                "Your focus is on identifying key performance indicators, uncovering trends, detecting anomalies, "
                "and segmenting performance. You provide clear, concise, and impactful insights that directly "
                "inform strategic decisions and drive business growth, acting as a senior business analyst."
            ),
            goal=(
                "Generate actionable business insights by analyzing performance, identifying trends, detecting "
                "anomalies, and segmenting data, providing clear recommendations for strategic decision-making."
            ),
            verbose=True,
            allow_delegation=False,
            llm=default_llm, # Using default_llm (Gemini) to avoid CUDA memory issues
            tools=[
                self.business_performance_analyzer,
                self.trend_seasonality_analyzer,
                self.segmentation_performance_analyzer,
                self.anomaly_pattern_detector
            ]
        )

    def statistical_testing_agent(self):
        return Agent(
            role='Statistical Testing and Validation Agent',
            backstory=(
                "You are the rigorous validator of hypotheses and statistical claims. With a deep understanding "
                "of statistical methodologies, you conduct precise tests (t-tests, chi-square, ANOVA) "
                "to validate findings and quantify business significance. Your work ensures that all conclusions "
                "are statistically sound and reliable, providing the confidence needed for data-driven decisions."
            ),
            goal=(
                "Perform rigorous statistical analysis and hypothesis testing to validate insights, "
                "quantify business significance, and ensure the reliability of data-driven conclusions."
            ),
            verbose=True,
            allow_delegation=False,
            llm=default_llm, # Using default_llm (Gemini)
            tools=[
                self.statistical_significance_testing
            ]
        )

    def professional_dashboard_builder(self):
        return Agent(
            role='Professional Dashboard Builder',
            backstory=(
                "You are the visual storyteller, transforming complex data into intuitive, Power BI/Tableau-style "
                "dashboards. Your artistry lies in crafting executive summaries and detailed interactive views "
                "that are both aesthetically pleasing and highly functional. You ensure that key insights are "
                "communicated effectively through compelling visualizations, making data accessible and actionable."
            ),
            goal=(
                "Design and build professional, interactive dashboards (executive summary and detailed analysis) "
                "with Power BI/Tableau style visualizations and mobile-responsive design."
            ),
            verbose=True,
            allow_delegation=False,
            llm=default_llm, # Using default_llm (Gemini)
            tools=[
                self.generate_interactive_chart,
                self.generate_professional_dashboard
            ]
        )

    def report_generation_agent(self):
        return Agent(
            role='Report Generation Agent',
            backstory=(
                "You are the meticulous documentarian, responsible for compiling all analytical findings "
                "into comprehensive, professional reports. Your expertise extends to integrating various "
                "outputs, including markdown reports and HTML dashboards, into a single, cohesive deliverable. "
                "You ensure that every insight, methodology, and recommendation is clearly articulated and "
                "that all supporting visualizations are seamlessly included, ready for presentation or distribution."
            ),
            goal=(
                "Generate comprehensive and professional analytical reports, including executive summaries, "
                "technical analyses, and business intelligence reports. Crucially, you will also read and embed "
                "the HTML content of the generated dashboards (executive_summary_dashboard.html and detailed_analysis_dashboard.html) "
                "and the markdown content of the executive report (executive_report.md) into a final JSON output. "
                "The final output MUST be a JSON string containing all these components."
            ),
            verbose=True,
            allow_delegation=False,
            llm=default_llm, # Using default_llm (Gemini) to avoid CUDA memory issues
            tools=[
                self.generate_executive_report,
                self.read_file_tool
            ]
        )

    def natural_language_query_processor(self):
        return Agent(
            role='Natural Language Query Processor',
            backstory=(
                "You are the intuitive interface between the user and the data. Your unique ability to understand "
                "and translate natural language questions into precise analytical queries (whether for files or databases) "
                "makes complex data accessible to everyone. You empower users to 'ask anything about their data' "
                "and receive targeted, visualized answers, acting as a personal data assistant."
            ),
            goal=(
                "Interpret natural language queries from users, translate them into executable analysis steps, "
                "and coordinate the generation of custom visualizations and insights."
            ),
            verbose=True,
            allow_delegation=False,
            llm=default_llm, # Using default_llm (Gemini) to avoid CUDA memory issues
            tools=[
                self.natural_language_query,
                self.execute_sql_query,
                self.generate_interactive_chart
            ]
        )
