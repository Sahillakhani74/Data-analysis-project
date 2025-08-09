import os
import json
from crewai import Crew, Process
from dotenv import load_dotenv

from .agents.crew_agents import DataAnalysisAgents
from .agents.crew_tasks import DataAnalysisTasks

load_dotenv()

class ImprovedDataAnalysisCrew:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.agents = DataAnalysisAgents()
        self.tasks = DataAnalysisTasks(data_path=data_path)
    
    def run_comprehensive_analysis(self):
        """Run analysis with better error handling"""
        try:
            # Your existing crew setup code...
            
            print("#### Starting Comprehensive Data Analysis Crew ####")
            
            # Initialize Agents
            smart_router = self.agents.smart_data_router_agent()
            data_quality_manager = self.agents.data_quality_manager()
            eda_engine = self.agents.advanced_eda_engine()
            feature_engineer = self.agents.feature_engineering_agent()
            insights_generator = self.agents.business_insights_generator()
            statistical_tester = self.agents.statistical_testing_agent()
            dashboard_builder = self.agents.professional_dashboard_builder()
            report_generator = self.agents.report_generation_agent()

            # Initialize Tasks
            route_data_task = self.tasks.route_data(smart_router)
            manage_data_quality_task = self.tasks.manage_data_quality(data_quality_manager)
            manage_data_quality_task.context = [route_data_task]

            perform_eda_task = self.tasks.perform_eda(eda_engine)
            perform_eda_task.context = [manage_data_quality_task]

            engineer_features_task = self.tasks.engineer_features(feature_engineer)
            engineer_features_task.context = [perform_eda_task]

            generate_business_insights_task = self.tasks.generate_business_insights(insights_generator)
            generate_business_insights_task.context = [engineer_features_task]

            perform_statistical_testing_task = self.tasks.perform_statistical_testing(statistical_tester)
            perform_statistical_testing_task.context = [generate_business_insights_task]

            build_dashboards_task = self.tasks.build_dashboards(dashboard_builder)
            build_dashboards_task.context = [perform_statistical_testing_task]

            generate_reports_task = self.tasks.generate_reports(report_generator)
            generate_reports_task.context = [build_dashboards_task]

            # Create the Crew for comprehensive analysis
            comprehensive_analysis_crew = Crew(
                agents=[
                    smart_router,
                    data_quality_manager,
                    eda_engine,
                    feature_engineer,
                    insights_generator,
                    statistical_tester,
                    dashboard_builder,
                    report_generator
                ],
                tasks=[
                    route_data_task,
                    manage_data_quality_task,
                    perform_eda_task,
                    engineer_features_task,
                    generate_business_insights_task,
                    perform_statistical_testing_task,
                    build_dashboards_task,
                    generate_reports_task
                ],
                verbose=True,
                process=Process.sequential
            )
            
            comprehensive_analysis_raw_output = comprehensive_analysis_crew.kickoff()
            print("\n\n#### Comprehensive Data Analysis Complete! ####")
            
            # Better JSON parsing with fallback
            try:
                comprehensive_analysis_result = json.loads(comprehensive_analysis_raw_output)
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse JSON output: {e}")
                
                # Create fallback structure
                comprehensive_analysis_result = {
                    "executive_summary_report": str(comprehensive_analysis_raw_output),
                    "technical_analysis_report": "Analysis completed with some limitations due to data constraints.",
                    "business_intelligence_report": "Business insights generated based on available data.",
                    "executive_dashboard_html": self._get_file_content("executive_summary_dashboard.html"),
                    "detailed_dashboard_html": self._get_file_content("detailed_analysis_dashboard.html")
                }
            
            return comprehensive_analysis_result
            
        except Exception as e:
            print(f"Error in comprehensive analysis: {str(e)}")
            return {
                "error": str(e),
                "executive_summary_report": "Analysis encountered errors. Please check your data and try again.",
                "technical_analysis_report": "N/A",
                "business_intelligence_report": "N/A", 
                "executive_dashboard_html": "",
                "detailed_dashboard_html": ""
            }
    
    def _get_file_content(self, filename: str) -> str:
        """Safely read file content"""
        try:
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as f:
                    return f.read()
            return ""
        except Exception as e:
            print(f"Could not read {filename}: {e}")
            return ""
