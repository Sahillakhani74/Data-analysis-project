import os
from crewai import Crew, Process
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from .agents.crew_agents import DataAnalysisAgents
from .agents.crew_tasks import DataAnalysisTasks

# Load environment variables from the specific .env file within data_analysis_project
load_dotenv()

class DataAnalysisCrew:
    def __init__(self, data_path: str = r"C:\Users\sahil\Downloads\stock_data.csv"):
        self.data_path = data_path
        self.agents = DataAnalysisAgents()
        self.tasks = DataAnalysisTasks(data_path=data_path)

    def run_comprehensive_analysis(self):
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

        print("#### Starting Comprehensive Data Analysis Crew ####")
        # The kickoff method returns the final output of the last task.
        # The generate_reports_task is now configured to return a JSON string.
        comprehensive_analysis_raw_output = comprehensive_analysis_crew.kickoff()
        print("\n\n#### Comprehensive Data Analysis Complete! ####")
        
        # Parse the JSON string output from the last task
        import json
        try:
            comprehensive_analysis_result = json.loads(comprehensive_analysis_raw_output)
        except json.JSONDecodeError:
            # Fallback if the output is not valid JSON (e.g., if an earlier task failed)
            comprehensive_analysis_result = {
                "executive_summary_report": comprehensive_analysis_raw_output,
                "technical_analysis_report": "N/A",
                "business_intelligence_report": "N/A",
                "executive_dashboard_html": "",
                "detailed_dashboard_html": ""
            }
        
        return comprehensive_analysis_result

    def run_custom_query(self, user_query):
        query_processor = self.agents.natural_language_query_processor()
        self.tasks.user_query = user_query
        process_custom_query_task = self.tasks.process_custom_query(query_processor)

        custom_query_crew = Crew(
            agents=[query_processor],
            tasks=[process_custom_query_task],
            verbose=True,
            process=Process.sequential
        )

        print(f"\n#### Processing Custom Query: '{user_query}' ####")
        custom_analysis_result = custom_query_crew.kickoff()
        print("\n\n#### Custom Analysis Result ####")
        return custom_analysis_result
