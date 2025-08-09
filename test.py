from main_crewai_orchestrator import DataAnalysisCrew

if __name__ == "__main__":
    # Initialize the crew with the path to your data
    data_crew = DataAnalysisCrew(data_path=r"C:\Users\sahil\Downloads\stock_data.csv")
    
    # Run the comprehensive analysis
    data_crew.run()
