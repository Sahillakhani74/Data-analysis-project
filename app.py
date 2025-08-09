import streamlit as st
from main_crewai_orchestrator import DataAnalysisCrew
import os

st.title("🔬 Professional Data Analyst Multi-Agent System")

data_path = None

st.sidebar.header("Data Source")
source_type = st.sidebar.radio("Select data source type", ["File Upload", "Database Connection"])

if source_type == "File Upload":
    uploaded_file = st.file_uploader("Upload your data file (CSV or Excel)", type=["csv", "xlsx"])
    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with open(os.path.join("temp_data.csv"), "wb") as f:
            f.write(uploaded_file.getbuffer())
        data_path = "temp_data.csv"
else:
    db_connection_string = st.text_input("Enter your database connection string")
    if db_connection_string:
        data_path = db_connection_string

if data_path:
    data_crew = DataAnalysisCrew(data_path=data_path)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "comprehensive_analysis" not in st.session_state:
        with st.spinner("Performing comprehensive analysis..."):
            comprehensive_analysis_result = data_crew.run_comprehensive_analysis()
            st.session_state.comprehensive_analysis = comprehensive_analysis_result
            st.session_state.messages.append({"role": "assistant", "content": comprehensive_analysis_result})

    st.write("### Comprehensive Analysis Report & Insights")
    st.markdown(st.session_state.comprehensive_analysis)

    cleaned_data_path = "data_analysis_project/cleaned_data.csv"
    if os.path.exists(cleaned_data_path):
        with open(cleaned_data_path, "rb") as file:
            st.download_button(
                label="Download Cleaned Data (CSV)",
                data=file,
                file_name="cleaned_data.csv",
                mime="text/csv",
            )

    if os.path.exists("executive_summary_dashboard.html"):
        st.write("### Executive Summary Dashboard")
        with open("executive_summary_dashboard.html", "r") as f:
            st.components.v1.html(f.read(), height=800)

    if os.path.exists("detailed_analysis_dashboard.html"):
        st.write("### Detailed Analysis Dashboard")
        with open("detailed_analysis_dashboard.html", "r") as f:
            st.components.v1.html(f.read(), height=800)

    # Only show custom analysis after comprehensive analysis and dashboards are displayed
    if "comprehensive_analysis" in st.session_state:
        st.write("### Custom Analysis")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "chart_path" in message:
                    with open(message["chart_path"], "r") as f:
                        st.components.v1.html(f.read(), height=400)

        if prompt := st.chat_input("Ask a question about your data"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.spinner("Processing your query..."):
                custom_analysis_result = data_crew.run_custom_query(prompt)
                message = {"role": "assistant", "content": custom_analysis_result}
                if "chart.html" in custom_analysis_result:
                    message["chart_path"] = "chart.html"
                st.session_state.messages.append(message)
                with st.chat_message("assistant"):
                    st.markdown(custom_analysis_result)
                    if "chart_path" in message:
                        with open(message["chart_path"], "r") as f:
                            st.components.v1.html(f.read(), height=400)
