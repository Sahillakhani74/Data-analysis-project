# Data Analysis Project

This project is a multi-agent system for performing data analysis tasks. It consists of a FastAPI backend that provides a RESTful API for data analysis and a Streamlit frontend for a user-friendly web interface.

## Features

- **Dual Interface**: Access the data analysis capabilities through either a RESTful API or a user-friendly web interface.
- **Data Source Flexibility**: Supports both file uploads (CSV and Excel) and direct database connections.
- **Comprehensive Analysis**: Automatically performs a thorough analysis of the provided data, generating insights and summaries.
- **Interactive Dashboards**: Creates and displays two interactive dashboards:
    - **Executive Summary Dashboard**: For a high-level overview of the key findings.
    - **Detailed Analysis Dashboard**: For a more in-depth exploration of the data.
- **Custom Queries**: A chat interface allows users to ask specific questions about their data and receive tailored analyses.
- **Data Cleaning**: The system cleans the data as part of its analysis, and the cleaned dataset is available for download.

## How to Run the Project

This project has two main components: the FastAPI backend and the Streamlit frontend. These components are independent and provide two different ways to interact with the data analysis agents. You only need to run the one you wish to use.

### Running the FastAPI Backend

1.  **Navigate to the project directory:**
    ```bash
    cd data_analysis_project
    ```
2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the FastAPI application:**
    ```bash
    uvicorn data_analysis_api.main:app --reload
    ```
4.  The API will be available at `http://localhost:8000`.

### Running the Streamlit Frontend

1.  **Navigate to the project directory:**
    ```bash
    cd data_analysis_project
    ```
2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
4.  Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

## API Usage

You can interact with the FastAPI backend using any API client, such as `curl` or Postman.

### Upload and Analyze a File

-   **Endpoint**: `POST /upload-and-analyze/`
-   **Description**: Uploads a CSV or Excel file for comprehensive analysis.
-   **Example**:
    ```bash
    curl -X POST -F "file=@/path/to/your/data.csv" http://localhost:8000/upload-and-analyze/
    ```

### Analyze Data from a MySQL Database

-   **Endpoint**: `POST /analyze-mysql/`
-   **Description**: Analyzes data from a MySQL database.
-   **Example**:
    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{
      "host": "your_database_host",
      "user": "your_username",
      "password": "your_password",
      "database": "your_database_name",
      "table_name": "your_table_name"
    }' http://localhost:8000/analyze-mysql/
    ```

### Perform a Custom Query

-   **Endpoint**: `POST /custom-query/`
-   **Description**: Asks a specific question about the last analyzed dataset.
-   **Example**:
    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{"query": "What is the average value of the sales column?"}' http://localhost:8000/custom-query/
    ```

### Download Cleaned Data

-   **Endpoint**: `GET /download-cleaned-data/`
-   **Description**: Downloads the cleaned data from the last analysis.
-   **Example**:
    ```bash
    curl -o cleaned_data.csv http://localhost:8000/download-cleaned-data/
    ```

## API Workflow Example

Here’s a step-by-step example of how to use the API to analyze a dataset.

### Step 1: Upload Your Data and Get a Comprehensive Analysis

First, upload your data file (e.g., `my_data.csv`) to the `/upload-and-analyze/` endpoint.

```bash
curl -X POST -F "file=@my_data.csv" http://localhost:8000/upload-and-analyze/
```

The server will process the file and respond with a JSON object containing the initial comprehensive analysis, along with the HTML for the executive and detailed dashboards.

**Example Response:**
```json
{
  "message": "Comprehensive analysis completed successfully.",
  "comprehensive_report": "...",
  "executive_dashboard_html": "<html>...</html>",
  "detailed_dashboard_html": "<html>...</html>",
  "cleaned_data_download_url": "/download-cleaned-data/"
}
```

You can save the dashboard HTML content to files and open them in a browser to view the visualizations.

### Step 2: Ask a Custom Question

After the initial analysis, you can ask specific questions about your data using the `/custom-query/` endpoint.

```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"query": "What are the top 5 products by sales?"}' \
     http://localhost:8000/custom-query/
```

**Example Response:**
```json
{
  "message": "Custom analysis completed successfully.",
  "custom_report": "The top 5 products by sales are: ...",
  "chart_html": "<html>...</html>"
}
```
This response includes a textual answer and may also include HTML for a new chart, which you can view.

### Step 3: Download the Cleaned Data

Finally, you can download the cleaned dataset that was generated during the analysis.

```bash
curl -o cleaned_data.csv http://localhost:8000/download-cleaned-data/
```

This command will save the cleaned data to a file named `cleaned_data.csv` in your current directory.

## Project Structure

```
data_analysis_project/
├── data_analysis_api/
│   ├── static/
│   │   └── index.html
│   └── main.py
├── agents/
│   ├── crew_agents.py
│   ├── crew_tasks.py
│   └── crew_tools.py
├── app.py
├── main_crewai_orchestrator.py
├── requirements.txt
└── README.md
```

-   `data_analysis_api/main.py`: The FastAPI application that serves the data analysis API.
-   `app.py`: The main Streamlit application file that creates the user interface.
-   `main_crewai_orchestrator.py`: Orchestrates the data analysis crew and their tasks.
-   `agents/`: Contains the definitions for the different agents, their tasks, and their tools.
-   `requirements.txt`: Lists the Python dependencies required to run the project.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
