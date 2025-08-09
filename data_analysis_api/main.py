from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import shutil
import pandas as pd
import mysql.connector
from typing import Optional

from data_analysis_project.main_crewai_orchestrator import DataAnalysisCrew

app = FastAPI()

# Mount static files for the frontend
app.mount("/static", StaticFiles(directory="data_analysis_project/data_analysis_api/static"), name="static")

class QueryRequest(BaseModel):
    query: str

class MySQLRequest(BaseModel):
    host: str
    port: int = 3306
    user: str
    password: str
    database: str
    table_name: Optional[str] = None
    sql_query: Optional[str] = None

@app.get("/")
async def read_root():
    with open("data_analysis_project/data_analysis_api/static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/upload-and-analyze/")
async def upload_and_analyze(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        temp_file_path = f"temp_uploaded_data_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Initialize and run the data analysis crew
        data_crew = DataAnalysisCrew(data_path=temp_file_path)
        comprehensive_analysis_result = data_crew.run_comprehensive_analysis()

        # Clean up the temporary file
        os.remove(temp_file_path)

        # Read dashboard HTML files if they exist
        executive_dashboard_html = ""
        if os.path.exists("executive_summary_dashboard.html"):
            with open("executive_summary_dashboard.html", "r") as f:
                executive_dashboard_html = f.read()
        
        detailed_dashboard_html = ""
        if os.path.exists("detailed_analysis_dashboard.html"):
            with open("detailed_analysis_dashboard.html", "r") as f:
                detailed_dashboard_html = f.read()

        return JSONResponse(content={
            "message": "Comprehensive analysis completed successfully.",
            "comprehensive_report": comprehensive_analysis_result,
            "executive_dashboard_html": executive_dashboard_html,
            "detailed_dashboard_html": detailed_dashboard_html,
            "cleaned_data_download_url": "/download-cleaned-data/" if os.path.exists("data_analysis_project/cleaned_data.csv") else None
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-mysql/")
async def analyze_mysql_data(request: MySQLRequest):
    try:
        conn = mysql.connector.connect(
            host=request.host,
            port=request.port,
            user=request.user,
            password=request.password,
            database=request.database
        )
        cursor = conn.cursor()

        if request.sql_query:
            query = request.sql_query
        elif request.table_name:
            query = f"SELECT * FROM {request.table_name}"
        else:
            raise HTTPException(status_code=400, detail="Either table_name or sql_query must be provided.")

        cursor.execute(query)
        columns = [col[0] for col in cursor.description]
        data = cursor.fetchall()
        df = pd.DataFrame(data, columns=columns)

        cursor.close()
        conn.close()

        temp_file_path = "temp_mysql_data.csv"
        df.to_csv(temp_file_path, index=False)

        data_crew = DataAnalysisCrew(data_path=temp_file_path)
        comprehensive_analysis_result = data_crew.run_comprehensive_analysis()

        os.remove(temp_file_path)

        executive_dashboard_html = ""
        if os.path.exists("executive_summary_dashboard.html"):
            with open("executive_summary_dashboard.html", "r") as f:
                executive_dashboard_html = f.read()
        
        detailed_dashboard_html = ""
        if os.path.exists("detailed_analysis_dashboard.html"):
            with open("detailed_analysis_dashboard.html", "r") as f:
                detailed_dashboard_html = f.read()

        return JSONResponse(content={
            "message": "MySQL data analysis completed successfully.",
            "comprehensive_report": comprehensive_analysis_result,
            "executive_dashboard_html": executive_dashboard_html,
            "detailed_dashboard_html": detailed_dashboard_html,
            "cleaned_data_download_url": "/download-cleaned-data/" if os.path.exists("data_analysis_project/cleaned_data.csv") else None
        })

    except mysql.connector.Error as err:
        raise HTTPException(status_code=500, detail=f"MySQL Error: {err}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download-cleaned-data/")
async def download_cleaned_data():
    cleaned_data_path = "data_analysis_project/cleaned_data.csv"
    if os.path.exists(cleaned_data_path):
        return FileResponse(path=cleaned_data_path, filename="cleaned_data.csv", media_type="text/csv")
    raise HTTPException(status_code=404, detail="Cleaned data not found.")

@app.post("/custom-query/")
async def custom_query(request: QueryRequest):
    try:
        # This needs to be dynamic or persistent. For now, assuming a default or last uploaded/fetched data.
        data_crew = DataAnalysisCrew(data_path="temp_data.csv") 

        custom_analysis_result = data_crew.run_custom_query(request.query)

        chart_html = ""
        if os.path.exists("chart.html"):
            with open("chart.html", "r") as f:
                chart_html = f.read()

        return JSONResponse(content={
            "message": "Custom analysis completed successfully.",
            "custom_report": custom_analysis_result,
            "chart_html": chart_html
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
