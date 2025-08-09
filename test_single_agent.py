# test_single_agent.py
# Test a single agent with a simple task to make sure it works

import os
from crewai import Agent, LLM, Task, Crew
from dotenv import load_dotenv

load_dotenv()

gemini_api_key = os.getenv("GOOGLE_API_KEY")

# Use the correct format with provider prefix
test_llm = LLM(
    model="ollama/llama3.1:8b",  # Use ollama/ prefix for Ollama models
    verbose=True,
    temperature=0.7,
)

# Create a simple test agent
test_agent = Agent(
    role='Test Agent',
    backstory="You are a test agent to verify the LLM configuration works correctly.",
    goal="Respond to simple queries to test the setup.",
    verbose=True,
    allow_delegation=False,
    llm=test_llm
)

# Create a simple test task
test_task = Task(
    description="Say hello and confirm that you can access the data file path: C:\\Users\\sahil\\Downloads\\stock_data.csv",
    expected_output="A simple greeting and confirmation that you received the file path.",
    agent=test_agent
)

# Create a simple crew with just one agent and task
test_crew = Crew(
    agents=[test_agent],
    tasks=[test_task],
    verbose=True
)

print("=== TESTING SINGLE AGENT WITH CORRECT PROVIDER ===")
try:
    result = test_crew.kickoff()
    print("✅ SUCCESS! Agent execution completed.")
    print(f"Result: {result}")
except Exception as e:
    print(f"❌ FAILED: {e}")
    print("The model format might still be incorrect.")