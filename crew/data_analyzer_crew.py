from crewai import Crew
from agents.agent_1_validator import input_validator_agent
from agents.agent_2_problem_analyzer import problem_analyzer_agent
from agents.agent_3_data_cleaner import data_cleaner_agent
from agents.agent_4_data_analyzer import data_analyzer_agent
from agents.agent_5_predictive_modeler import predictive_modeler_agent
from agents.agent_6_insight_generator import insight_agent
from agents.agent_7_web_validator import web_scraper_agent
from agents.agent_8_qa_agent import qa_agent
from agents.agent_9_executive_report import executive_report_agent 

from tasks.task_definitions import all_tasks


class DataAnalyzerCrew:
    def __init__(self):
        self.crew = Crew(
            agents=[
                input_validator_agent,
                problem_analyzer_agent,
                data_cleaner_agent,
                data_analyzer_agent,    
                predictive_modeler_agent,
                insight_agent,
                web_scraper_agent,
                qa_agent,
                executive_report_agent
            ],
            tasks=all_tasks,
            verbose=True
        )
    
    def analyze(self, file_path: str, problem_statement: str):
        return self.crew.kickoff(
            inputs={
                "file_path": file_path,
                "problem_statement": problem_statement
            }
        )
