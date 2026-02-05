# src/financial_researcher/tools/serper_wrapper.py
from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
from crewai_tools import SerperDevTool

class SerperWrapperInput(BaseModel):
    query: str = Field(
        ...,
        description="Plain text search query. Example: 'Apple current financial status'"
    )

class SerperWrapperTool(BaseTool):
    name: str = "internet_search"
    description: str = (
        "Search the internet using a SINGLE plain-text query. "
        "Input must be an object with a single field query (string)"
    )
    args_schema: Type[BaseModel] = SerperWrapperInput

    def _run(self, query: str) -> str:
        if not isinstance(query, str):
            raise ValueError("Query must be a string")

        serper = SerperDevTool()
        return serper.run(search_query=query)
