import operator
from typing import List, Annotated

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class Analyst(BaseModel):
    affiliation: str = Field(
        description='Primary affiliation of the analyst.',
    )
    name: str = Field(
        description='Name of the analyst.'
    )
    role: str = Field(
        description='Role of the analyst in the context of the topic.',
    )
    description: str = Field(
        description='Description of the analyst focus, concerns, and motives.',
    )

    @property
    def persona(self) -> str:
        return f'Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n'


class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(
        description='Comprehensive list of Analysts Personas with their roles and affiliations.',
    )


class GenerateAnalystsState(TypedDict):
    topic: str  # Topic of the research
    max_analysts: int  # Number of analysts
    human_analyst_feedback: str  # Human feedback
    analysts: List[Analyst]  # Analyst asking questions


class InterviewState(MessagesState):
    max_num_turns: int  # Number turns of conversation
    context: Annotated[list, operator.add]  # Source docs
    analyst: Analyst  # Analyst asking questions
    interview: str  # Interview transcript
    sections: list  # Final key we duplicate in outer state for Send() API


class SearchQuery(BaseModel):
    search_query: str = Field(None, description='Search query for retrieval.')


class ResearchGraphState(TypedDict):
    topic: str  # Research topic
    max_analysts: int  # Number of analysts
    human_analyst_feedback: str  # Human feedback
    analysts: List[Analyst]  # Analyst asking questions
    sections: Annotated[list, operator.add]  # report sections, built in parallel via Send() call
    introduction: str  # Introduction for the final report
    content: str  # Content for the final report
    conclusion: str  # Conclusion for the final report
    final_report: str  # Final report
