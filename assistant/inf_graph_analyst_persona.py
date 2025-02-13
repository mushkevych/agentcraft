from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph

from assistant.inf_graph_schema import GenerateAnalystsState, Analyst
from assistant.services import safe_invoke_perspective

INSTRUCTIONS_CREATE_ANALYST_PERSONAS = """
You are tasked with creating a set of Analyst Personas. Follow these instructions carefully:
                    
1. First, review the research topic: {topic}
                    
2. Examine any editorial feedback that has been optionally provided to guide creation of the Analyst Personas:
                    
{human_analyst_feedback}
                    

3. Determine the most interesting themes based upon documents and / or feedback above.
                    
4. Pick the top {max_analysts} themes.
                    
5. Assign one analyst to each theme.
"""


def create_analysts(state: GenerateAnalystsState) -> dict[str, list[Analyst]]:
    """ Create AI Analysts Personas """

    topic = state['topic']
    max_analysts = state['max_analysts']
    human_analyst_feedback = state.get('human_analyst_feedback', '')

    # System message
    system_message = INSTRUCTIONS_CREATE_ANALYST_PERSONAS.format(
        topic=topic, human_analyst_feedback=human_analyst_feedback, max_analysts=max_analysts
    )

    # Generate question
    analysts = safe_invoke_perspective(
        [SystemMessage(content=system_message)] + [HumanMessage(content='Generate the set of Analyst Personas.')]
    )

    # Write the list of analysis to state
    return {'analysts': analysts.analysts}


def human_feedback(state: GenerateAnalystsState) -> None:
    """ No-op node that should be interrupted on """
    pass


def should_continue(state: GenerateAnalystsState) -> str:
    """ Return the next node to execute """

    # Check if human feedback
    human_analyst_feedback = state.get('human_analyst_feedback', None)
    if human_analyst_feedback:
        return 'create_analysts'

    # Otherwise end
    return END


def build_graph() -> StateGraph:
    # Add nodes and edges
    builder = StateGraph(GenerateAnalystsState)
    builder.add_node('create_analysts', create_analysts)
    builder.add_node('human_feedback', human_feedback)
    builder.add_edge(START, 'create_analysts')
    builder.add_edge('create_analysts', 'human_feedback')
    builder.add_conditional_edges('human_feedback', should_continue, ['create_analysts', END])
    return builder


memory = MemorySaver()
graph = build_graph().compile(interrupt_before=['human_feedback'], checkpointer=memory)
