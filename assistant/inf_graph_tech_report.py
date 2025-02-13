from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import Send, START, END
from langgraph.graph import StateGraph

from assistant.inf_graph_interview import build_graph as interview_builder
from assistant.inf_graph_schema import ResearchGraphState, Analyst, InterviewState
from assistant.services import safe_invoke


INSTRUCTIONS_FULL_REPORT_WRITER = """You are a technical writer creating a report on this overall topic:

{topic}
    
You have a team of analysts. Each analyst has done two things: 

1. They conducted an interview with an expert on a specific sub-topic.
2. They write up their finding into a memo.

Your task: 

1. You will be given a collection of memos from your analysts.
2. Think carefully about the insights from each memo.
3. Consolidate these into a crisp overall summary that ties together the central ideas from all of the memos. 
4. Summarize the central points in each memo into a cohesive single narrative.

To format your report:
 
1. Use markdown formatting. 
2. Include no pre-amble for the report.
3. Use no sub-heading. 
4. Start your report with a single title header: ## Insights
5. Do not mention any analyst names in your report.
6. Preserve any citations in the memos, which will be annotated in brackets, for example [1] or [2].
7. Create a final, consolidated list of sources and add to a Sources section with the `## Sources` header.
8. List your sources in order and do not repeat.

[1] Source 1
[2] Source 2

Here are the memos from your analysts to build your report from: 

{context}"""


def initialize_graph(state: ResearchGraphState) -> dict[str, Any]:
    # Full set of sections
    sections = state['sections']
    topic = state['topic']

    # Concat all sections together
    formatted_str_sections = '\n\n'.join([f'{section}' for section in sections])

    # Summarize the sections into a final report
    system_message = INSTRUCTIONS_FULL_REPORT_WRITER.format(topic=topic, context=formatted_str_sections)
    report = safe_invoke(
        [SystemMessage(content=system_message)] + [HumanMessage(content=f'Write a report based upon these memos.')]
    )
    return {'content': report.content}


def write_report(state: ResearchGraphState) -> dict[str, Any]:
    # Full set of sections
    sections = state['sections']
    topic = state['topic']

    # Concat all sections together
    formatted_str_sections = '\n\n'.join([f'{section}' for section in sections])

    # Summarize the sections into a final report
    system_message = INSTRUCTIONS_FULL_REPORT_WRITER.format(topic=topic, context=formatted_str_sections)
    report = safe_invoke(
        [SystemMessage(content=system_message)] + [HumanMessage(content=f'Write a report based upon these memos.')]
    )
    return {'content': report.content}


INSTRUCTIONS_FULL_REPORT_INTRO_AND_CONCLUSION = """You are a technical writer finishing a report on {topic}

You will be given all of the sections of the report.

You job is to write a crisp and compelling introduction or conclusion section.

The user will instruct you whether to write the introduction or conclusion.

Include no pre-amble for either section.

Target around 100 words, crisply previewing (for introduction) or recapping (for conclusion) all of the sections of the report.

Use markdown formatting. 

For your introduction, create a compelling title and use the # header for the title.

For your introduction, use ## Introduction as the section header. 

For your conclusion, use ## Conclusion as the section header.

Here are the sections to reflect on for writing: {formatted_str_sections}"""


def write_introduction(state: ResearchGraphState) -> dict[str, Any]:
    # Full set of sections
    sections = state['sections']
    topic = state['topic']

    # Concat all sections together
    formatted_str_sections = '\n\n'.join([f'{section}' for section in sections])

    # Summarize the sections into a final report

    instructions = INSTRUCTIONS_FULL_REPORT_INTRO_AND_CONCLUSION.format(topic=topic, formatted_str_sections=formatted_str_sections)
    intro = safe_invoke([instructions] + [HumanMessage(content=f'Write the report introduction')])
    return {'introduction': intro.content}


def write_conclusion(state: ResearchGraphState) -> dict[str, Any]:
    # Full set of sections
    sections = state['sections']
    topic = state['topic']

    # Concat all sections together
    formatted_str_sections = '\n\n'.join([f'{section}' for section in sections])

    # Summarize the sections into a final report

    instructions = INSTRUCTIONS_FULL_REPORT_INTRO_AND_CONCLUSION.format(topic=topic, formatted_str_sections=formatted_str_sections)
    conclusion = safe_invoke([instructions] + [HumanMessage(content=f'Write the report conclusion')])
    return {'conclusion': conclusion.content}


def finalize_report(state: ResearchGraphState) -> dict[str, str]:
    """ This is the "reduce" step where we gather all the sections, combine them, and reflect on them to write the intro/conclusion """
    # Save full final report
    content = state['content']
    if content.startswith('## Insights'):
        content = content.strip('## Insights')
    if '## Sources' in content:
        try:
            content, sources = content.split('\n## Sources\n')
        except:
            sources = None
    else:
        sources = None

    final_report = state['introduction'] + '\n\n---\n\n' + content + '\n\n---\n\n' + state['conclusion']
    if sources is not None:
        final_report += '\n\n## Sources\n' + sources
    return {'final_report': final_report}


def build_graph() -> StateGraph:
    # Add nodes and edges
    builder = StateGraph(ResearchGraphState)

    builder.add_node('write_report', write_report)
    builder.add_node('write_introduction', write_introduction)
    builder.add_node('write_conclusion', write_conclusion)
    builder.add_node('finalize_report', finalize_report)

    # Logic
    builder.add_edge(START, 'write_report')
    builder.add_edge(START, 'write_introduction')
    builder.add_edge(START, 'write_conclusion')
    builder.add_edge(['write_conclusion', 'write_report', 'write_introduction'], 'finalize_report')
    builder.add_edge('finalize_report', END)
    return builder


memory = MemorySaver()
graph = build_graph().compile(checkpointer=memory)
