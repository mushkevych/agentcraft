from typing import Literal, Any

from langchain_community.document_loaders import WikipediaLoader
from langchain_core.messages import get_buffer_string, SystemMessage, AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from assistant.inf_graph_schema import InterviewState
from assistant.services import safe_invoke, tavily_search, structured_searchquery_llm

INSTRUCTIONS_ANALYST_INTERVIEWS_EXPERT = """You are an analyst tasked with interviewing an expert to learn about a specific topic. 

Your goal is boil down to interesting and specific insights related to your topic.

1. Interesting: Insights that people will find surprising or non-obvious.
        
2. Specific: Insights that avoid generalities and include specific examples from the expert.

Here is your topic of focus and set of goals: {goals}
        
Begin by introducing yourself using a name that fits your persona, and then ask your question.

Continue to ask questions to drill down and refine your understanding of the topic.
        
When you are satisfied with your understanding, complete the interview with: "Thank you so much for your help!"

Remember to stay in character throughout your response, reflecting the persona and goals provided to you."""


def generate_question(state: InterviewState):
    """ Node to generate a question """

    # Get state
    analyst = state['analyst']
    messages = state['messages']

    # Generate question
    system_message = INSTRUCTIONS_ANALYST_INTERVIEWS_EXPERT.format(goals=analyst.persona)
    question = safe_invoke([SystemMessage(content=system_message)] + messages)

    # Write messages to state
    return {'messages': [question]}


# Search query writing
INSTRUCTIONS_COMPOSE_SEARCH_QUERY = SystemMessage(content=f"""You will be given a conversation between an Analyst and an Expert.

Your goal is to generate a well-structured query for use in retrieval and / or web-search related to the conversation.
        
First, analyze the full conversation.

Pay particular attention to the final question posed by the analyst.

Convert this final question into a well-structured web search query""")


def search_web(state: InterviewState) -> dict[str, list[str]]:
    """ Retrieve docs from web search """

    # Search query
    search_query = structured_searchquery_llm.invoke([INSTRUCTIONS_COMPOSE_SEARCH_QUERY] + state['messages'])

    # Search
    search_docs = tavily_search.invoke(search_query.search_query)

    # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

    return {'context': [formatted_search_docs]}


def search_wikipedia(state: InterviewState) -> dict[str, list[str]]:
    """ Retrieve docs from wikipedia """

    # Search query
    search_query = structured_searchquery_llm.invoke([INSTRUCTIONS_COMPOSE_SEARCH_QUERY] + state['messages'])

    # Search
    search_docs = WikipediaLoader(query=search_query.search_query, load_max_docs=2).load()

    # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )

    return {'context': [formatted_search_docs]}


INSTRUCTIONS_EXPERT_ANSWER = """You are an Expert being interviewed by an Analyst.

Here is analyst area of focus: {goals}. 
        
You goal is to answer a question posed by the interviewer.

To answer question, use this context:
        
{context}

When answering questions, follow these guidelines:
        
1. Use only the information provided in the context. 
        
2. Do not introduce external information or make assumptions beyond what is explicitly stated in the context.

3. The context contain sources at the topic of each individual document.

4. Include these sources your answer next to any relevant statements. For example, for source # 1 use [1]. 

5. List your sources in order at the bottom of your answer. [1] Source 1, [2] Source 2, etc
        
6. If the source is: <Document source="assistant/docs/llama3_1.pdf" page="7"/>' then just list: 
        
[1] assistant/docs/llama3_1.pdf, page 7 
        
And skip the addition of the brackets as well as the Document source preamble in your citation."""


def generate_answer(state: InterviewState) -> dict[str, list[dict[str, Any]]]:
    """ Node to answer a question """

    # Get state
    analyst = state['analyst']
    messages = state['messages']
    context = state['context']

    # Answer question
    system_message = INSTRUCTIONS_EXPERT_ANSWER.format(goals=analyst.persona, context=context)
    answer = safe_invoke([SystemMessage(content=system_message)] + messages)

    # Name the message as coming from the expert
    answer.name = 'expert'

    # Append it to state
    return {'messages': [answer]}


def save_interview(state: InterviewState) -> dict[str, str]:
    """ Save interviews """

    # Get messages
    messages = state['messages']

    # Convert interview to a string
    interview = get_buffer_string(messages)

    # Save to interviews key
    return {'interview': interview}


def route_messages(state: InterviewState, name: str = 'expert') -> Literal['save_interview', 'ask_question']:
    """ Route between question and answer """

    # Get messages
    messages = state['messages']
    max_num_turns = state.get('max_num_turns', 2)

    # Check the number of expert answers
    num_responses = len(
        [m for m in messages if isinstance(m, AIMessage) and m.name == name]
    )

    # End if expert has answered more than the max turns
    if num_responses >= max_num_turns:
        return 'save_interview'

    # This router is run after each question - answer pair
    # Get the last question asked to check if it signals the end of discussion
    last_question = messages[-2]

    if 'Thank you so much for your help' in last_question.content:
        return 'save_interview'
    return 'ask_question'


INSTRUCTION_SECTION_WRITER = """You are an expert Technical Writer.
            
Your task is to create a short, easily digestible section of a report based on a set of source documents.

1. Analyze the content of the source documents: 
- The name of each source document is at the start of the document, with the <Document tag.
        
2. Create a report structure using markdown formatting:
- Use ## for the section title
- Use ### for sub-section headers
        
3. Write the report following this structure:
a. Title (## header)
b. Summary (### header)
c. Sources (### header)

4. Make your title engaging based upon the focus area of the analyst: 
{focus}

5. For the summary section:
- Set up summary with general background / context related to the focus area of the analyst
- Emphasize what is novel, interesting, or surprising about insights gathered from the interview
- Create a numbered list of source documents, as you use them
- Do not mention the names of interviewers or experts
- Aim for approximately 400 words maximum
- Use numbered sources in your report (e.g., [1], [2]) based on information from source documents
        
6. In the Sources section:
- Include all sources used in your report
- Provide full links to relevant websites or specific document paths
- Separate each source by a newline. Use two spaces at the end of each line to create a newline in Markdown.
- It will look like:

### Sources
[1] Link or Document name
[2] Link or Document name

7. Be sure to combine sources. For example this is not correct:

[3] https://ai.meta.com/blog/meta-llama-3-1/
[4] https://ai.meta.com/blog/meta-llama-3-1/

There should be no redundant sources. It should simply be:

[3] https://ai.meta.com/blog/meta-llama-3-1/
        
8. Final review:
- Ensure the report follows the required structure
- Include no preamble before the title of the report
- Check that all guidelines have been followed"""


def write_section(state: InterviewState) -> dict[str, list[str]]:
    """ Node to answer a question """

    # Get state
    interview = state['interview']
    context = state['context']
    analyst = state['analyst']

    # Write section using either the gathered source docs from interview (context) or the interview itself (interview)
    system_message = INSTRUCTION_SECTION_WRITER.format(focus=analyst.description)
    section = safe_invoke(
        [SystemMessage(content=system_message)] + [
            HumanMessage(content=f'Use this source to write your section: {context}')]
    )

    # Append it to state
    return {'sections': [section.content]}


def build_graph() -> StateGraph:
    # Add nodes and edges
    interview_builder = StateGraph(InterviewState)
    interview_builder.add_node('ask_question', generate_question)
    interview_builder.add_node('search_web', search_web)
    interview_builder.add_node('search_wikipedia', search_wikipedia)
    interview_builder.add_node('answer_question', generate_answer)
    interview_builder.add_node('save_interview', save_interview)
    interview_builder.add_node('write_section', write_section)

    # Flow
    interview_builder.add_edge(START, 'ask_question')
    interview_builder.add_edge('ask_question', 'search_web')
    interview_builder.add_edge('ask_question', 'search_wikipedia')
    interview_builder.add_edge('search_web', 'answer_question')
    interview_builder.add_edge('search_wikipedia', 'answer_question')
    interview_builder.add_conditional_edges('answer_question', route_messages, ['ask_question', 'save_interview'])
    interview_builder.add_edge('save_interview', 'write_section')
    interview_builder.add_edge('write_section', END)

    return interview_builder


# Interview
memory = MemorySaver()
graph = build_graph().compile(checkpointer=memory).with_config(run_name='Conduct Interviews')
