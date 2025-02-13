import base64
from typing import Any, List

import networkx as nx
import panel as pn
from langchain_core.messages import HumanMessage
from langchain_openai import OpenAI
from langgraph.graph import StateGraph
from pyvis.network import Network
from tavily import TavilyClient

from assistant.inf_graph_analyst_persona import graph as graph_analyst_persona
from assistant.inf_graph_schema import ResearchGraphState, Analyst
from assistant.inf_graph_tech_report import graph as graph_tech_report
from assistant.inf_graph_interview import graph as graph_interview
from utils.fs_utils import load_api_key


def generate_graph_html(graph: StateGraph) -> pn.pane.HTML:
    """Generates a PyVis graph from the LangGraph instance, embedded via a Base64 data URI."""
    # Build a DiGraph from the LangGraph edges
    G: nx.DiGraph = nx.DiGraph()
    edges = graph.get_graph(xray=1).edges
    for edge in edges:
        G.add_edge(edge[0], edge[1])

    # Create a PyVis network
    nt: Network = Network(height='500px', width='500px', directed=True)
    nt.from_nx(G)

    # Get the raw HTML for the PyVis network
    net_html: str = nt.generate_html()

    # Convert the HTML to a Base64 data URI and embed it in an iframe
    net_html_b64: str = base64.b64encode(net_html.encode()).decode()
    iframe_html: str = f"""
        <iframe 
            src="data:text/html;base64,{net_html_b64}"
            height="500px" 
            width="500px" 
            style="border:none;"
        ></iframe>
        """
    return pn.pane.HTML(iframe_html, sizing_mode='stretch_both')


class ChatFeed(pn.Column):
    """
    A simple chat feed widget that displays conversation messages with scrolling enabled.
    """

    def __init__(self, height: int = 400, **params: Any) -> None:
        super().__init__(**params)
        self.chat_messages: List[str] = []
        self.margin = 0
        self.spacing = 5
        self.height = height  # Set a fixed height for scrolling
        self.scroll = True  # Enable scrolling
        self.sizing_mode = 'stretch_width'

        # Initialize the chat feed with a placeholder
        self.update_feed()

    def update_feed(self) -> None:
        """Refresh the chat feed display with current messages."""
        self.objects = [pn.pane.Markdown(msg) for msg in self.chat_messages]

    def add_message(self, msg: str) -> None:
        """Append a new message and update the display."""
        self.chat_messages.append(msg)
        self.update_feed()


class AssistantApp:
    def __init__(self) -> None:
        # Initialize API Clients
        self.llm = OpenAI(api_key=load_api_key('openai.api_key'))
        self.web_search = TavilyClient(api_key=load_api_key('tavily.api_key'))

        # LLM conversation artifacts
        self.conversation_thread = {'configurable': {'thread_id': '1'}}
        self.analyst_personas: list[Analyst] = list()
        self.report_sections: list[str] = list()
        self.final_report: str = ''

        # UI Components for query processing
        self.query_input = pn.widgets.TextInput(name='Enter your question', sizing_mode = 'stretch_width')
        self.query_input.value = 'What compounds have a similar molecular fingerprint to Ibuprofen?'

        self.submit_button = pn.widgets.Button(name='Next', button_type='primary')
        self.submit_button.on_click(self.create_analyst_personas)

        # -----------------------------
        # Construct Analyst Personas Panel
        # -----------------------------
        html_graph_analyst: pn.pane.HTML = generate_graph_html(graph_analyst_persona)

        self.ti_analyst_number = pn.widgets.TextInput(name='Number of Analyst Personas')
        self.ti_analyst_number.value = '3'

        self.ti_analyst_topic = pn.widgets.TextInput(name='Analyst Theme', sizing_mode='stretch_width')
        self.ti_analyst_topic.value = 'Enhancing Bioengineering Research by the Review and Analysis of Scientific Literature, Clinical Trials, and Drug Discovery Data'

        self.ti_analyst_input = pn.widgets.TextInput(placeholder='Enter human input', sizing_mode='stretch_width')
        self.btn_analyst_submit: pn.widgets.Button = pn.widgets.Button(name='Submit', button_type='primary')
        self.btn_analyst_submit.on_click(self.update_analyst_personas)

        self.clmn_analyst_personas: pn.Column = pn.Column('Constructed Analyst Personas')
        pnl_analyst_right: pn.Column = pn.Column(
            self.ti_analyst_number,
            self.ti_analyst_topic,
            pn.Row(self.ti_analyst_input, self.btn_analyst_submit),
            self.clmn_analyst_personas,
            sizing_mode='stretch_both',
            margin=10
        )

        self.panel_analyst = pn.Row(
            html_graph_analyst,
            pnl_analyst_right,
            sizing_mode='stretch_width'
        )

        # -----------------------------
        # Construct Interview Panel
        # -----------------------------
        html_graph_interview: pn.pane.HTML = generate_graph_html(graph_interview)
        self.ti_interview_question = pn.widgets.TextInput(sizing_mode='stretch_width')
        self.ti_interview_question.value = 'So you said you were writing an article on {topic}?'

        self.btn_interview_start = pn.widgets.Button(name='Start interview', button_type='primary')
        self.btn_interview_start.on_click(self.perform_interview)
        self.pb_interview_progress = pn.indicators.Progress(name='Interview Progress', value=0, max=100, width=400, height=20)
        self.pb_interview_progress.visible = False

        self.chat_interview = ChatFeed()
        pnl_interview_right = pn.Column(
            self.ti_interview_question,
            pn.Row(self.btn_interview_start, self.pb_interview_progress),
            self.chat_interview,
            sizing_mode='stretch_both',
            margin=10
        )

        self.panel_interview = pn.Row(
            html_graph_interview,
            pnl_interview_right,
            sizing_mode='stretch_width'
        )

        # -----------------------------
        # Construct Tech Report Panel
        # -----------------------------
        html_graph_report: pn.pane.HTML = generate_graph_html(graph_tech_report)

        self.btn_report_start = pn.widgets.Button(name='Construct Report', button_type='primary')
        self.btn_report_start.on_click(self.construct_report)

        self.chat_report_sections = ChatFeed()
        self.chat_report_final = ChatFeed()
        pnl_report_right = pn.Column(
            self.btn_report_start,
            pn.Tabs(
                ('Report Sections', self.chat_report_sections),
                ('Final Report', self.chat_report_final),
                dynamic=True
            ),
            sizing_mode='stretch_both',
            margin=10
        )

        self.panel_report = pn.Row(
            html_graph_report,
            pnl_report_right,
            sizing_mode='stretch_width'
        )

        # -----------------------------
        # Assemble the complete dashboard
        # -----------------------------
        self.accordion = pn.Accordion(
            ('Construct Analyst Personas', self.panel_analyst),
            ('Perform Interview', self.panel_interview),
            ('Construct Report', self.panel_report),
            toggle=True,
            active=[]  # all collapsed by default; e.g. [0] to open the first panel
        )

        self.dashboard = pn.Column(
            '# Assistant Dashboard',
            self.query_input,
            self.submit_button,
            pn.layout.Divider(),
            self.accordion
        )

    def create_analyst_personas(self, event: Any = None) -> None:
        self.accordion.active = [0]

        max_analysts = int(self.ti_analyst_number.value)
        topic = self.ti_analyst_topic.value
        for event in graph_analyst_persona.stream(input={'topic': topic, 'max_analysts': max_analysts},
                                                  config=self.conversation_thread,
                                                  stream_mode='values'):
            # Review
            self.analyst_personas = event.get('analysts', [])
            if self.analyst_personas:
                self.clmn_analyst_personas.clear()
                for analyst in self.analyst_personas:
                    self.clmn_analyst_personas.append(
                        f'Name: {analyst.name} Affiliation: {analyst.affiliation} Role: {analyst.role} Description: {analyst.description}')

                    print(f'Name: {analyst.name}')
                    print(f'Affiliation: {analyst.affiliation}')
                    print(f'Role: {analyst.role}')
                    print(f'Description: {analyst.description}')
                    print('-' * 50)

    def update_analyst_personas(self, event: Any = None) -> None:
        further_feedack = self.ti_analyst_input.value
        if not further_feedack.strip():
            # set to None if no additional instructions were provided by a user
            further_feedack = None

        graph_analyst_persona.update_state(
            config=self.conversation_thread,
            values={
                'human_analyst_feedback': further_feedack
            },
            as_node='human_feedback'
        )

        self.ti_analyst_input.value = ''
        self.create_analyst_personas(event)

    def perform_interview(self, event: Any = None) -> None:
        self.pb_interview_progress.visible = True
        self.pb_interview_progress.value = 0
        self.accordion.active = [1]
        self.btn_interview_start.disabled = True

        topic = self.ti_analyst_topic.value
        question = self.ti_interview_question.value.format(topic=topic)
        messages = [HumanMessage(question)]

        for i, analyst in enumerate(self.analyst_personas):
            interview = graph_interview.invoke(
                {'analyst': analyst, 'messages': messages, 'max_num_turns': 2},
                config=self.conversation_thread
            )

            for section in interview.get('sections', []):
                self.report_sections.append(section)
                self.chat_report_sections.add_message(section)
                self.chat_interview.add_message(section)
                # print(f'Section: {section}')
                # print('-' * 50)

            # Update progress bar
            self.pb_interview_progress.value = int(((i + 1) / len(self.analyst_personas)) * 100)

        self.btn_interview_start.disabled = False
        self.pb_interview_progress.visible = False

    def construct_report(self, event: Any = None) -> None:
        self.accordion.active = [2]

        topic = self.ti_analyst_topic.value
        self.final_report = ''
        self.chat_report_sections.clear()
        self.chat_report_final.clear()

        tech_report_state = ResearchGraphState(
            topic=topic,
            max_analysts=len(self.analyst_personas),
            human_analyst_feedback=None,
            analysts=self.analyst_personas,
            sections=self.report_sections,
            introduction='',
            content='',
            conclusion='',
            final_report=''
        )

        tech_report = graph_tech_report.invoke(tech_report_state, config=self.conversation_thread)
        self.final_report = tech_report.get('final_report')
        self.chat_report_final.add_message(self.final_report)

        print(f'Report: {self.final_report}')
        print('-' * 50)

    def get_dashboard(self) -> pn.Column:
        """Returns the Panel dashboard."""
        return self.dashboard
