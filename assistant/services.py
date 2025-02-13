import os

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from ratelimit import limits, sleep_and_retry

from assistant.inf_graph_schema import Perspectives, SearchQuery
from utils.fs_utils import load_api_key

os.environ['OPENAI_API_KEY'] = load_api_key('openai.api_key')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_API_KEY'] = load_api_key('langchain.api_key')
os.environ['LANGCHAIN_PROJECT'] = 'langchain-academy'
os.environ['TAVILY_API_KEY'] = load_api_key('tavily.api_key')

# Web search tool
tavily_search = TavilySearchResults(max_results=3)

# LLM models
llm_4o_mini = ChatOpenAI(model='gpt-4o-mini', temperature=0)
llm_3_5_turbo = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
llm_o1_mini = ChatOpenAI(model='o1-mini', temperature=0)


@sleep_and_retry  # Automatically retries after waiting
@limits(calls=3, period=60)  # Limits to 3 calls per minute
def safe_invoke(*args, **kwargs):
    return llm_4o_mini.invoke(*args, **kwargs)

# Enforce structured output
structured_perspective_llm = llm_4o_mini.with_structured_output(Perspectives)
structured_searchquery_llm = llm_4o_mini.with_structured_output(SearchQuery)

@sleep_and_retry
@limits(calls=3, period=60)
def safe_invoke_perspective(*args, **kwargs):
    return structured_perspective_llm.invoke(*args, **kwargs)


@sleep_and_retry
@limits(calls=3, period=60)
def safe_invoke_searchquery(*args, **kwargs):
    return structured_searchquery_llm.invoke(*args, **kwargs)
