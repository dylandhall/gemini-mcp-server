from pydantic import Field
from typing import Annotated, Literal, Union
from .utils import (
    TextToolOutput,
    WebSearchToolOutput,
    process_grounding_to_structured_citations,
    get_current_date,
    get_gemini_client,
)


web_search_prompt = """Conduct targeted Google Searches to gather the most recent, credible information on "{query}" and synthesize it into a verifiable text artifact.

Instructions:
- Query should ensure that the most current information is gathered. The current date is {current_date_str}.
- Conduct multiple, diverse searches to gather comprehensive information.
- Consolidate key findings while meticulously tracking the source(s) for each specific piece of information.
- The output should be a well-written summary or report based on your search findings. 
- Only include the information found in the search results, don't make up any information.

Research Topic:
{query}
"""


async def web_search(
    query: Annotated[
        str,
        Field(
            description="The query used to search the web. This query will be potential composed and optimized by the tool."
        ),
    ],
    include_citations: Annotated[
        bool,
        Field(
            description="""Whether to include citations and web search queries used during the search in the response. 
            This leads to a bigger response object. Make sure to only use this if the user asks for it or the response would benefit from it. 
            Default is False.""",
        ),
    ] = False,
) -> Union[WebSearchToolOutput, TextToolOutput]:
    """Use this tool to perform a Google web search on a given prompt to access the real-time and up-to-date information.
    It synthesizes findings from multiple, automatically generated Google searches into a coherent, verifiable summary.
    """

    genai_client = await get_gemini_client()
    current_date_str = get_current_date()

    response = await genai_client.aio.models.generate_content(
        model="gemini-3.5-flash",
        contents=web_search_prompt.format(
            query=query, current_date_str=current_date_str
        ),
        config={
            "temperature": 0.0,
            "tools": [{"google_search": {}}],
        },
    )

    structured_citations = []
    web_search_queries_used = []

    if response.candidates[0].grounding_metadata:
        structured_citations = process_grounding_to_structured_citations(
            response.candidates[0].grounding_metadata
        )

        # Extract web search queries if available
        if response.candidates[0].grounding_metadata.web_search_queries:
            web_search_queries_used = list(
                response.candidates[0].grounding_metadata.web_search_queries
            )

    if include_citations:
        return {
            "text": response.text,
            "web_search_queries": web_search_queries_used,
            "citations": structured_citations,
        }
    else:
        return {
            "text": response.text,
        }


async def use_gemini(
    prompt: Annotated[
        str,
        Field(
            description="The prompt or task for Gemini. This can be a question, a request for planning, reflection, or any other support."
        ),
    ],
    model: Annotated[
        Literal["gemini-3.5-flash", "gemini-3.1-flash-lite"],
        Field(
            description="The Gemini model to use. Use 'gemini-2.5-pro-preview-06-05' for complex tasks needing advanced reasoning and 'gemini-2.5-flash-preview-05-20' for speed and cost-efficiency."
        ),
    ] = "gemini-3.5-flash",
) -> TextToolOutput:
    """Use this tool to delegate a task to a specified Gemini model (Pro or Flash).

    This tool can be used for a wide range of tasks, including complex reasoning,
    content generation, summarization, and planning. It acts as a powerful
    assistant for requests that require advanced AI capabilities.
    """

    genai_client = await get_gemini_client()

    response = await genai_client.aio.models.generate_content(
        model=model,
        contents=prompt,
    )

    return {
        "text": response.text,
    }
