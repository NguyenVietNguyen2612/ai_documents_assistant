from typing import List
from app.services.retriever import Retriever
from app.services.context_builder import ContextBuilder

from app.tools.retrieve_tool import make_retrieve_tool
from app.tools.list_documents_tool import make_list_documents_tool
from app.tools.document_overview_tool import make_document_overview_tool
from app.tools.calculator_tool import make_calculator_tool


def make_agent_tools(retriever: Retriever, context_builder: ContextBuilder) -> List:
    """Khởi tạo toàn bộ bộ công cụ Multi-Tool cho Agent."""
    return [
        make_retrieve_tool(retriever, context_builder),
        make_list_documents_tool(retriever),
        make_document_overview_tool(retriever),
        make_calculator_tool(),
    ]


__all__ = [
    "make_retrieve_tool",
    "make_list_documents_tool",
    "make_document_overview_tool",
    "make_calculator_tool",
    "make_agent_tools",
]
