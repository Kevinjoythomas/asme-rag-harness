"""Table extraction — Azure Document Intelligence (primary) + pdfplumber (fallback).

Azure DI preserves cell-level structure including merged cells and row/column spans.
pdfplumber is used when Azure credentials are not set. PyMuPDF flattens tables into
linear text and loses the row-column relationships that downstream calculations depend on,
which is why it is not used for this path.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.core.credentials import AzureKeyCredential

    _AZURE_SDK = True
except ImportError:
    _AZURE_SDK = False

try:
    import pdfplumber

    _PDFPLUMBER = True
except ImportError:
    _PDFPLUMBER = False


TableData = dict[str, Any]  # {page, clause, rows, headers, source}


def extract_tables_azure(pdf_path: Path) -> list[TableData]:
    """Cell-level table extraction via Azure Document Intelligence prebuilt-layout."""
    endpoint = os.environ.get("AZURE_DI_ENDPOINT", "")
    key = os.environ.get("AZURE_DI_KEY", "")
    if not endpoint or not key:
        raise EnvironmentError("Set AZURE_DI_ENDPOINT and AZURE_DI_KEY")
    if not _AZURE_SDK:
        raise ImportError("Install azure extra: uv sync --extra azure")

    client = DocumentIntelligenceClient(endpoint, AzureKeyCredential(key))
    with open(pdf_path, "rb") as f:
        poller = client.begin_analyze_document(
            "prebuilt-layout",
            analyze_request=f,
            content_type="application/pdf",
        )
    result = poller.result()

    tables: list[TableData] = []
    for table in result.tables or []:
        grid: list[list[str]] = [
            [""] * table.column_count for _ in range(table.row_count)
        ]
        for cell in table.cells or []:
            r, c = cell.row_index, cell.column_index
            if r < table.row_count and c < table.column_count:
                grid[r][c] = cell.content or ""

        page = (
            table.bounding_regions[0].page_number
            if table.bounding_regions
            else 0
        )
        tables.append(
            {
                "page": page,
                "clause": "",  # filled by caller from surrounding parse context
                "rows": grid,
                "headers": grid[0] if grid else [],
                "source": "azure_di",
            }
        )
    return tables


def extract_tables_pdfplumber(pdf_path: Path) -> list[TableData]:
    """Fallback table extraction via pdfplumber (does not handle merged cells)."""
    if not _PDFPLUMBER:
        return []

    tables: list[TableData] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            for raw in page.extract_tables() or []:
                if not raw:
                    continue
                rows = [[cell or "" for cell in row] for row in raw]
                tables.append(
                    {
                        "page": page_num,
                        "clause": "",
                        "rows": rows,
                        "headers": rows[0] if rows else [],
                        "source": "pdfplumber",
                    }
                )
    return tables


def extract_tables(pdf_path: Path, prefer_azure: bool = True) -> list[TableData]:
    """Extract tables, preferring Azure DI when credentials are configured."""
    if prefer_azure and _AZURE_SDK and os.environ.get("AZURE_DI_KEY"):
        return extract_tables_azure(pdf_path)
    return extract_tables_pdfplumber(pdf_path)
