# Student Name: jonatahn ahuche
# Student Index Number: 10022200183

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd
from pypdf import PdfReader


@dataclass
class Document:
    doc_id: str
    source: str
    title: str
    text: str
    metadata: dict


def _clean_text(text: str) -> str:
    text = text.replace("�", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\x00", "")
    return text.strip()


def load_csv_documents(csv_path: str) -> List[Document]:
    df = pd.read_csv(csv_path)
    df = df.dropna(how="all").fillna("")
    docs: List[Document] = []
    # Add an explicit schema document so the assistant can state dataset limits.
    schema_text = (
        "Dataset schema for Ghana_Election_Result.csv: "
        + ", ".join(df.columns.astype(str).tolist())
        + ". This dataset is primarily region-level vote records by year, candidate, and party. "
        "It does not include a constituency column."
    )
    docs.append(
        Document(
            doc_id="csv_schema",
            source=Path(csv_path).name,
            title="Ghana Election Dataset Schema",
            text=_clean_text(schema_text),
            metadata={"row_index": -1, "type": "schema"},
        )
    )

    # Add computed election summaries for easier retrieval of winner-style questions.
    if {"Year", "Candidate", "Party", "Votes"}.issubset(set(df.columns)):
        temp = df.copy()
        temp["Votes_num"] = pd.to_numeric(temp["Votes"], errors="coerce").fillna(0)
        yearly = (
            temp.groupby(["Year", "Candidate", "Party"], as_index=False)["Votes_num"]
            .sum()
            .sort_values(["Year", "Votes_num"], ascending=[True, False])
        )
        for year, year_df in yearly.groupby("Year"):
            top = year_df.iloc[0]
            summary_text = (
                f"Election yearly summary for {int(year)}. "
                f"Winner by total votes in this dataset: {top['Candidate']} ({top['Party']}) "
                f"with {int(top['Votes_num'])} votes aggregated across listed regions."
            )
            docs.append(
                Document(
                    doc_id=f"csv_year_summary_{int(year)}",
                    source=Path(csv_path).name,
                    title=f"Ghana Election Year Summary {int(year)}",
                    text=_clean_text(summary_text),
                    metadata={"year": int(year), "type": "year_summary"},
                )
            )

    for i, row in df.iterrows():
        parts = [f"{col}: {row[col]}" for col in df.columns if str(row[col]).strip()]
        text = _clean_text(" | ".join(parts))
        docs.append(
            Document(
                doc_id=f"csv_{i}",
                source=Path(csv_path).name,
                title="Ghana Election Result Row",
                text=text,
                metadata={"row_index": int(i)},
            )
        )
    return docs


def load_pdf_documents(pdf_path: str) -> List[Document]:
    reader = PdfReader(pdf_path)
    docs: List[Document] = []
    for i, page in enumerate(reader.pages):
        text = _clean_text(page.extract_text() or "")
        if not text:
            continue
        docs.append(
            Document(
                doc_id=f"pdf_{i}",
                source=Path(pdf_path).name,
                title=f"2025 Budget Statement Page {i+1}",
                text=text,
                metadata={"page_number": i + 1},
            )
        )
    return docs


def load_all_documents(csv_path: str, pdf_path: str) -> List[Document]:
    return load_csv_documents(csv_path) + load_pdf_documents(pdf_path)
