import os
from dotenv import load_dotenv
from rag.ingest import ingest_pdfs
from agent import run_agent

load_dotenv()

if __name__ == "__main__":
    # 1) Ingest PDFs once (path examples)
    pdfs = [
        "docs/48_Laws_of_Power.pdf",
        "docs/Quantum_computing_for_everyone_The_MIT_Press.pdf",
        "docs/The_Intelligent_Investor.pdf"
    ]
    # Comment this after first run if corpus is stable
    ingest_pdfs(pdfs)

    # 2) Queries
    print("Q1:", run_agent("What is section 3.1 about? Provide a brief answer and cite the source path."))
    print("Q2:", run_agent("Compute (17*24)+5 and show the final number."))
