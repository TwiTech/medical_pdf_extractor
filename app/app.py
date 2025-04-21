import sys
import sqlite3
from tempfile import NamedTemporaryFile
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import chainlit as cl
from chainlit.types import AskFileResponse

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.vectorstores import FAISS

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.chat_models import ChatOpenAI

import spacy

from prompt import EXAMPLE_PROMPT, PROMPT, WELCOME_MESSAGE

# Initialize medical NER (SciSpaCy)
nlp = spacy.load("en_ner_bc5cdr_md")

# Simple SQLite connection (optional, can store file metadata or logs if needed)
conn = sqlite3.connect("mydatabase.db")


def extract_medical_terms(text):
    """ Extract medical entities using SciSpaCy """
    doc = nlp(text)
    terms = [(ent.text, ent.label_) for ent in doc.ents]
    return terms


def process_file(*, file: AskFileResponse) -> list:
    if file.type != "application/pdf":
        raise TypeError("Only PDF files are supported")

    with NamedTemporaryFile(delete=False) as tempfile:
        with open(file.path, "rb") as source_file:
            tempfile.write(source_file.read())

        loader = PDFPlumberLoader(tempfile.name)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=100
        )

        docs = text_splitter.split_documents(documents)

        for i, doc in enumerate(docs):
            doc.metadata["source"] = f"source_{i}"

        if not docs:
            raise ValueError("PDF file parsing failed.")

        return docs


def create_search_engine(*, file: AskFileResponse):
    docs = process_file(file=file)
    cl.user_session.set("docs", docs)

    encoder = HuggingFaceEmbeddings(model_name="monologg/biobert_v1.1")

    search_engine = FAISS.from_documents(docs, encoder)
    return search_engine


@cl.on_chat_start
async def start():
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content=WELCOME_MESSAGE,
            accept=["application/pdf"],
            max_size_mb=20,
        ).send()

    file = files[0]
    msg = cl.Message(content=f"Processing medical document `{file.name}`...")
    await msg.send()

    try:
        search_engine = await cl.make_async(create_search_engine)(file=file)
    except Exception as e:
        await cl.Message(content=f"Error: {e}").send()
        raise SystemError

    cl.user_session.set("search_engine", search_engine)

    msg.content = f"`{file.name}` processed. You can now ask medical questions about the document!"
    await msg.update()


async def get_answer_with_fallback(question: str, context_docs: list):
    """ Try GPT-4 first, fall back to LLaMA if OpenAI fails """
    full_text = "\n".join([doc.page_content for doc in context_docs])
    terms = extract_medical_terms(full_text)

    prompt = f"""
    Based on the provided medical document, answer the following question.
    Document Text:
    {full_text}

    Question: {question}
    """

    # Attempt GPT-4 first
    try:
        llm = ChatOpenAI(model='gpt-4', temperature=0)
        response = llm.invoke(prompt)
    except Exception as e:
        print(f"OpenAI failed: {e}")
        response = None

    # Fallback to LLaMA if GPT-4 fails
    if not response:
        print("Falling back to LLaMA")
        llm = LlamaCpp(model_path="path/to/llama-7b.gguf", temperature=0)
        response = llm.invoke(prompt)

    # Append detected terms to response
    terms_str = ", ".join([f"{text} ({label})" for text, label in terms]) or "None detected"
    final_answer = f"{response}\n\n**Detected Medical Terms:** {terms_str}"

    return final_answer


@cl.on_message
async def main(message: cl.Message):
    search_engine = cl.user_session.get("search_engine")
    docs = cl.user_session.get("docs")

    retriever = search_engine.as_retriever(search_kwargs={"k": 5})
    relevant_docs = retriever.get_relevant_documents(message.content)

    if not relevant_docs:
        await cl.Message(content="No relevant content found in the document.").send()
        return

    # Hybrid Answer Generation
    answer = await get_answer_with_fallback(message.content, relevant_docs)

    source_elements = []
    metadatas = [doc.metadata for doc in docs]
    all_sources = [m["source"] for m in metadatas]

    # Collect actual sources used
    found_sources = [doc.metadata["source"] for doc in relevant_docs]
    for source_name in found_sources:
        try:
            index = all_sources.index(source_name)
            text = docs[index].page_content
            source_elements.append(cl.Text(content=text, name=source_name))
        except ValueError:
            continue

    if found_sources:
        answer += f"\n\n**Sources:** {', '.join(found_sources)}"
    else:
        answer += "\n\n**Sources:** No specific source found."

    await cl.Message(content=answer, elements=source_elements).send()
