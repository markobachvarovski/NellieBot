from typing import List

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import bs4

session_store = {}

months = ["october", "november", "december", "january", "february", "march", "april", "may", "june"]


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]


def get_urls() -> List[str]:
    urls = []
    for year in range(2023, 2024):
        for month in months:
            urls.append("https://www.basketball-reference.com/leagues/NBA_" + str(year) + "_games-" + month + ".html")

    return urls


if __name__ == '__main__':
    print("Nellie: Hi! I'm Nellie, your personal NBA assistant. Hang tight while I load some information for you\n")

    print("Fetching information")
    vectorstore = Chroma(persist_directory="./vectorstores/2023season", embedding_function=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    chat = ChatOpenAI()

    print("Creating prompt to contextualize question")
    add_context_to_question_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """Given a chat history and the latest user question which might reference context in the chat 
            history, formulate a standalone question which can be understood without the chat history. Do NOT answer the 
            question, just reformulate it if needed and otherwise return it as is. The context consists of information 
            about NBA games played"""),
            MessagesPlaceholder("messages"),
            ("human", "{input}"),
        ]
    )
    add_context_to_question_retriever = create_history_aware_retriever(
        chat, retriever, add_context_to_question_prompt
    )

    print("Creating prompt to answer questions")
    answer_question_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Your name is Nellie. You are a chatbot having a conversation with a human about the NBA. You should "
                "answer questions about the NBA players, past or present, their career statistics and accolades. In "
                "addition to your trained data, use the following pieces of retrieved context to answer the question. "
                "If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the "
                "answer concise."
                ""
                "{context}",
            ),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{input}")
        ]
    )
    answer_question_retriever = create_stuff_documents_chain(chat, answer_question_prompt)

    chain = create_retrieval_chain(add_context_to_question_retriever, answer_question_retriever)
    print("Chaining retrievers together\n")
    conversation = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="messages",
        output_messages_key="answer",
    )

    print("Nellie: I'm ready! Ask me anything or enter ':q' to quit")
    while True:
        userMessage = input()
        if userMessage in [":q", ":Q"]:
            print("Nellie: See you soon!\n\nExited successfully")
            break
        else:
            res = conversation.invoke(
                {"input": userMessage},
                {"configurable":
                     {"session_id": "3"}
                 },
            )['answer']
            print(res)
