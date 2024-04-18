from typing import List

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings, OpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

session_store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]


if __name__ == '__main__':
    load_dotenv()
    print("Nellie: Hi! I'm Nellie, your personal NBA assistant. Hang tight while I load some information for you\n")

    print("Fetching information")
    vectorstore = Chroma(persist_directory="./vectorstores/2023season", embedding_function=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    chat = ChatOpenAI()

    from langchain_core.prompts import PromptTemplate

    db = SQLDatabase.from_uri("sqlite:///games.db")
    sql_template = '''Given an input question, first create a syntactically correct sqlite query to run, then look at 
    the results of the query and return the answer. Select up to {top_k} rows from the following tables: {table_info}
    Use the following format:

    Question: "{question}"
    SQL Query: "query"
    SQL Result: "result"
    Answer: "answer"
    
    If given a date, format it in the format "MMMM DD, YYYY". Example: november 1st 2022 would be "November 1, 2022". Search for dates in this format
    {input}
    '''
    sql_prompt = PromptTemplate.from_template(sql_template
        # [
        #     ("system", sql_template),
        #     MessagesPlaceholder("messages"),
        #     ("human", "{input}"),
        # ]
    )
    # sql_chain = sql_prompt | chat
    sql_chain = create_sql_query_chain(chat, db, prompt=sql_prompt)
    # sql_chain = create_stuff_documents_chain(chat, sql_prompt)
    # sql_prompt = PromptTemplate.from_template(sql_template)

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
        sql_chain,
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
            # res = conversation.invoke(
            #     {"input": userMessage},
            #     {"configurable":
            #          {"session_id": "4"}
            #      },
            # )
            # print(res)

            res = sql_chain.invoke({"input": userMessage, "question": userMessage, "top_k": "20", "table_info": "games"})
            print(res)
