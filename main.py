from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.runnables import RunnablePassthrough
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv

session_store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]


if __name__ == '__main__':
    load_dotenv()
    print("Hi! I'm Nellie, your personal NBA assistant. Hang tight while I load some information for you\n")

    chat = ChatOpenAI()
    output_parser = StrOutputParser()
    db = SQLDatabase.from_uri("sqlite:///games.db")

    sql_template = '''Given a question, first create a syntactically correct sqlite query to run, then look at 
    the results of the query and return the answer. Select up to {top_k} rows from the following tables: {table_info}
    Retrieve only the output from the database. Use the following format to process the query:

    Question: "Question"
    SQLQuery: "SQL Query to run"
    SQLResult: "Result of the SQLQuery"
    Answer: "Final answer here"
    
    Only return the answer, without the SQL result and SQL query.
    
    If given a date, format it in the format "MMMM DD, YYYY". Example: november 1st 2022 would be "November 1, 
    2022". Search for dates in this format. If asked for games on a specific date, list them all, not just the first 
    one.
    
    Question: {input}
    '''
    sql_prompt = PromptTemplate.from_template(sql_template)
    sql_chain = create_sql_query_chain(chat, db, prompt=sql_prompt)

    add_context_to_question_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """Given a chat history and the latest user question which might reference context in the chat 
            history, formulate a standalone question which can be understood without the chat history. Do NOT answer 
            the question, just reformulate it if needed and otherwise return it as is. The context consists of 
            information about NBA games played. Keep question concide, use the same wording for every question when 
            adding context. When a general question is asked about who played a game, assume the user is asking for 
            all teams playing that day, unless specified otherwise. Mention explicitly when the question asks for a 
            specific team versus all teams"""),
            MessagesPlaceholder("messages"),
            ("human", "{input}"),
        ]
    )

    context_chain = add_context_to_question_prompt | chat | {"input": output_parser,
                                                             "question": output_parser,
                                                             "top_k": RunnablePassthrough(),
                                                             "table_info": RunnablePassthrough()}

    format_answer_prompt = PromptTemplate.from_template('''You should receive an input that contains a substring 
    "Answer:". Return everything after that substring. Do not modify it, only return it. If the substring is not 
    present, return that you don't know the answer
    
    {input}''')

    format_answer_chain = format_answer_prompt | chat | output_parser

    chain = context_chain | sql_chain

    conversation = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="messages"
    )
    print("Please enter a session ID:")
    sessionID = input()

    print("\nI'm ready! Ask me anything or enter ':q' to quit")
    while True:
        userMessage = input()
        if userMessage in [":q", ":Q"]:
            print("See you soon!\n\nExited successfully")
            break
        else:
            sql_res = conversation.invoke(
                {"input": userMessage, "top_k": "20", "table_info": "games"},
                {"configurable":
                     {"session_id": sessionID}
                 },
            )

            format_answer_res = format_answer_chain.invoke({"input": sql_res})
            print(format_answer_res)
