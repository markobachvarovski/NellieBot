from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

chat = ChatOpenAI()
output_parser = StrOutputParser()
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a chatbot having a conversation with a human about the NBA. You should answer questions about "
            "the NBA players, past or present, their career statistics and accolades.",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{input}")
    ]
)

chain = prompt | chat | output_parser
chat_history = ChatMessageHistory()

conversation = RunnableWithMessageHistory(
    chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="messages",
)

print("Welcome to your NBA chatbot! Ask it anything or enter ':q' to quit")
while True:
    userMessage = input()
    if userMessage == ":q":
        print("Exited successfully")
        break
    else:
        chat_history.add_user_message(userMessage)
        res = conversation.invoke(
            {"input": userMessage},
            {"configurable": {"session_id": "unused"}},
        )
        chat_history.add_ai_message(res)
        print(res)
