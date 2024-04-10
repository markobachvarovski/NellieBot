from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine

if __name__ == '__main__':

    load_dotenv()

    # print("Loading information")

    df = pd.read_csv("./assets/2022-2023_season.csv")

    engine = create_engine("sqlite:///games.db")
    # df.to_sql("games", engine, index=False)

    db = SQLDatabase(engine=engine)
    print(db.dialect)
    print(db.get_usable_table_names())
    # print(db.run("SELECT * FROM games WHERE 'Home team' == 'Toronto Raptors';"))

    agent_executor = create_sql_agent(ChatOpenAI(), db=db, agent_type="openai-tools", verbose=True)
    print(agent_executor.invoke({"input": "what games were played on october 18th 2022?"})['output'])

    # loader = CSVLoader(file_path='./assets/2022-2023_season.csv', csv_args={
    #     'delimiter': ',',
    #     'fieldnames': ['Date', 'Start (ET)', 'Visitor team', 'Visitor team\'s points', 'Home team',
    #                    'Home team\'s points', 'Attendance', 'Arena', 'Notes']
    # })
    # data = loader.load()
    #
    # print("Splitting retrieved information")
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=200000, chunk_overlap=200)
    # splits = text_splitter.split_documents(data)
    #
    # print("Storing information in a vector store (this might take a while)")
    # vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(),
    #                                     persist_directory="./vectorstores/2023season")
    #
    # print("Storage complete")

