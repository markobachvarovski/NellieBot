from langchain_community.utilities.sql_database import SQLDatabase
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine

if __name__ == '__main__':

    load_dotenv()

    df = pd.read_csv("./assets/2022-2023_season.csv")

    engine = create_engine("sqlite:///games.db")
    df.to_sql("games", engine, index=False)

    db = SQLDatabase(engine=engine)


