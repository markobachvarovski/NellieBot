from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

if __name__ == '__main__':

    load_dotenv()

    print("Loading information")
    loader = CSVLoader(file_path='./assets/2022-2023_season.csv', csv_args={
        'delimiter': ',',
        'fieldnames': ['Date', 'Start (ET)', 'Visitor team', 'Visitor team\'s points', 'Home team',
                       'Home team\'s points', 'Attendance', 'Arena', 'Notes']
    })
    data = loader.load()

    print("Splitting retrieved information")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200000, chunk_overlap=200)
    splits = text_splitter.split_documents(data)

    print("Storing information in a vector store (this might take a while)")
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(),
                                        persist_directory="./vectorstores/2023season")

    print("Storage complete")

