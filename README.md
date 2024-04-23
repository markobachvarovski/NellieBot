# NellieBot - Your own NBA chatbot
## Introduction
Nellie is your own personal NBA chatbot. You can ask the score of any game in NBA history, concluding with the current 2023-2024 season. Nellie uses a retrieval-augmented generation from an external source to give you a concise answer to your query. It also memorizes your previous messages, so feel free to talk to it as if it were human!
## How to use
1. Create a .env file in the root directory and add an OpenAI key, in this format:  
   ```OPENAI_API_KEY=insert key here```
2. Run ```load_games_from_csv.py```. The CSV contains game history, and a database called ```games.db``` will be created in the root directory for Nellie to use
3. Run ```main.py```. After a few seconds, Nellie will be ready to take in a query. Enter any query or :q to exit
