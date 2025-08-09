#%%
import os
openai_api_key = os.environ["OPENAI_API_KEY"]


# %%
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
# %%
llm = ChatOpenAI(openai_api_key=openai_api_key,temperature=0)

# %%
prompt = """
Dada la siguiente tabla:

<table>
  <tr>
    <th>Company</th>
    <th>Contact</th>
    <th>Country</th>
  </tr>
  <tr>
    <td>Alfreds Futterkiste</td>
    <td>Maria Anders</td>
    <td>Germany</td>
  </tr>
  <tr>
    <td>Centro comercial Moctezuma</td>
    <td>Francisco Chang</td>
    <td>Mexico</td>
  </tr>
</table>

Q: Quién es el contacto del Centro Comercial Moctezuma y en qué país está?
"""
llm.invoke(prompt)
# %%

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai.chat_models import ChatOpenAI
from langchain.memory import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory



prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're an assistant who's good at {ability}. Respond in 20 words or fewer",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


# %%
runnable = prompt | llm

# %%
from langchain_core.runnables.history import RunnableWithMessageHistory
# %%
chat_memory = ChatMessageHistory()

def get_session_history(session_id: str) -> BaseChatMessageHistory:
        return chat_memory

with_message_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
) 
# %%
with_message_history.invoke(
        {
        "ability": "Music",
        "question": "Is Vai better than Hendrix?"
    }, config={"configurable": {"session_id": "foo"}},       
)
# %%
with_message_history.invoke(
        {
        "ability": "Music",
        "question": "are you sure?"
    }, config={"configurable": {"session_id": "foo"}},       
)


# %%
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# %%
embeddings = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-3-small")
db = Chroma(collection_name="textos_pobres",
                embedding_function=embeddings,
                persist_directory="./docs")
# %%
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.create_documents(["Los zorros son marrones", "La bolognesa lleva carne picada", "Messi es el GOAT"])
# %%
texts
# %%
db.add_documents(texts)
# %%
retriever = db.as_retriever(search_kwargs={"k": 2})

# %%

retriever.invoke("Comida")
# %%
query = "Comida"
db.get()

# %%
db.
# %%
from langchain_core.output_parsers import StrOutputParser

# %%

total_chain = with_message_history | StrOutputParser()
# %%
total_chain.invoke(
        {
        "ability": "Music",
        "question": "are you sure?"
    }, config={"configurable": {"session_id": "foo"}},       
)
# %%
from langchain import hub

# %%
prompt = hub.pull("rlm/rag-prompt")


#%%
prompt 

# %%
