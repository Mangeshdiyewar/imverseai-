from openai import AsyncOpenAI
import chainlit as cl
from dotenv import load_dotenv
import os
from langchain.embeddings import HuggingFaceEmbeddings
import pinecone
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Pinecone
import time
# Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
index_name="vivekandarag"
model_name = "sentence-transformers/all-MiniLM-L6-v2"
device="cpu"
embeddings = HuggingFaceEmbeddings(model_name=model_name,
                             model_kwargs={"device": device},
                            )

if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        index_name,
        dimension=len(embeddings[0]),
        metric='cosine'
    )
    # wait for index to finish initialization
    while not pinecone.describe_index(index_name).status['ready']:
        time.sleep(1)

index = pinecone.Index(index_name)
index.describe_index_stats()


doc_search_existing = Pinecone.from_existing_index(index_name, embeddings)

docs=doc_search_existing.similarity_search(query,k=2)



client = AsyncOpenAI(api_key=OPENAI_API_KEY)


settings = {
    "model": "gpt-3.5-turbo",
    "temperature": 0.7,
    "max_tokens": 500,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}


def context(query):
  contextual=[]
  docs=doc_search_existing.similarity_search(query,k=3)
  for i in range(len(docs)):
    contextual.append(docs[i].page_content)
  return ' '.join(contextual)

@cl.on_chat_start
def start_chat():
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": f"You are a Swami Vivekananda and give the answer to query but don't give answers that are beyond your domain and timeline and out of the context but complete the sentence in meaningful manner.Use the following piece of context to answer the question.response should be in english but whatever indic words are there in response replace them with their devnagri counterpart and english words into english alphabets for ex Bhakti Yoga should be भक्ति योग and so on. context:{info.strip()}"},
         ],
    )


@cl.on_message
async def main(message: cl.Message):
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": message.content})
    query = message.content

    msg = cl.Message(content="")
    await msg.send()

    stream = await client.chat.completions.create(
        messages=message_history, stream=True, **settings
    )

    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await msg.stream_token(token)

    message_history.append({"role": "assistant", "content": msg.content})
    await msg.update()