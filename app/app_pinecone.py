import os

import chainlit as cl
import pinecone
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig

# Load environment variables from .env file
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # type: ignore
    environment=PINECONE_ENV,  # type: ignore
)

#MODEL_PATH = "./../model/llama-2-7b-chat.ggmlv3.q8_0.bin"
loaded_llm = ChatOpenAI(model="gpt-3.5-turbo",streaming=True)

index = pinecone.Index("vivekandarag")

prompt_template = """You are a Swami Vivekananda and give the answer to query but don't give answers that are beyond your domain and timeline and out of the context but complete the sentence in meaningful manner.Use the following piece of context to answer the question.response should be in english but whatever indic words are there in response replace them with their devnagri counterpart and english words into english alphabets for ex Bhakti Yoga should be भक्ति योग take context from revelent documents.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""


#system_template = """You are a Swami Vivekananda and give the answer to query but don't give answers that are beyond your domain and timeline and out of the context but complete the sentence in meaningful manner.Use the following piece of context to answer the question.response should be in english but whatever indic words are there in response replace them with their devnagri counterpart and english words into english alphabets for ex Bhakti Yoga should be भक्ति योग take context from revelent documents."""


#messages = [
    #SystemMessagePromptTemplate.from_template(system_template),
    #HumanMessagePromptTemplate.from_template("{query}"),
#]
#prompt = ChatPromptTemplate.from_messages(messages)
#chain_type_kwargs = {"prompt": prompt}

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    return prompt


def create_retrieval_qa_chain(llm,prompt, db):
    """
    Creates a Retrieval Question-Answering (QA) chain using a given language model, prompt, and database.

    This function initializes a RetrievalQA object with a specific chain type and configurations,
    and returns this QA chain. The retriever is set up to return the top 3 results (k=3).

    Args:
        llm (any): The language model to be used in the RetrievalQA.
        prompt (str): The prompt to be used in the chain type.
        db (any): The database to be used as the retriever.

    Returns:
        RetrievalQA: The initialized QA chain.
    """
    qa_chain = RetrievalQA.from_chain_type(
        llm=loaded_llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain


#def load_model(
    #model_path=MODEL_PATH,
    #model_type="llama",
    #max_new_tokens=512,
    #temperature=0.7,
#):
   # """
    #Load a locally downloaded model.

    #Parameters:
        #model_path (str): The path to the model to be loaded.
        #model_type (str): The type of the model.
        #max_new_tokens (int): The maximum number of new tokens for the model.
        #temperature (float): The temperature parameter for the model.

    #Returns:
        #CTransformers: The loaded model.

    #Raises:
        #FileNotFoundError: If the model file does not exist.
        #SomeOtherException: If the model file is corrupt.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model file found at {model_path}")

    # Additional error handling could be added here for corrupt files, etc.

    llm = CTransformers(
        model=model_path,
        model_type=model_type,
        max_new_tokens=max_new_tokens,  # type: ignore
        temperature=temperature,  # type: ignore
    )

    return llm"""


def create_retrieval_qa_bot(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    index_name="vivekandarag",
    device="cpu"
):
    """
    This function creates a retrieval-based question-answering bot.

    Parameters:
        model_name (str): The name of the model to be used for embeddings.
        persist_dir (str): The directory to persist the database.
        device (str): The device to run the model on (e.g., 'cpu', 'cuda').

    Returns:
        RetrievalQA: The retrieval-based question-answering bot.

    Raises:
        FileNotFoundError: If the persist directory does not exist.
        SomeOtherException: If there is an issue with loading the embeddings or the model.
    """

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
        )
    except Exception as e:
        raise Exception(
            f"Failed to load embeddings with model name {model_name}: {str(e)}"
        )

    db = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)

    try:
        llm = loaded_llm # Assuming this function exists and works as expected
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")

    qa_prompt = (
        set_custom_prompt()
    )  # Assuming this function exists and works as expected

    try:
        qa = create_retrieval_qa_chain(
            llm=llm, db=db, prompt=qa_prompt
        )  # Assuming this function exists and works as expected
    except Exception as e:
        raise Exception(f"Failed to create retrieval QA chain: {str(e)}")

    return qa


def retrieve_bot_answer(query):
    """
    Retrieves the answer to a given query using a QA bot.

    This function creates an instance of a QA bot, passes the query to it,
    and returns the bot's response.

    Args:
        query (str): The question to be answered by the QA bot.

    Returns:
        dict: The QA bot's response, typically a dictionary with response details.
    """
    qa_bot_instance = create_retrieval_qa_bot()
    bot_response = qa_bot_instance({"query": query})
    return bot_response


@cl.on_chat_start
async def initialize_bot():
    """
    Initializes the bot when a new chat starts.

    This asynchronous function creates a new instance of the retrieval QA bot,
    sends a welcome message, and stores the bot instance in the user's session.
    """
    qa_chain = create_retrieval_qa_bot()
    welcome_message = cl.Message(content="Starting the bot...")
    await welcome_message.send()
    welcome_message.content = (
        "Hi, Welcome to Chat With Documents using Llama2 and LangChain."
    )
    await welcome_message.update()
    runnable = set_custom_prompt() | loaded_llm | StrOutputParser() |create_retrieval_qa_bot()
    #cl.user_session.set("chain", qa_chain)
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def process_chat_message(message :cl.Message):
    """
    Processes incoming chat messages.

    This asynchronous function retrieves the QA bot instance from the user's session,
    sets up a callback handler for the bot's response, and executes the bot's
    call method with the given message and callback. The bot's answer and source
    documents are then extracted from the response.
    """
    #qa_chain = cl.user_session.get("chain")
    #print(qa_chain)
    runnable =cl.user_session.get("runnable")
    msg =cl.Message(content=" ")

    # callback_handler = cl.AsyncLangchainCallbackHandler(
    #     stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    # )
    # callback_handler.answer_reached = True
    # response = await qa_chain.acall(message, callbacks=[callback_handler])
    # bot_answer = response["result"]
    # source_documents = response["source_documents"]

    # if source_documents:
    #     bot_answer += f"\nSources:" + str(source_documents)
    # else:
    #     bot_answer += "\nNo sources found"

    # await cl.Message(content=bot_answer).send()

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
