from dto import ChatbotRequest
import logging
from dotenv import load_dotenv
import os

import chromadb

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper


load_dotenv()
logger = logging.getLogger("Callback")

SEARCH = GoogleSearchAPIWrapper(
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
    google_cse_id=os.environ.get("GOOGLE_CSE_ID"),
)

CUR_DIR = os.getcwd()
HISTORY_DIR = os.path.join(CUR_DIR, "chat_histories")
SEARCH_TOOL = Tool(
    name="Google Search",
    description="Search Google for recent results.",
    func=SEARCH.run,
)

def read_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
    return data

def create_chain(llm, template_path, output_key):
    return LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_template(
            template=read_file(template_path)
        ),
        output_key=output_key,
        verbose=True,
    )

INPUT_STEP1_PROMPT_TEMPLATE = os.path.join(CUR_DIR, "prompts/input_analyze.txt")
INPUT_STEP2_PROMPT_TEMPLATE = os.path.join(CUR_DIR, "prompts/input_solution.txt")
DEFAULT_RESPONSE_PROMPT_TEMPLATE = os.path.join(CUR_DIR, "prompts/default_response.txt")
INTENT_PROMPT_TEMPLATE = os.path.join(CUR_DIR, "prompts/parse_intent.txt")
SEARCH_VALUE_CHECK_PROMPT_TEMPLATE = os.path.join(CUR_DIR, "prompts/search_value_check.txt")
SEARCH_COMPRESSION_PROMPT_TEMPLATE = os.path.join(CUR_DIR, "prompts/search_compress.txt")
INTENT_LIST_TXT = os.path.join(CUR_DIR, "prompts/intent_list.txt")

llm = ChatOpenAI(temperature=1, max_tokens=200, model="gpt-3.5-turbo")

input_step1_chain = create_chain(
    llm=llm,
    template_path=INPUT_STEP1_PROMPT_TEMPLATE,
    output_key="input_analysis",
)
input_step2_chain = create_chain(
    llm=llm,
    template_path=INPUT_STEP2_PROMPT_TEMPLATE,
    output_key="output",
)
parse_intent_chain = create_chain(
    llm=llm,
    template_path=INTENT_PROMPT_TEMPLATE,
    output_key="intent",
)
default_chain = create_chain(
    llm=llm, template_path=DEFAULT_RESPONSE_PROMPT_TEMPLATE, output_key="output"
)

search_value_check_chain = create_chain(
    llm=llm,
    template_path=SEARCH_VALUE_CHECK_PROMPT_TEMPLATE,
    output_key="output",
)
search_compression_chain = create_chain(
    llm=llm,
    template_path=SEARCH_COMPRESSION_PROMPT_TEMPLATE,
    output_key="output",
)

def extract_data_into_sections(text_path):
    structured_data = {}
    current_section_title = None
    with open(text_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line.startswith('#'):
                current_section_title = line[1:].strip()
                structured_data[current_section_title] = ""
            else:
                if current_section_title:
                    structured_data[current_section_title] += line + " "
    return structured_data

def set_vector_db():
    files = {
        "kakao_sync": "project_data_카카오싱크",
        "kakao_social": "project_data_카카오소셜",
        "kakao_channel": "project_data_카카오톡채널"
    }

    for collection_name, file_name in files.items():
        file_data = extract_data_into_sections(os.path.join(CUR_DIR, f"data/{file_name}.txt"))

        ids = []
        documents = []

        for key, item in file_data.items():
            document = f"{key}: {item}"
            ids.append(key)
            documents.append(document)

        client = chromadb.PersistentClient()
        client.delete_collection(name=collection_name)
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        collection.add(
            documents=documents,
            ids=ids
        )

def query_db(query: str, collection_name: str):
    client = chromadb.PersistentClient()
    db_collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    query_result = db_collection.query(
        query_texts=[query],
        n_results=2
    )

    results = []
    for document in query_result['documents'][0]:
        docs_extracted = document.split(':')
        results.append(
            {
                "content": docs_extracted[1]
            }
        )
    return results

def query_web_search(user_message: str) -> str:
    context = {"user_message": user_message}
    context["related_web_search_results"] = SEARCH_TOOL.run(user_message)

    has_value = search_value_check_chain.run(context)

    if has_value == "Y":
        return search_compression_chain.run(context)
    else:
        return ""


def load_conversation_history(conversation_id: str):
    file_path = os.path.join(HISTORY_DIR, f"{conversation_id}.json")
    return FileChatMessageHistory(file_path)


def log_user_message(history: FileChatMessageHistory, user_message: str):
    history.add_user_message(user_message)


def log_bot_message(history: FileChatMessageHistory, bot_message: str):
    history.add_ai_message(bot_message)


def get_chat_history(conversation_id: str):
    history = load_conversation_history(conversation_id)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="user_message",
        chat_memory=history,
    )

    return memory.buffer


def generate_answer(user_message, conversation_id: str='fa1010') -> dict[str, str]:
    history_file = load_conversation_history(conversation_id)

    context = dict(user_message=user_message)
    context["input"] = context["user_message"]
    context["intent_list"] = read_file(INTENT_LIST_TXT)
    context["chat_history"] = get_chat_history(conversation_id)

    intent = parse_intent_chain.run(context)

    if intent == "kakao_sync" or intent == "kakao_social" or intent == "kakao_channel":
        context["related_documents"] = query_db(context["user_message"], intent)
        answer = ""
        for step in [input_step1_chain, input_step2_chain]:
            context = step(context)
            answer += context[step.output_key]
            answer += "\n\n"
    else:
        context["compressed_web_search_results"] = query_web_search(
            context["user_message"]
        )
        answer = default_chain.run(context)

    log_user_message(history_file, user_message)
    log_bot_message(history_file, answer)
    return {"answer": answer}

def callback_handler(request: ChatbotRequest) -> dict:
    set_vector_db()
    return generate_answer(request.userRequest.utterance)
