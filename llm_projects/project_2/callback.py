from dto import ChatbotRequest
import aiohttp
import logging
from dotenv import load_dotenv
import os

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

# 환경 변수 처리 필요!
load_dotenv()
SYSTEM_MSG = "당신은 카카오 서비스 제공자입니다."
logger = logging.getLogger("Callback")

async def callback_handler(request: ChatbotRequest) -> dict:

    def read_file(file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        return prompt_template

    def create_chain(llm, template_path, output_key):
        return LLMChain(
            llm=llm,
            prompt=ChatPromptTemplate.from_template(
                template=read_file(template_path),
            ),
            output_key=output_key,
            verbose=True,
        )

    # ===================== start =================================
    # 카카오 싱크 관련 data load
    docs = read_file(os.path.join(os.getcwd(), 'data/project_data_카카오싱크.txt'))

    # llm, prompt 로드
    llm = ChatOpenAI(temperature=1, max_tokens=2000, model="gpt-3.5-turbo-16k")
    answer_prompt_path = os.path.join(os.getcwd(), 'data/answer_prompt.txt')
    score_prompt_path = os.path.join(os.getcwd(), 'data/score_prompt.txt')
    reanswer_prompt_path = os.path.join(os.getcwd(), 'data/reanswer_prompt.txt')

    # chain steps
    step_1_chain = create_chain(llm, answer_prompt_path, "answer")

    score_chain = create_chain(llm, score_prompt_path, "score")
    reanswer_chain = create_chain(llm, reanswer_prompt_path, "output")

    preprocess_chain = SequentialChain(
      chains=[
          step_1_chain,
          score_chain,
      ],
      input_variables=["docs", "question"],
      output_variables=["answer", "score"],
      verbose=True,
  )

    context = dict(
        docs=docs,
        question=request.userRequest.utterance,
    )
    context = preprocess_chain(context)

    context["answers"] = [context["answer"]]
    # score & reanswer
    for _ in range(3):
        if float(context["score"]) >= 8.0:
            break
        context = reanswer_chain(context)
        context["answers"].append(context["output"])
        context["answer"] = context["output"]
        context = score_chain(context)

    output_text = context["answer"]

   # 참고링크 통해 payload 구조 확인 가능
    payload = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": output_text
                    }
                }
            ]
        }
    }
    # ===================== end =================================
    # 참고링크1 : https://kakaobusiness.gitbook.io/main/tool/chatbot/skill_guide/ai_chatbot_callback_guide
    # 참고링크1 : https://kakaobusiness.gitbook.io/main/tool/chatbot/skill_guide/answer_json_format

    url = request.userRequest.callbackUrl
    if url:
        async with aiohttp.ClientSession() as session:
            async with session.post(url=url, json=payload, ssl=False) as resp:
                await resp.json()