from dotenv import load_dotenv
import json
import os 

import chromadb
import openai
import pandas as pd
import tkinter as tk
from tkinter import scrolledtext
import tkinter.filedialog as filedialog


load_dotenv()
openai.api_key = os.environ.get('OPENAI_KEY')

client = chromadb.PersistentClient()
collection = client.get_or_create_collection(
    name="kakao_channel",
    metadata={"hnsw:space": "cosine"}# l2 is the default
)

def extract_data_from_vector_db(query: str, limit=2):
    db_result = collection.query(
        query_texts=[query],
        n_results=limit,
    )
    return db_result


def send_message(message_log, functions, gpt_model="gpt-3.5-turbo", temperature=0.1):
    response = openai.ChatCompletion.create(
        model=gpt_model,
        messages=message_log,
        temperature=temperature,
        functions=functions,
        function_call='auto',
    )

    response_message = response["choices"][0]["message"]

    if response_message.get("function_call"):
        available_functions = {
            "extract_data_from_vector_db": extract_data_from_vector_db,
        }
        function_name = response_message["function_call"]["name"]
        fuction_to_call = available_functions[function_name]
        function_args = json.loads(response_message["function_call"]["arguments"])
        # 사용하는 함수에 따라 사용하는 인자의 개수와 내용이 달라질 수 있으므로
        # **function_args로 처리하기
        function_response = fuction_to_call(**function_args)

        # 함수를 실행한 결과를 GPT에게 보내 답을 받아오기 위한 부분
        message_log.append(response_message)  # GPT의 지난 답변을 message_logs에 추가하기
        message_log.append(
            {
                "role": "function",
                "name": function_name,
                "content": str(function_response),
            }
        )  # 함수 실행 결과도 GPT messages에 추가하기
        response = openai.ChatCompletion.create(
            model=gpt_model,
            messages=message_log,
            temperature=temperature,
        )  # 함수 실행 결과를 GPT에 보내 새로운 답변 받아오기
    return response.choices[0].message.content


def extract_data_into_sections(text_path):
    structured_data = {}
    current_section_title = None
    with open(text_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line.startswith('#'):  # Section title
                current_section_title = line[1:].strip()
                structured_data[current_section_title] = ""
            else:  # Section content
                if current_section_title:
                    structured_data[current_section_title] += line + " "
    return structured_data

def add_data_into_vector_db(data):
    ids = []
    documents = []

    for key, item in data.items():
        document = f"{key}: {item}"
        ids.append(key)
        documents.append(document)

    collection.add(
        documents=documents,
        ids=ids
    )


def main():
    file_name = '/Users/selene/Workspace/llm-course/selene.p/project_1/data/project_data_카카오톡채널.txt'
    data = extract_data_into_sections(file_name)

    add_data_into_vector_db(data)

    message_log = [
        {
            "role": "system",
            "content": '''
            너는 카카오 서비스 챗봇이야.
            유저가 어떤 말을 하면, 그에 맞게 assistant가 되어서 답변해야해.
            챗봇처럼 답변 해줘.
            너가 필요한 지식은 vector db로부터 얻을 수 있어.
            봇인 만큼 간략하고 정확하게 답변해주고, 불필요한 정보는 제거해야해.
            '''
        }
    ]

    functions = [
        {
            "name": "extract_data_from_vector_db",
            "description": "vector db로 부터 query로 요청한 카카오 채널 API 소개 문서를 추출합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "추출하고자 하는 카카오 채널 소개 문서 내용",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "추출하고자 하는 n가지 결과",
                    }
                },
                "required": ["query"],
            },
        }
    ]

    def show_popup_message(window, message):
        popup = tk.Toplevel(window)
        popup.title("")

        # 팝업 창의 내용
        label = tk.Label(popup, text=message, font=("맑은 고딕", 12))
        label.pack(expand=True, fill=tk.BOTH)

        # 팝업 창의 크기 조절하기
        window.update_idletasks()
        popup_width = label.winfo_reqwidth() + 20
        popup_height = label.winfo_reqheight() + 20
        popup.geometry(f"{popup_width}x{popup_height}")

        # 팝업 창의 중앙에 위치하기
        window_x = window.winfo_x()
        window_y = window.winfo_y()
        window_width = window.winfo_width()
        window_height = window.winfo_height()

        popup_x = window_x + window_width // 2 - popup_width // 2
        popup_y = window_y + window_height // 2 - popup_height // 2
        popup.geometry(f"+{popup_x}+{popup_y}")

        popup.transient(window)
        popup.attributes('-topmost', True)

        popup.update()
        return popup

    def on_send():
        user_input = user_entry.get()
        user_entry.delete(0, tk.END)

        if user_input.lower() == "quit":
            window.destroy()
            return

        message_log.append({"role": "user", "content": user_input})
        conversation.config(state=tk.NORMAL)  # 이동
        conversation.insert(tk.END, f"You: {user_input}\n", "user")  # 이동
        thinking_popup = show_popup_message(window, "처리중...")
        window.update_idletasks()
        # '생각 중...' 팝업 창이 반드시 화면에 나타나도록 강제로 설정하기
        response = send_message(message_log, functions)
        thinking_popup.destroy()

        message_log.append({"role": "assistant", "content": response})

        # 태그를 추가한 부분(1)
        conversation.insert(tk.END, f"gpt assistant: {response}\n", "assistant")
        conversation.config(state=tk.DISABLED)
        # conversation을 수정하지 못하게 설정하기
        conversation.see(tk.END)

    window = tk.Tk()
    window.title("GPT AI")

    font = ("맑은 고딕", 10)

    conversation = scrolledtext.ScrolledText(window, wrap=tk.WORD, bg='#f0f0f0', font=font)
    conversation.insert(tk.END, "gpt assistant: 저는 카카오 챗봇입니다\n", "assistant")

    # width, height를 없애고 배경색 지정하기(2)
    conversation.tag_configure("user", background="#c9daf8")
    # 태그별로 다르게 배경색 지정하기(3)
    conversation.tag_configure("assistant", background="#e4e4e4")
    # 태그별로 다르게 배경색 지정하기(3)
    conversation.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    # 창의 폭에 맞추어 크기 조정하기(4)

    input_frame = tk.Frame(window)  # user_entry와 send_button을 담는 frame(5)
    input_frame.pack(fill=tk.X, padx=10, pady=10)  # 창의 크기에 맞추어 조절하기(5)

    user_entry = tk.Entry(input_frame)
    user_entry.pack(fill=tk.X, side=tk.LEFT, expand=True)

    send_button = tk.Button(input_frame, text="Send", command=on_send)
    send_button.pack(side=tk.RIGHT)

    window.bind('<Return>', lambda event: on_send())
    window.mainloop()


if __name__ == "__main__":
    main()
