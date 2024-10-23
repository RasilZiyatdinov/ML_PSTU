import os
import urllib.request

def download_file(file_link, filename):
    if not os.path.isfile(filename):
        urllib.request.urlretrieve(file_link, filename)
        print("File downloaded successfully.")
    else:
        print("File already exists.")

ggml_model_path = "https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q4_0.gguf"
filename = "zephyr-7b-beta.Q4_0.gguf"

download_file(ggml_model_path, filename)

from llama_cpp import Llama

llm = Llama(model_path="zephyr-7b-beta.Q4_0.gguf", n_ctx=5120, n_batch=126)

def generate_text(
    prompt="Who is the CEO of Apple?",
    max_tokens=256,
    temperature=0.1,
    top_p=0.5,
    echo=False,
    stop=["#"],
):
    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        echo=echo,
        stop=stop,
    )
    output_text = output["choices"][0]["text"].strip()
    return output_text


def generate_prompt_from_template(input):
    chat_prompt_template = f"""<|im_start|>system
You are a helpful chatbot.<|im_end|>
<|im_start|>user
{input}<|im_end|>"""
    return chat_prompt_template

import requests
from bs4 import BeautifulSoup

# 1. Попросить модель пересказать для ребенка содержание эссе The Bitter Lesson на английском языке
url = 'http://www.incompleteideas.net/IncIdeas/BitterLesson.html'
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')
text = soup.get_text()

query = "Retell for a child the follow article: {article}".format(article = text)
prompt = generate_prompt_from_template(query)
print(generate_text(
    prompt,
    max_tokens=5120,
))

# 2. Попросить перевести указанное эссе на русский язык, оценить перевод
query = "Translate into rusuian the follow article: {article}".format(article = text)
prompt = generate_prompt_from_template(query)
print(generate_text(
    prompt,
    max_tokens=5120,
))