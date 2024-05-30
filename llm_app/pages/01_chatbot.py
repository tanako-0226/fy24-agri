import streamlit as st
from openai import AzureOpenAI
import faiss
import numpy as np
import pandas as pd
import os


AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
EMBEDDING_DEPLOY_NAME = os.getenv("EMBEDDING_DEPLOY_NAME")
GPT35TURBO_DEPLOY_NAME = os.getenv("GPT35TURBO_DEPLOY_NAME")

print("DEPLY_NAME", GPT35TURBO_DEPLOY_NAME)

client = AzureOpenAI(
    api_key=AZURE_API_KEY,  
    api_version="2024-02-01",
    azure_endpoint=AZURE_ENDPOINT
)


def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding


@st.cache_data
def read_data(filepath):
    """
    キャッシュをすることでリロードするたびにロードされるのを防ぐ関数
    """
    return pd.read_csv(filepath)


@st.cache_data
def make_index(df):
    """
    インデックスの作成
    """
    # インデックスの生成
    index = faiss.IndexFlatIP(1536)
    # 対象テキストの追加
    tmp = [i for i in df["answer_emb"].map(lambda x: [float(j) for j in x[1:-1].split(', ')]).values]
    index.add(np.array(tmp).astype('float32'))
    return index

# データフレーム
df = read_data("../answer_emb.csv")
index = make_index(df)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "なにかご質問はありますか?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])



if prompt := st.chat_input("質問してください"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    question_emb = get_embedding(st.session_state.messages[1:][-1]['content'])

    question_emb_np = np.array([question_emb]).astype('float32')

    retreval_num = 3
    _, I = index.search(question_emb_np, retreval_num)

    system_prompt = "以下の質問に答えてください。参考になる情報をいくつか提示しますので、それを踏まえて一言で回答を作成してください。\n"
    messages = [{"role": "system", "content": system_prompt}]  + st.session_state.messages[1:]

    for i in range(retreval_num):
            source  = df.loc[I[0][i]]
            ans = source.answer
            messages.append({'role': 'user', 'content': f'情報1:{ans}\n'})

    # st.dataframe(df.loc[I[0]])
    
    stream = client.chat.completions.create(model="gpt-4o", 
                                                    temperature=0.3,
                                                    max_tokens=3000,
                                                    messages=messages,
                                                    stream=True)

    response = st.chat_message("assistant").write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
