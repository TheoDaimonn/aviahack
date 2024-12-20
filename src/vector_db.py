from openai import OpenAI
from langchain_ollama.llms import OllamaLLM

import pandas as pd
import re
import os
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.docstore.document import Document
from markitdown import MarkItDown
import json
import configparser
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate
import langchain_openai
import httpx


prompt_template = """
# Отвечай только на русском

Твоя задача: написать 5 семантически схожих запросов, которые помогут получить наиболее релевантные документы в результате поиска.
Каждый вопрос должен освещать различные аспекты оригинального запроса.

Оригинальный запрос: {question}

Варианты запроса:
1. 
2. 
3. 
4.
5.
"""

prompt = PromptTemplate(
    input_variables=["question"],
    template=prompt_template
)


cfg = configparser.ConfigParser()
cfg.read('/home/ivzarru/hacks/aviahack/src/config.ini')
print(cfg.sections())
model_name = cfg['MAIN_INFO']['model_name']
llm_url = cfg['MAIN_INFO']['llm_url']

llm_client = OpenAI(
    base_url=llm_url,
    api_key="EMPTY"
)
llm_model = OllamaLLM(
    model="llama3",
    base_url=llm_url  
)
OUTPUT_DIR = 'output_docs'
INPUT_DIR = 'input_docs'

#model_name="deepvk/USER-bge-m3"


#model_kwargs = {'device': 'cuda:0'}
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


def parse_pdf():
    md = MarkItDown()
    files = [f for f in os.listdir(INPUT_DIR) if os.path.isfile(os.path.join(INPUT_DIR, f))]

    for i in files:
        result = md.convert(f"{INPUT_DIR}/{i}")
        with open(OUTPUT_DIR + '/' + i.replace('.pdf', '.md'), 'w') as f:
            f.write(result.text_content)


def chunk_split() -> pd.DataFrame:
    chunks = {}
    files = [f for f in os.listdir(OUTPUT_DIR) if os.path.isfile(os.path.join(OUTPUT_DIR, f))]
    for i in files:
        with open(OUTPUT_DIR + '/' + i, 'r') as file:
            chunks[i] = file.read().split('')[1::]
    rows = []
    for source_file, chunk in chunks.items():
        for chunk_text in chunk:
            rows.append({'source_file': source_file, 'chunk_text': chunk_text})

    df = pd.DataFrame(rows)
    return df



def preprocess_text(text):
    text = re.sub(r'[0-9\s]', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)  
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', ' ', text) 
    text = re.sub(r'\s+', ' ', text).strip()  
    
    tokens = text.lower().split()
    cleaned_text = ' '.join(tokens)

    return cleaned_text

def dataframe_preprocess(df):
    try:
        df = df.drop(columns=['Unnamed: 0'])
    except Exception:
        pass
    df = df.dropna().reset_index(drop=True)

    # очистка текста + дедубликация
    df['preprocessed_text'] = df['chunk_text'].apply(lambda x: preprocess_text(x))

    df['embedded_text'] = df['preprocessed_text'].apply(lambda x: model.embed_query(x))

    return df


def VB_build(df):
    sample_text = "Sample text."
    embedding = model.embed_query(sample_text)

    dimension = len(embedding)
    index = faiss.IndexFlatIP(dimension)
    
    df['embedded_text'] = df['embedded_text'].apply(lambda x: np.array(x))
    embeddings = np.stack(df['embedded_text'].tolist())
    faiss.normalize_L2(embeddings) 
    index.add(embeddings)

    documents = [
        Document(page_content=text, metadata={'source_file': sf})
        for text, sf in zip(df['chunk_text'], 
                                         df['source_file'])
    ]
    
    docstore = InMemoryDocstore(dict(enumerate(documents)))
    
    vector_store = FAISS(
        embedding_function=model,
        index=index,
        docstore=docstore,
        index_to_docstore_id=dict(zip(range(len(embeddings)), range(len(embeddings)))),
    )
    vector_store.save_local('vector_database/vector_database')
    df.to_csv('vector_database/docs_info.csv', index=False)

    return vector_store

def search_VB(query, k=10, threshold=0.87):
    
    vector_store = FAISS.load_local(
        'vector_database/vector_database',
        embeddings=model,
        allow_dangerous_deserialization=True
    )

    retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(), llm=llm_model, prompt=prompt
)
    results = retriever_from_llm.invoke(query)
    print(results)
    return results

def vb_increment(path_to_doc : str, path_vb : str):
    new_file_path = parse_pdf(path_to_doc)
    # Допилить
    
def vb_rebuild():
    parse_pdf()
    df = dataframe_preprocess(chunk_split())
    VB_build()


def dump_to_json(x: Document, i: int):
    x = x.model_dump()
    x['id'] = i
    return x

if not os.path.isdir('vector_database'):
    print('no VB')
    vb_rebuild()
search_VB('Что делать если поломался транспортер?')