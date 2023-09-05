import os
from pathlib import Path
import chromadb
from langchain.llms.openai import OpenAIChat
from llama_index import download_loader
from llama_index import (
   VectorStoreIndex,
   GPTSimpleKeywordTableIndex,
   LLMPredictor,
   ServiceContext,
   StorageContext
)
from llama_index.vector_stores import ChromaVectorStore

os.environ["OPENAI_API_KEY"] = 'YOU-OPENAI-API-KEY-HERE'

# 从网页内读取文档，并以dict格式缓存
years = [2022, 2021, 2020, 2019]
UnstructuredReader = download_loader("UnstructuredReader", refresh_cache=True)
loader = UnstructuredReader()
doc_set = {}
all_docs = []
for year in years:
    year_docs = loader.load_data(
        file=Path(f"./data/UBER/UBER_{year}.html"), split_documents=False
    )
    # insert year metadata into each year
    for d in year_docs:
        d.metadata = {"year": year}
    doc_set[year] = year_docs
    all_docs.extend(year_docs)

# 这里是将document embedding序列化的过程，这部分使用轻量级的Chroma，后期可替换为Milvus
llm_predictor_chatgpt = LLMPredictor(llm=OpenAIChat(temperature=0, model_name="gpt-3.5-turbo"))
# store to chromadb
if not os.path.exists("./chroma_db"):
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.create_collection("uber_sec_collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor_chatgpt, chunk_size=512)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index_set = {}
    for year in years:
        cur_index = VectorStoreIndex.from_documents(
            documents=doc_set[year], id=year, storage_context=storage_context, service_context=service_context
        )

# load from chromadb
db2 = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db2.get_collection("uber_sec_collection")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor_chatgpt, chunk_size=512)
index_set = {}
index_summaries = {}
for year in years:
    cur_index = VectorStoreIndex.from_vector_store(
        id=year, vector_store=vector_store, service_context=service_context
    )
    index_set[year] = cur_index
    index_summaries[year] = f"UBER 10-k Filing for {year} fiscal year"


# Decomposable Querying over Documents
from llama_index.indices.composability import ComposableGraph
graph = ComposableGraph.from_indices(
    GPTSimpleKeywordTableIndex,
    [index for _, index in index_set.items()],
    [summary for _, summary in index_summaries.items()],
    max_keywords_per_chunk = 50
)

from llama_index.indices.query.query_transform.base import DecomposeQueryTransform
decompose_transform = DecomposeQueryTransform(
    llm_predictor_chatgpt, verbose=True
)


# define a decompose transform
from llama_index.query_engine.transform_query_engine import TransformQueryEngine

custom_query_engines = {}
for index in index_set.values():
    query_engine = index.as_query_engine(service_context=service_context)
    transform_extra_info = {'index_summary': index.index_struct.summary}
    transformed_query_engine = TransformQueryEngine(query_engine, decompose_transform, transform_metadata=transform_extra_info)
    custom_query_engines[index.index_id]=transformed_query_engine

# add the root of the decomposable graph
custom_query_engines[graph.root_index.index_id] = graph.root_index.as_query_engine(
    retriever='simple',
    response_mode='tree_summarize',
    service_context=service_context
)

query_engine_decompose = graph.as_query_engine(
    custom_query_engines=custom_query_engines
)

response_chat = query_engine_decompose.query(
    "What were some of the biggest risk factors in 2020 for Uber"
)

print(str(response_chat))