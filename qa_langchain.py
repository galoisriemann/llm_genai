__author__ = "Biswa Sengupta"

import os
from pprint import pprint
from typing import List

import numpy as np
import ray
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from ray import serve
from starlette.requests import Request

from embeddings import LocalHuggingFaceEmbeddings
from utils import read_yaml, conditional_decorator, timeit

perform_serial_embedding = True
params = read_yaml("llm_params.yaml")


class LLMDocumentsChain():
    def __init__(self, params, perform_embedding):
        self.params = params
        if self.params.get('langchain').get('RAY'):
            ray.init()
            self.shards = params.get('general').get('DB_SHARDS')
        self.perform_embedding = perform_embedding
        self.doc_dir = params.get('general').get('doc_dir')
        self.embedding_model = params.get('langchain').get('EMBEDDING_MODEL')
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=params.get('langchain').get('CHUNK_SIZE'),
                                                            chunk_overlap=params.get('langchain').get('CHUNK_OVERLAP'),
                                                            length_function=len, )

    def parseDocuments(self):
        loader = DirectoryLoader(self.doc_dir)
        docs = loader.load()
        chunks = self.text_splitter.create_documents([doc.page_content for doc in docs],
                                                     metadatas=[doc.metadata for doc in docs])
        return chunks

    @ray.remote(num_gpus=params.get('general').get('NUM_GPUS'))
    def process_shard(self, typ):
        embeddings = LocalHuggingFaceEmbeddings(self.embedding_model)
        result = FAISS.from_documents(typ, embeddings)
        return result

    # @timeit
    def inject_docs(self, chunks):

        if not self.params.get('langchain').get('RAY'):
            embeddings = LocalHuggingFaceEmbeddings(self.embedding_model)
            db = FAISS.from_documents(chunks, embeddings)
        else:
            print(f'Loading chunks into vector store ... using {self.shards} shards')
            shards = np.array_split(chunks, self.shards)
            futures = [self.process_shard.remote(self, shards[i]) for i in range(self.shards)]
            results = ray.get(futures)

            print('Merging shards ...')
            # Straight serial merge of others into results[0]
            db = results[0]
            try:
                for i in range(1, self.shards):
                    db.merge_from(results[i])
            except NotImplementedError as e:
                print(f'NotImplementedError: IndexFlatL2 does not support merging. Hang in there ...')

        try:
            os.remove(self.params.get('langchain').get('DOC_INDEX_PATH'))
        except OSError:
            db.save_local(self.params.get('langchain').get('DOC_INDEX_PATH'))


# @serve.deployment
@conditional_decorator(serve.deployment, params.get('langchain').get('RAY'))
class LLMDocumentsChainServe():
    def __init__(self, params):
        self.params = params
        self.embeddings = LocalHuggingFaceEmbeddings(params.get('langchain').get('EMBEDDING_MODEL'))
        self.db = FAISS.load_local(params.get('langchain').get('DOC_INDEX_PATH'), self.embeddings)

    def search(self, query):
        results = self.db.max_marginal_relevance_search(query)
        retval = ""
        for i in range(len(results)):
            chunk = results[i]
            source = chunk.metadata["source"]
            retval = retval + f"From http://{source}\n\n"
            retval = retval + chunk.page_content
            retval = retval + "\n====\n\n"

        return retval

    async def __call__(self, request: Request) -> List[str]:
        return self.search(request.query_params["text"])


def nsLLM():
    q = "What is the relationship between Tyrion and Lannisters?"

    if perform_serial_embedding:
        lchain_bot = LLMDocumentsChain(params, perform_serial_embedding)
        chunks = lchain_bot.parseDocuments()
        lchain_bot.inject_docs(chunks)
        output = lchain_bot
    else:
        if params.get('langchain').get('RAY'):
            lchain_bot_serve = LLMDocumentsChainServe.bind(params)
            handle = serve.run(lchain_bot_serve)
            # print(ray.get(handle.remote()))
            output = handle
        else:
            lchain_bot_serve = LLMDocumentsChainServe(params)
            output = lchain_bot_serve.search(q)
    return output


output = nsLLM()
pprint(output)
