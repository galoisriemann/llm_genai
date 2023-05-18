import os
from pprint import pprint

from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack.nodes import FARMReader
from haystack.nodes import PromptNode, PromptTemplate
from haystack.pipelines import ExtractiveQAPipeline
from haystack.pipelines.standard_pipelines import TextIndexingPipeline
from utils import read_yaml

from haystack import Pipeline

# TODO get GPT3.5 to write the docstrings for the functions
perform_embedding = False


class LLMDocumentsHay():
    def __init__(self, params, perform_embedding):
        self.params = params
        self.doc_dir = params.get('general').get('doc_dir')
        self.top_k_reader = params.get('haystack').get('TOP_K_READER')
        self.top_k_retriever = params.get('haystack').get('TOP_K_RETRIEVER')
        if perform_embedding:
            try:
                self.document_store = FAISSDocumentStore(
                    faiss_index_factory_str=params.get('haystack').get('FAISS_INDEX'),
                    sql_url=params.get('haystack').get('SQL_URL'))
            except ValueError as ve:
                print(ve)
                print("Try running the script with perform_embedding=False")
        else:
            self.document_store = FAISSDocumentStore.load(index_path=params.get('haystack').get('DOC_INDEX_PATH'),
                                                          config_path=params.get('haystack').get('INDEX_CONFIG_PATH'))
        self.embedding_model = params.get('haystack').get('EMBEDDING_MODEL')
        self.retriever = EmbeddingRetriever(document_store=self.document_store, embedding_model=self.embedding_model)
        self.reader = FARMReader(model_name_or_path=params.get('haystack').get('READER_MODEL'),
                                 use_gpu=params.get('haystack').get('USE_GPU'))
        self.pipe_1 = ExtractiveQAPipeline(self.reader, self.retriever)

    def parseDocuments(self):
        files_to_index = [self.doc_dir + "/" + f for f in os.listdir(self.doc_dir)]
        indexing_pipeline = TextIndexingPipeline(self.document_store)
        indexing_pipeline.run_batch(file_paths=files_to_index)

    def embedDocuments(self, params):
        self.document_store.update_embeddings(self.retriever)
        self.document_store.save(index_path=params.get('haystack').get('DOC_INDEX_PATH'),
                                 config_path=params.get('haystack').get('INDEX_CONFIG_PATH'))

    def getAnswer1(self, retrieved_answers, query):
        pipe_2 = Pipeline()
        if self.params.get('general').get('OPENAI'):
            lfqa_prompt = PromptTemplate(name="lfqa",
                                         prompt_text="""Synthesize a comprehensive answer from the following topk most relevant paragraphs and the given question. Provide a clear and concise response that summarizes the key points and information presented in the paragraphs. Your answer should be in your own words and be no longer than 50 words. \n\n Paragraphs: {join(documents)} \n\n Question: {query} \n\n Answer:""")
            node = PromptNode(self.params.get('openai').get('MODEL_TYPE'), default_prompt_template=lfqa_prompt,
                              api_key=self.params.get('openai').get('API_KEY'))
        else:
            lfqa_prompt = PromptTemplate(
                name="lfqa",
                prompt_text="""Synthesize a comprehensive answer from the following text for the given question. 
                                         Provide a clear and concise response that summarizes the key points and information presented in the text. 
                                         Your answer should be in your own words and be no longer than 50 words. 
                                         \n\n Related text: {join(documents)} \n\n Question: {query} \n\n Answer:""",
            )

            node = PromptNode(model_name_or_path=self.params.get('open_source').get('MODEL_TYPE'),
                              default_prompt_template=lfqa_prompt)
        pipe_2.add_node(component=node, name="prompt_node", inputs=["Query"])
        output = pipe_2.run(query=query, documents=retrieved_answers.get('documents'))
        return output

    def getAnswer2(self, query):
        pipe = Pipeline()
        lfqa_prompt = PromptTemplate(
            name="lfqa",
            prompt_text="""Synthesize a comprehensive answer from the following text for the given question. 
                                     Provide a clear and concise response that summarizes the key points and information presented in the text. 
                                     Your answer should be in your own words and be no longer than 50 words. 
                                     \n\n Related text: {join(documents)} \n\n Question: {query} \n\n Answer:""",
        )

        prompt_node = PromptNode(model_name_or_path="declare-lab/flan-alpaca-large",
                                 default_prompt_template=lfqa_prompt)

        pipe.add_node(component=self.retriever, name="retriever", inputs=["Query"])
        pipe.add_node(component=prompt_node, name="prompt_node", inputs=["retriever"])
        output = pipe.run(query=query)

        print(output["results"])

        return output


def nsLLM():
    params = read_yaml("llm_params.yaml")

    if perform_embedding:
        qa_bot = LLMDocumentsHay(params, perform_embedding)
        qa_bot.parseDocuments()
        qa_bot.embedDocuments(params)
    else:
        qa_bot = LLMDocumentsHay(params, perform_embedding)

    q = "What is the relationship between Tyrion and Lannisters?"
    prediction = qa_bot.pipe_1.run(
        query=q,
        params={"Retriever": {"top_k": params.get('haystack').get('TOP_K_RETRIEVER')},
                "Reader": {"top_k": params.get('haystack').get('TOP_K_READER')}}
    )

    pprint(prediction)

    # r = qa_bot.getAnswer1(prediction, q)
    r = qa_bot.getAnswer2(q)

    pprint(r.get('results'))


nsLLM()
