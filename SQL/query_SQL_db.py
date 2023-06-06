__author__ = "Biswa Sengupta"

from langchain import OpenAI
import logging
import sys
import torch
from llama_index import (
    GPTSQLStructStoreIndex,
    SQLDatabase,
    LLMPredictor,
    ServiceContext,
    GPTVectorStoreIndex,
    PromptHelper,
)
from llama_index.indices.struct_store import SQLContextContainerBuilder
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    select,
    column,
    Integer,
    insert,
)

from utils import read_yaml, CustomLLM, create_prompt_helper
import os
from langchain.llms.base import LLM
from transformers import pipeline
from typing import Optional, List, Mapping, Any

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

params = read_yaml("sql.yml")

os.environ["OPENAI_API_KEY"] = params.get("openai").get("API_KEY")

if params.get("general").get("OPEN_AI"):
    llm_predictor = LLMPredictor(
        llm=OpenAI(temperature=0, model_name=params.get("openai").get("MODEL_TYPE"))
    )
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
else:
    # define our LLM
    llm_predictor = LLMPredictor(llm=CustomLLM())
    prompt_helper = create_prompt_helper()
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )


def create_dummy_data():
    engine = create_engine("sqlite:///:memory:")
    # metadata_obj = MetaData(bind=engine)
    metadata_obj = MetaData()
    # metadata_obj.bind = engine

    # create city SQL table
    table_name = params.get("general").get("table_name")
    city_stats_table = Table(
        table_name,
        metadata_obj,
        Column("city_name", String(16), primary_key=True),
        Column("population", Integer),
        Column("country", String(16), nullable=False),
    )
    metadata_obj.create_all(bind=engine)
    # print tables
    metadata_obj.tables.keys()

    # insert rows into city table
    rows = [
        {"city_name": "Toronto", "population": 2731571, "country": "Canada"},
        {"city_name": "Tokyo", "population": 13929286, "country": "Japan"},
        {"city_name": "Berlin", "population": 600000, "country": "Germany"},
    ]
    for row in rows:
        stmt = insert(city_stats_table).values(**row)
        with engine.connect() as connection:
            cursor = connection.execute(stmt)
            connection.commit()

    # view current table
    stmt = select(column("city_name")).select_from(city_stats_table)
    with engine.connect() as connection:
        results = connection.execute(stmt).fetchall()
        print(results)

    return engine, table_name, city_stats_table, cursor


def build_index(engine, table_name, context_vector, service_context):
    # create SQLDatabase object and the index
    sql_database = SQLDatabase(engine, include_tables=[table_name])
    if not params.get("general").get("CONTEXT"):
        index = GPTSQLStructStoreIndex(
            [],
            sql_database=sql_database,
            table_name=table_name,
            service_context=service_context,
        )
        table_schema_index = None
    else:
        if not params.get("general").get("AUTO_CONTEXT"):
            # create SQLContextContainer object
            table_context_dict = {"city_stats": context_vector}
            context_builder = SQLContextContainerBuilder(
                sql_database, context_dict=table_context_dict
            )
            context_container = context_builder.build_context_container()

            # building the index
            index = GPTSQLStructStoreIndex(
                [],
                sql_database=sql_database,
                table_name="city_stats",
                sql_context_container=context_container,
            )
        else:
            # build a vector index from the table schema information
            context_builder = SQLContextContainerBuilder(sql_database)
            table_schema_index = context_builder.derive_index_from_context(
                GPTVectorStoreIndex, store_index=True
            )
    return index, table_schema_index


def txt2sql(query, index, context_container=None):
    if not params.get("general").get("RETURN_CONTEXT"):
        query_engine = index.as_query_engine()
        response = query_engine.query(query)
        print(response)
    else:
        # query the table schema index using the helper method
        # to retrieve table context
        SQLContextContainerBuilder.query_index_for_context(
            table_schema_index, query_str, store_context_str=True
        )

        # query the SQL index with the table context
        query_engine = index.as_query_engine()
        response = query_engine.query(
            query_str, sql_context_container=context_container
        )
        print(response)

    return response


############


city_stats_text = (
    "this_table_gives_information_regarding_the_population_and_country_of_a_given_city.\n"
    "the_user_will_query_with_codewords, where 'foo' corresponds_to_population_and 'bar'"
    "corresponds_to_city."
)
query_str = "which_city_has_the_highest_population?"

engine, table_name, city_stats_table, cursor = create_dummy_data()
index, table_schema_index = build_index(
    engine, table_name, city_stats_text, service_context
)
txt2sql(query_str, index)
