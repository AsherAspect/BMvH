import os
# import weaviate
import json
import openai
import uuid

# import weaviate.classes as wvc

from dotenv import load_dotenv
from pathlib import Path

# from aspect_foundry.chunker import SpecChunker
from aspect_foundry.azurite import AzuriteBlobHandler
from aspect_foundry.sql import SQLStorageHandler
# from aspect_foundry.weaviate import WeaviateHandler
from aspect_foundry.chat import ChatClient
from aspect_foundry.classifier import ScopeClassifier

from handling_stabu.stabu_helpers import build_stabu_from_csv, get_stabu_list, get_stabu_chunk, get_additional_stabu_info

#######################################################################################
# Credentials
#######################################################################################
# Load .env file
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Docker/.env"))
load_dotenv(dotenv_path=env_path)

# <<SQLStorageHandler>> credentials
server = os.getenv("SQL_IP")
port = int(os.getenv("SQL_PORT"))
user = os.getenv("SQL_USERNAME")
password = os.getenv("SQL_PASSWORD")

# <<AzuriteBlobHandler>> credentials
connection_string = (
    f"DefaultEndpointsProtocol={os.getenv("AZURE_DEFAULT_ENDPOINTS_PROTOCOL")};"
    f"AccountName={os.getenv("AZURE_ACCOUNT_NAME")};"
    f"AccountKey={os.getenv("AZURE_ACCOUNT_KEY")};"
    f"BlobEndpoint={os.getenv("AZURE_BLOB_ENDPOINT")};"
)

# <<WeaviateHandler>> credentials
# clientWeaviate = weaviate.WeaviateClient(
#     connection_params=weaviate.connect.ConnectionParams.from_url(
#         url=os.getenv("WEAVIATE_URL"), 
#         grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT")),
#     ),
#     additional_headers={
#         "X-OpenAI-Api-Key": os.getenv("OPENAI_PASSWORD"),
#     }
# )

# <<ChatClient>> credentials
clientOpenAI = openai.OpenAI(base_url="https://api.openai.com/v1", api_key=os.getenv("OPENAI_PASSWORD"))

#######################################################################################
# Configurations (IMPERATIVE FOR INIT FUNCTIONS)
#######################################################################################

# Create tmp path for temporary files
TMP_DIR = Path("tmp")
TMP_DIR.mkdir(exist_ok=True)

# SQL configurations, such as creating the blobs table that will 1:1 with the Azurite blob storage
table_blobs_name = "blobs"
table_blobs_creation = """
    IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='blobs' AND xtype='U')
    CREATE TABLE blobs (
        uuid UNIQUEIDENTIFIER NOT NULL PRIMARY KEY,
        filename NVARCHAR(255) NOT NULL,
        date_last_chunked DATETIME NULL,
        date_created DATETIME NOT NULL DEFAULT GETDATE()
    )
"""
# SQL configurations to build the STABU coding table, from which categories can be retrieved for classification
table_stabu_creation = """
    IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='stabu' AND xtype='U')
    CREATE TABLE stabu (
        category VARCHAR(7) NOT NULL,
        name VARCHAR(255) NOT NULL
    )
"""
# CSV of STABU codes, which can be accessed to restore the SQL in case it is reset
stabu_doc = r"C:\Users\Asher\OneDrive - Aspect ICT\Documenten\_Projects\BMvH\Data\STABU2_Systematiek.csv"
# SQL configurations for the chunk table with recognized STABU codes
table_classifications_name = "classifications"
# TODO: Add more tables to include price_per_unit, unit, total_price, total_unit
table_classifications_creation = """
    IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='classifications' AND xtype='U')
    CREATE TABLE classifications (
        blob_uuid UNIQUEIDENTIFIER NOT NULL,
        category VARCHAR(5) NOT NULL,
        confidence VARCHAR(7) NOT NULL,
        reasoning VARCHAR(1020) NOT NULL,
        totaal_prijs FLOAT NOT NULL, 
        totaal_eenheid FLOAT NOT NULL, 
        eenheid VARCHAR(10) NOT NULL, 
        prijs_per_eenheid FLOAT NOT NULL, 
        prijs_per_eenheid_onderbouwing VARCHAR(1020) NOT NULL,
        context VARCHAR(1020) NOT NULL
    )
"""
table_creation_queries=[
    table_blobs_creation, 
    table_stabu_creation, 
    table_classifications_creation
    ]


sqlHandler = SQLStorageHandler(server=server, user=user, password=password, port=port, table_creation_queries=table_creation_queries)

# Upload STABU .csv to SQL if it is not there yet
if len(sqlHandler.select("SELECT * FROM stabu")) == 0:
    build_stabu_from_csv(sqlHandler=sqlHandler, stabu_doc=stabu_doc)


# Azurite handling setup (requires SQL instance --> bound to "table_blobs_name")
container_name = "bestekken"

blobHandler = AzuriteBlobHandler(container_name=container_name, connection_string=connection_string, sqlHandler=sqlHandler, table_blobs_name=table_blobs_name)

# Weaviate collection configuration
# chunk_collection_name = "Chunks"
# collectionsWeaviate = [
#     {
#         "name": chunk_collection_name,
#         "description": "Bestek chunks",
#         "vector_config": wvc.config.Configure.Vectors.text2vec_openai(
#             name="chunk_vector",
#             model="text-embedding-3-small",
#             source_properties=["text"],
#         )
#     }
# ]

# weaviateHandler = WeaviateHandler(client=clientWeaviate, collections=collectionsWeaviate)

# Chat with openai configuration
chat = ChatClient(
    client=clientOpenAI,
    model="gpt-4.1",
    role="assistant"
)

# STABU classifier configurations
# NOTE: FOR THE DEMO WE WILL ONLY ACCESS CATEGORY 22 and 30
stabu_list = get_stabu_list(sqlHandler=sqlHandler, filter_category=(22, 30))

with open(file="handling_stabu/prompt_classify.txt", mode="r") as file:
    prompt_classify = file.read()

scopeClassifier = ScopeClassifier(
    categories=stabu_list,
    chat=chat,
    prompt=prompt_classify
)

# Load prompt details doc
with open(file="handling_stabu/prompt_details.txt", mode="r") as file:
    prompt_details = file.read()

#######################################################################################
# Bestek handling
#######################################################################################
class Bestek:
    def __init__(self):
        pass



    def preprocess(self, path, n, batch):
        """
        This function allows for the preprocessing of a certain file, which is obtained from a provided path.
        The preprocessing includes:
            1. Uploading the file to Azure with a generated UUID is name to allow for duplicates (an SQL instance keeps track of which file is associated to which UUID).
            2. Enriching the documents with STABU categories that were matched to it, then uploading it to SQL.

        Preprocessing provides the ability to:
            - Access all preprocessed files in one central blob database.
            - Query SQL based on: blob_uuid, chunk_uuid, category, confidence, reasoning.

        Parameters:
            path (str): The full path to the file to be uploaded and preprocessed.
            n (int): The number of times the chat client should be called to classify the same piece of text.
            batch (int): Hard cap on when to split categories (XX), based on the max amount of criteria (YY) that is allowed in one prompt. This function will help to prevent the LLM from overloading by the sheer amount of categories provided.
        Example:
            path = home/user/Documents/bestek.pdf
            n = 3
            batch = 10
        """

        # Upload .pdf and save the uuid_bestek inside of the class
        self.uuid_bestek = blobHandler.upload(path=path)
        # Download .pdf into tmp for chunking and save the name_bestek inside of the class
        self.name_bestek = blobHandler.load.from_uuid(self.uuid_bestek, output_path=f"{TMP_DIR}/")
        
        # TODO: replace this with an import to read_pdf(pdf_path=f"tmp/{self.name_bestek}") --> str
        with open(file=path, mode="r") as file:
            text = file.read()
        # Chunk the .pdf to obtain a chunks dict
        # chunker = SpecChunker()
        # chunks_list = chunker.chunk_pdf(pdf_path=f"tmp/{self.name_bestek}")
        
        # Fetch SFI codes that were detected inside of chunk and append to SQL
    
        # Read text, compare to stabu's and create one large dict of FILTERED classifications
        filtered_classification = get_stabu_chunk(text=text, n=n, batch=batch, scopeClassifier=scopeClassifier)
        
        # Prompt the AI to figure out the exact details of each stabu found
        detailed_classification = get_additional_stabu_info(text=text, prompt=prompt_details, classification=filtered_classification, chat=chat)
        
        # Iterate every classification with 'met' = x/x and upload to classifications SQL db
        for criteria_values in detailed_classification.values():
            for criterion, criterion_values in criteria_values.items():
                reasoning = criterion_values["reasoning"]
                met = criterion_values["met"]
                totaal_prijs = criterion_values["totaal_prijs"]
                totaal_eenheid = criterion_values["totaal_eenheid"]
                eenheid = criterion_values["eenheid"]
                prijs_per_eenheid = criterion_values["prijs_per_eenheid"]
                prijs_per_eenheid_onderbouwing = criterion_values["prijs_per_eenheid_onderbouwing"]
                context =  criterion_values["context"]
                
                params = (str(self.uuid_bestek), criterion, met, reasoning[0], totaal_prijs, totaal_eenheid, eenheid, prijs_per_eenheid, prijs_per_eenheid_onderbouwing, context)
                sqlHandler.insert(f"INSERT INTO {table_classifications_name} (blob_uuid, category, confidence, reasoning, totaal_prijs, totaal_eenheid, eenheid, prijs_per_eenheid, prijs_per_eenheid_onderbouwing, context) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);", params=params)

        # Delete the tmp .pdf file right away
        os.remove(f"tmp/{self.name_bestek}")


    # TODO: requires more testing before commercialization
    def remove(self, name=None, uuid=None, all=False):
        """
        THIS FUNCTION IS NOT STABLE YET.
        """

        # Reset the db
        if all==True:
            sqlHandler.drop(query="DROP TABLE blobs;")
            sqlHandler.drop(query="DROP TABLE stabu;")
            sqlHandler.drop(query="DROP TABLE classifications;")
            # clientWeaviate.connect()
            # clientWeaviate.collections.delete(chunk_collection_name)
            # clientWeaviate.close()
            # TODO: azurite



#######################################################################################
# Test input
#######################################################################################
if __name__ == "__main__":
    # TODO: Find some way to input the bestek.pdf from the front-end
    # file_name = "2211-metselwerk-offerte.txt" # Geindexeerd
    # file_name = '2211-metselwerk-offerte-APR.txt' # Zelfde als bovenstaande
    file_name = '2211-metselwerk-offerte-LEEN.txt' # NOTE: tijdens demo
    # file_name = '3011-kozijnen-offerte-vdvin.txt' # Geindexeerd
    # file_name = '3011-kozijnen-offerte-vErk.txt'
    local_file_path = f'../../Data/offertes_txt/{file_name}'

    bestek = Bestek()

    # --------< Bestek resetting >--------
    # bestek.remove(all=True)

    # --------< Bestek preprocessing >--------
    bestek.preprocess(local_file_path, n=2, batch=10)

    # Bestek uuid fetching
    blob_uuid = blobHandler.get_blob_uuid(file_name)
    # blob_uuid = uuid.UUID('9ffbe8f3-a9b3-4a51-b1c8-de4ace036f82')


    # --------< Query examples with SQL response>--------
    results = sqlHandler.select(f"SELECT * FROM {table_classifications_name} WHERE blob_uuid='{blob_uuid}'")
    for result in results:
        blob_uuid = result[0]
        category = result[1]
        confidence = result[2]
        reasoning = result[3]
        totaal_prijs = result[4]
        totaal_eenheid = result[5]
        eenheid = result[6]
        prijs_per_eenheid = result[7]
        prijs_per_eenheid_onderbouwing = result[8]
        context = result[9]

        print(category, "\n", confidence, "\n", reasoning, "\n", totaal_prijs, "\n", totaal_eenheid, "\n", eenheid, "\n", prijs_per_eenheid, "\n", prijs_per_eenheid_onderbouwing, "\n", context, "\n\n")

    # SQL stabu browser
    # results = sqlHandler.select("SELECT * FROM stabu")
    # for result in results:
    #     print(result)