# python -m streamlit run app.py
import os

from dotenv import load_dotenv

import streamlit as st # uv pip install --upgrade pip setuptools wheel; uv pip install --only-binary=:all: streamlit pyarrow
import pandas as pd

from aspect_foundry.sql import SQLStorageHandler
from aspect_foundry.azurite import AzuriteBlobHandler
from handling_stabu.main import Bestek

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

#######################################################################################
# Configurations (IMPERATIVE FOR INIT FUNCTIONS)
#######################################################################################
sqlHandler = SQLStorageHandler(server=server, user=user, password=password, port=port, table_creation_queries=[])

# Azurite handling setup (requires SQL instance --> bound to "table_blobs_name")
container_name = "bestekken"
table_blobs_name = "blobs"
blobHandler = AzuriteBlobHandler(container_name=container_name, connection_string=connection_string, sqlHandler=sqlHandler, table_blobs_name=table_blobs_name)

#######################################################################################
# Interface communications (and restructuring)
#######################################################################################
class Interface():
    def __init__(self):
        df = pd.DataFrame(sqlHandler.select("SELECT * FROM classifications"))
        df = df.set_axis(["blob_uuid", "category", "confidence", "reasoning", "totaal_prijs", "totaal_eenheid", "eenheid", "prijs_per_eenheid", "prijs_per_eenheid_onderbouwing", "context"], axis=1)

        # Convert blob_uuid into bestand
        files = []
        for blob_uuid in df["blob_uuid"].to_list():
            files.append(blobHandler.get_blob_name(blob_uuid))
        df["bestand"] = pd.Series(files)

        # TODO: GET MORE DATA INSTEAD OF MOCK
        df_mock = pd.read_csv("../../Data/df_with_mock_data.csv", dtype={'category': object}).drop("Unnamed: 0", axis=1)
        self.df = pd.concat([df, df_mock], ignore_index=True)

    def show_table(self):
        df_filtered = self.df.copy().sort_values(by=["category", "eenheid"]).drop_duplicates()
        
        # --------< Sidebar >--------

        # Filters
        st.sidebar.header("Filters")
        filter_offerte = st.sidebar.multiselect("Filter op offerte", options=self.df["bestand"].unique())
        filter_category = st.sidebar.multiselect("Filter op categorie", options=self.df["category"].unique())
        filter_eenheid = st.sidebar.multiselect("Filter op eenheid", options=self.df["eenheid"].unique())

        # Apply filters
        if filter_offerte:
            df_filtered = df_filtered[df_filtered["bestand"].isin(filter_category)]
        if filter_category:
            df_filtered = df_filtered[df_filtered["category"].isin(filter_category)]
        if filter_eenheid:
            df_filtered = df_filtered[df_filtered["eenheid"].isin(filter_eenheid)]

        # File uploader widget
        st.sidebar.header("Upload")
        uploaded_file = st.sidebar.file_uploader("Kies een bestand", type=None)  # type=None allows any file type



        # --------< Main screen >--------
        # App title
        st.title("File Upload App")

        # Show the full table below for reference
        st.write("### Full Table")
        st.dataframe(df_filtered[["bestand", "category", "prijs_per_eenheid", "eenheid"]], hide_index=True)

        # Check if a file was uploaded
        if uploaded_file is not None:
            self._preprocess_doc(uploaded_file=uploaded_file)
            
            
            
    
    def _preprocess_doc(self, uploaded_file):
        print("START")
        # Write the file to disk
        path = f"tmp/{uploaded_file.name}"

        with open(path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        bestek = Bestek()
        bestek.preprocess(path=path, n=2, batch=10)
        print("DONE")


interface = Interface()
interface.show_table()