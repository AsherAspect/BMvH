import os
import openai

from dotenv import load_dotenv

import streamlit as st # uv pip install --upgrade pip setuptools wheel; uv pip install --only-binary=:all: streamlit pyarrow
from st_aggrid import AgGrid, GridOptionsBuilder
import pandas as pd
import plotly.express as px

from aspect_foundry.sql import SQLStorageHandler
from aspect_foundry.azurite import AzuriteBlobHandler
from aspect_foundry.chat import ChatClient
from handling_stabu.main import Bestek
from handling_stabu.stabu_helpers import get_stabu_list

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

# <<ChatClient>> credentials
clientOpenAI = openai.OpenAI(base_url="https://api.openai.com/v1", api_key=os.getenv("OPENAI_PASSWORD"))

#######################################################################################
# Configurations (IMPERATIVE FOR INIT FUNCTIONS)
#######################################################################################
sqlHandler = SQLStorageHandler(server=server, user=user, password=password, port=port, table_creation_queries=[])

# Azurite handling setup (requires SQL instance --> bound to "table_blobs_name")
container_name = "bestekken"
table_blobs_name = "blobs"
blobHandler = AzuriteBlobHandler(container_name=container_name, connection_string=connection_string, sqlHandler=sqlHandler, table_blobs_name=table_blobs_name)

# Chat with openai configuration
chat = ChatClient(
    client=clientOpenAI,
    model="gpt-5.1",
    role="assistant"
)

with open(file="handling_comparison/prompt_compare.txt", mode="r") as file:
    prompt_compare = file.read()

#######################################################################################
# Interface communications (and restructuring)
#######################################################################################
class Interface():
    """
    The interface class manages interface pages, where a function call will load a page.
    It also holds internal static functions, which function as subprocesses to make all functions more readable.
    """

    def __init__(self):
        """
        Initialize visualization pages.

        Prepare data for querying:
            - Loading all available classifications from all documents into a single dataframe for access.
                - Fix: Loading the queried categories live from SQL into a dataframe
            - Generate all file names from the found UUID's to ensure a user knows which document they are handling.
            - Generate all STABU names from the found STABU codes, for more direct insights.

        """

        self.pages = {
            "Table": self.page_table,
            "Boxplot": self.page_boxplot
        }
        
        # Convert SQL into dataframe
        df = pd.DataFrame(sqlHandler.select("SELECT * FROM classifications"))
        df = df.set_axis(["blob_uuid", "category", "confidence", "reasoning", "totaal_prijs", "totaal_eenheid", "eenheid", "prijs_per_eenheid", "prijs_per_eenheid_onderbouwing", "context"], axis=1).astype({"blob_uuid": str})

        # Convert blob_uuid into bestand
        files = []
        for blob_uuid in df["blob_uuid"].to_list():
            files.append(blobHandler.get_blob_name(blob_uuid))
        df["bestand"] = pd.Series(files)

        # TODO: GET MORE DATA INSTEAD OF MOCK
        df_mock = pd.read_csv("../../Data/df_with_mock_data.csv", dtype={"category": float}).round(2).drop("Unnamed: 0", axis=1)
        df_mock["category"] = df_mock["category"].apply(lambda x: f"{x:.2f}")
        self.df = pd.concat([df, df_mock], ignore_index=True)

        # Append a column with category names to self.df
        stabu_list = get_stabu_list(sqlHandler=sqlHandler, filter_category=(22, 30))
        stabu_dict = {}
        for stabu_category in stabu_list:
            stabu_dict |= stabu_category["criteria"]
        self.df['category_name'] = self.df['category'].map(stabu_dict)

        # --------< Main screen >--------
        st.title("Demo-mockup")



    @staticmethod
    def _get_file_overlap(df, names_list):
        """
        Takes a dataframe and two files (by name), and drops every row between the files, where "category" and "eenheid" are not overlapping.

        Parameters:
            df (pd.DataFrame): The dataframe that contains rows with at least the the files inside of names_list that are contained within "bestand". This is where the overlapping "category" and "eenheid" between the files will be returned from.
            names_list (list): A list with 2 items, where each item is a filename.

        Returns:
            pd.DataFrame: The same DataFrame from the input, but without the categories that are not overlapping.
        """

        # Filter out values with no given price per unit
        df = df[(df["prijs_per_eenheid"] != 0)]
        
        # Get categories for each filter_offerte value
        pair_0 = set(df.loc[df["bestand"] == names_list[0], ["category", "eenheid"]].apply(tuple, axis=1))
        pair_1 = set(df.loc[df["bestand"] == names_list[1], ["category", "eenheid"]].apply(tuple, axis=1))
        
        # Find categories present in both
        common_pairs = pair_0.intersection(pair_1)
        
        # If you want to filter the original DataFrame to only those common categories:
        return df[df.apply(lambda row: (row["category"], row["eenheid"]) in common_pairs, axis=1)]


    @staticmethod
    def _remove_outliers(df, column, std_threshold=3):
        """
        Removes rows from df where values in "column" are more than "std_threshold" standard deviations from the mean.
        
        Parameters:
            df (pd.DataFrame): The input DataFrame.
            column (str): Column name to check for outliers.
            std_threshold (float): Number of standard deviations away from mean to consider as outlier.
        
        Returns:
            pd.DataFrame: Filtered DataFrame without outliers.
        """
        mean_val = df[column].mean()
        std_val = df[column].std()

        lower_bound = mean_val - std_threshold * std_val
        upper_bound = mean_val + std_threshold * std_val
        
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]



    @staticmethod
    def _build_boxplot(df, lines=None, title="Prijs per Eenheid per Categorie en Eenheid"):
        """
        This function will directly construct a plotly box plot and insert it on the Streamlit page.

        Parameters:
            df (pd.DataFrame): DataFrame with columns ["eenheid", "category", "prijs_per_eenheid"]
            lines (list): List of tuples that will be used to produce dotted horizontal lines to point out certain values that are selected [(y_value: float, label: str), ...]
            title (str): The title of the box plot.
        """
        # Basic validation to help catch missing columns early
        required_cols = {"eenheid", "category", "prijs_per_eenheid"}
        missing = required_cols - set(df.columns)
        if missing:
            st.error(f"DataFrame mist verplichte kolommen: {missing}")
            return

        # Create the grouped boxplot
        fig = px.box(
            df,
            # x="category",                 # one box per category on the X-axis
            y="prijs_per_eenheid",        # numeric values for the box
            # color="eenheid",              # separate traces/colors per 'eenheid'
            points="outliers",            # show outliers; use "all" to show all points
            title=title
        )

        # Expand layout a bit for annotations on the right side
        fig.update_layout(
            legend_title_text="Eenheid",
            # xaxis_title="Categorie",
            yaxis_title="Prijs per Eenheid",
        )

        # Add horizontal lines + labels
        if lines:
            for y_val, label in lines:
                fig.add_hline(
                    y=y_val,
                    line_dash="dot",
                    line_color="lightgrey",
                    annotation_text=label,
                    annotation_position="bottom right"
                )

        # Render in Streamlit
        st.plotly_chart(fig, use_container_width=True)


    
    @staticmethod
    def _upload_doc(uploaded_file):
        """
        This function uses the handling_stabu.main.bestek.preprocess() function to take a document and insert it into the blob storage, whilst identifying STABU categories.

        Parameters:
            uploaded_file (file): A document that has to be uploaded to the blob storage, and that has to be analyzed.

        Returns:
            True: (Bad practice) Show the parent process that we can now continue, for example by querying the document that was just uploaded.
        """
        st.title("Offerte wordt geupload")
        st.write("- U wordt vanzelf naar een andere pagina gestuurd wanneer het laden klaar is.")


        # Write the file to disk
        path = f"tmp/{uploaded_file.name}"

        with open(path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        bestek = Bestek()
        bestek.preprocess(path=path, n=2, batch=10)

        # Inform of mission success
        return True



    def page_table(self):
        """
        Calling this function will load the searchable table page.
        NOTE: calling this directly will render this page on top of the already existing HTML. This should only be called from "Pagina" selectbox.
        """

        df_filtered = self.df.copy().sort_values(by=["category", "eenheid"]).drop_duplicates()
        
        # --------< Sidebar >--------

        # Filters
        st.sidebar.header("Filters")
        filter_offerte = st.sidebar.multiselect("Offerte", options=self.df["bestand"].unique())
        filter_category = st.sidebar.multiselect("Categorie", options=self.df["category"].unique())
        filter_eenheid = st.sidebar.multiselect("Eenheid", options=self.df["eenheid"].unique())
        filter_zeros = st.sidebar.toggle("Alleen met waarden")

        # Apply filters
        if filter_offerte:
            df_filtered = df_filtered[df_filtered["bestand"].isin(filter_offerte)]
        if filter_category:
            df_filtered = df_filtered[df_filtered["category"].isin(filter_category)]
        if filter_eenheid:
            df_filtered = df_filtered[df_filtered["eenheid"].isin(filter_eenheid)]
        if filter_zeros:
            df_filtered = df_filtered[(df_filtered["prijs_per_eenheid"] != 0) & (df_filtered["totaal_prijs"] != 0) & (df_filtered["totaal_eenheid"] != 0)]
            

        # File uploader widget
        st.sidebar.header("Upload")
        uploaded_file = st.sidebar.file_uploader("Kies een bestand", type=None)  # type=None allows any file type



        # --------< Main screen >--------
        st.subheader("Geidentificeerde STABU-codes")

        # Show the full table below for reference
        # st.dataframe(df_filtered[["bestand", "category", "prijs_per_eenheid", "eenheid"]], hide_index=True)
        gb = GridOptionsBuilder.from_dataframe(df_filtered[["category", "category_name", "prijs_per_eenheid", "eenheid", "totaal_prijs", "totaal_eenheid"]])
        gb.configure_selection('single')  # or 'multiple'
        grid_options = gb.build()

        grid_response = AgGrid(df_filtered, gridOptions=grid_options, enable_enterprise_modules=False)

        # More details about clicked row
        st.write("### Extra informatie:\n", grid_response['selected_rows'])



        # Check if a file was uploaded
        if uploaded_file is not None:
            # Redirect to table page
            success = self._upload_doc(uploaded_file=uploaded_file)

            if success:
                st.write("- Uploaden is gelukt!")
            
            

    def page_boxplot(self):
        """
        Calling this function will load the searchable box plot page, which also has a compare documents function, with an AI-generated comparison.
        NOTE: calling this directly will render this page on top of the already existing HTML. This should only be called from "Pagina" selectbox.
        """

        df_filtered = self.df.copy().sort_values(by=["category", "eenheid"]).drop_duplicates()
        df_filtered = df_filtered[(df_filtered["prijs_per_eenheid"] != 0) & (df_filtered["totaal_prijs"] != 0) & (df_filtered["totaal_eenheid"] != 0)]
        
        # --------< Sidebar >--------
        lines = []

        # Filters
        # NOTE: Interessante filters --> ("22.72", "m2"), ("30.33", "st"), ("30.32", "m2")
        st.sidebar.header("Filters")
        filter_offerte = st.sidebar.multiselect("Offerte", options=self.df["bestand"].unique())
        filter_category = st.sidebar.multiselect("Categorie", options=self.df["category"].unique(), default=["30.33"])
        filter_eenheid = st.sidebar.multiselect("Eenheid", options=self.df["eenheid"].unique(), default=["st"])
        filter_std = st.sidebar.slider("Standaarddeviatie", 1, 4, 3)

        # Apply filters
        if filter_category:
            df_filtered = df_filtered[df_filtered["category"].isin(filter_category)]
        if filter_eenheid:
            df_filtered = df_filtered[df_filtered["eenheid"].isin(filter_eenheid)]
        if filter_offerte:
            lines_unprocessed = df_filtered[df_filtered["bestand"].isin(filter_offerte)][["prijs_per_eenheid", "bestand"]]
            
            # Create a list of filtered items you want to compare to the entire dataset
            for prijs_per_eenheid, bestand in zip(lines_unprocessed["prijs_per_eenheid"], lines_unprocessed["bestand"]):
                if len(lines_unprocessed["bestand"]) > 1:
                    lines.append((prijs_per_eenheid, f"{prijs_per_eenheid} ({bestand})"))
                else:
                    lines.append((prijs_per_eenheid, f"{prijs_per_eenheid}"))
        if filter_std:
            df_filtered = self._remove_outliers(df_filtered, column="prijs_per_eenheid", std_threshold=filter_std)
        
        # --------< Main screen >--------
        st.subheader("Vergelijk offertes met distributie")

        # Visualize the boxplot on the page
        self._build_boxplot(df_filtered, lines)

        # Compare contents of documents
        st.subheader("Vergelijk geselecteerde offertes")        
        if len(filter_offerte) != 2:
            st.markdown("⚠️ Filter op 2 offertes om deze onderling te kunnen vergelijken.")
        # If 2 documents are selected
        else:
            df_overlap = self._get_file_overlap(self.df[(self.df["bestand"] == filter_offerte[0]) | (self.df["bestand"] == filter_offerte[1])], names_list=filter_offerte).drop_duplicates(subset=["category", "bestand"])
            
            # NOTE: Print statement to show interesting data that helps finding interesting visualization that can be used as default visualizations
            # print(df_overlap[["category", "bestand", "prijs_per_eenheid", "eenheid"]])

            # If at least one category is overlapping between the 2 selected files; create button to compare with AI
            if len(df_overlap) >= 2:

                # Button click triggers AI prompt to clarify the difference between the two selected docs
                if st.button("Vraag AI-Agent", type="primary"):
                    st.markdown("Even geduld; de offertes worden vergeleken:")

                    # Request the documents from storage into tmp/
                    loaded_texts = []
                    loaded_categories = []
                    for single_blob_name in df_overlap["bestand"].unique().tolist():
                        blobHandler.load.from_name(single_blob_name, output_path=f"tmp/")

                        # Read the document and load into ram
                        with open(f"tmp/{single_blob_name}", mode="r") as f:
                            loaded_texts.append(f.read())

                        # remove tmp/...
                        os.remove(f"tmp/{single_blob_name}")

                        # Get JSON from dataframe where all categories are from the same document
                        loaded_categories.append(df_overlap[(df_overlap["bestand"] == single_blob_name)][["bestand", "category", "reasoning", "totaal_prijs", "totaal_eenheid", "eenheid", "prijs_per_eenheid", "prijs_per_eenheid_onderbouwing", "context"]].to_json(orient="records", indent=4))


                    # Configure the prompt
                    configured_prompt_compare = prompt_compare.replace("<<TEXT1>>", loaded_texts[0]).replace("<<TEXT2>>", loaded_texts[1]).replace("<<CATEGORIES1>>", loaded_categories[0]).replace("<<CATEGORIES2>>", loaded_categories[1])

                    # Prompt the LLM
                    st.markdown(chat.openai_llm(configured_prompt_compare).choices[0].message.content)
                    

            else:
                st.markdown("⚠️ Geselecteerde offertes hebben geen onderlinge overlap op een categorie.")



    def run(self):
        page = st.sidebar.selectbox("Pagina", list(self.pages.keys()))
        self.pages[page]()


# python -m streamlit run app.py
# you have to create a new terminal
if __name__ == "__main__":
    interface = Interface()
    interface.run()