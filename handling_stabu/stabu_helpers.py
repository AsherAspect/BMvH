import os
import json

import pandas as pd

from dotenv import load_dotenv

from aspect_foundry.sql import SQLStorageHandler

#######################################################################################
# Uploading SFI .csv categories to SQL
#######################################################################################
def build_stabu_from_csv(sqlHandler, stabu_doc):
    """
    This function transforms a simple document with categories and category names into a SQL table.

    Parameters:
        sqlHandler (SQLStorageHandler): Credentials to connect to the required SQL instance to keep an overview of the files inside of the blob storage.
        stabu_doc (.csv): Takes a .csv document with the following headers: "category;name".
    """


    df = pd.read_csv(stabu_doc, sep=";", dtype=str)

    # Fetch all data
    categories = df["category"].to_numpy().tolist()
    names = df["name"].to_numpy().tolist()

    # top_tier_categories = []
    for category, name in zip(categories, names):
        kwargs = {
            "query": "INSERT INTO stabu (category, name) VALUES (%s, %s);",
            "params": (category, name)
        }
        sqlHandler.insert(**kwargs)


#######################################################################################
# Obtaining list of STABU categories from SQL
#######################################################################################
def get_stabu_list(sqlHandler, filter_category=None):
    """
    Get a STABU category list that can be provided to the init function of a ScopeClassifier

    Parameters:
        sqlHandler (SQLStorageHandler): Credentials to connect to the required SQL instance to keep an overview of the files inside of the blob storage.
        filter_category (str/tuple): Use this arg to only get the sub-categories of a certain STABU filter.

    Example:
        sqlHandler: SQLStorageHandler(server=server, user=user, password=password, port=port)
        filter_category: "10" or (10, 20)

    Returns:
        [
            {'category': '10', 'name': 'ESTIMATING, DRAWING & OFFERS W.R.T. CHANGE ORDERS', 'criteria': {}},
            {...}
        ]
    """

    categories = []

    if filter_category == None:
        SQL_categories_len = "SELECT * FROM stabu WHERE LEN(category) = 2"
    else:
        SQL_categories_len = ("SELECT * FROM stabu WHERE LEN(category) = 2 AND category IN %(filter_category)s" % {"filter_category":filter_category})
    
    SQL_categories = sqlHandler.select(SQL_categories_len)
    
    for SQL_category in SQL_categories:
        category = {}

        category["category"] = SQL_category[0]
        category["name"] = SQL_category[1]

        SQL_category_len = ("SELECT * FROM stabu WHERE LEN(category) = 5 AND category LIKE '%(category)s^^^'" % {"category":category["category"]}).replace("^^^", "%")
        
        criteria = {}
        
        SQL_criteria = sqlHandler.select(SQL_category_len)
        for SQL_criterion in SQL_criteria:
            criteria[SQL_criterion[0]] = SQL_criterion[1]

        category["criteria"] = criteria

        categories.append(category)


    return categories



#######################################################################################
# Read text, compare to stabu's and create one large dict of classifications
#######################################################################################
def get_stabu_chunk(text, n, batch, scopeClassifier):
    """
    This function returns a dict with all categories that have been positively classified by the LLM.

    Parameters:
        text (str): The text that has to be classified by the scopeClassifier.
        n (int): The number of times the chat client should be called to classify the same piece of text.
        batch (int): Hard cap on when to split categories, based on the max amount of criteria that is allowed in one prompt. This function will help to prevent the LLM from overloading by the sheer amount of categories provided.
        scopeClassifier (ScopeClassifier): This function is used to classify the scope of a a provided text, based on a set of categories.

    Returns:
        dict(): A dict with categories as keys and each key has a dict as value, where the nested dict contains keys:
            - "met": The amount of positive classifications out of the total amount of classifications, which can look like "2/2".
            - "reasoning": An explanation from the LLM as to why it has classified the item as such.
    """

    # Filter out the STABU codes which had their conditions met
    # TODO: Integrate as arg into ScopeClassifier.classify, since it should be an option to only return values that scored above a certain threshold
    def get_positive_stabu(classification):
        """
        This function removes all classifications that aren't "x/x", in order to create a clean output.

        Parameters:
            classification (dict()): A dict with categories as keys and each key has a dict as value, where the nested dict contains keys:
                - "met": The amount of positive classifications out of the total amount of classifications, which can look like "2/2".
                - "reasoning": An explanation from the LLM as to why it has classified the item as such.

        Returns:
            dict(): A dict with categories as keys and each key has a dict as value, where the nested dict contains keys:
                - "met": The amount of positive classifications out of the total amount of classifications, which is ALWAYS "x/x", since we only allow classifications with 100% certainty.
                - "reasoning": An explanation from the LLM as to why it has classified the item as such.
        """
        # Filter out criteria, with FULL positive classification
        positive_classification = {}
        for full_category, criteria_dict in classification.items():
            positive_classification[full_category] = {}
            for criterion, criterion_contents in criteria_dict.items():
                met = criterion_contents["met"].split("/")
                # if int(met.split("/")[0]) > 0: # filter on at least one positive
                if int(met[0]) == int(met[1]):
                    positive_classification[full_category][criterion] = criterion_contents
        # Remove categories that ended up being empty, since none of the criteria matched the context
        filtered_classification = {}
        for full_category, criteria_dict in positive_classification.items():
            if len(criteria_dict) > 0:
                filtered_classification[full_category] = criteria_dict

        return filtered_classification

    # Fetches a list of stabu categories from a classify function output
    def get_stabu_category_list(classification):
        """
        This function can be used to find which nested categories should be persued, according to the already identified categories.
        The difference is that it creates a simple list of categories, which makes it easier to use the classifier.

        Parameters:
            classification (dict()): A dict with categories as keys and each key has a dict as value, where the nested dict contains keys:
                - "met": The amount of positive classifications out of the total amount of classifications, which can look like "2/2".
                - "reasoning": An explanation from the LLM as to why it has classified the item as such.

        Returns:
            list(): ["10", "22", "30"]
        """

        categories = []
        for criteria_dict in classification.values():
            for criterion in criteria_dict.keys():
                categories.append(criterion)

        return categories
    

    # Retrieve full classification for every category, then ONLY fetch the classifications with 'met' > 1
    full_classification = scopeClassifier.classify(text, batch_size=batch, n=n, simplify_output=True)
    
    filtered_classification = get_positive_stabu(full_classification)

    return filtered_classification



#######################################################################################
# Reiterate provided positive classifications and look up additional information
#######################################################################################
def get_additional_stabu_info(text, prompt, classification, chat):
    """
    After the classification step, the LLM is prompted again to get more additional information that can accompany the classification, such as prices, units and more context.

    Parameters:
        text (str): The text that has to be classified by the scopeClassifier.
        prompt (str): The prompt to ask for additional info, which includes <<TEXT>> and <<CATEGORIES>>, since those will be replaced with the contents from the other paramters.
        classification (dict()): A dict with categories as keys and each key has a dict as value, where the nested dict contains keys:
            - "met": The amount of positive classifications out of the total amount of classifications, which can look like "2/2".
            - "reasoning": An explanation from the LLM as to why it has classified the item as such.
        chat (ChatClient): An instance of the ChatClient class, which is used to classify the text.

    Returns:
        JSON(str): A json string where the actual structure has been defined by the prompt, taking the already existing classification structure into consideration.
    """

    prompt = prompt.replace("<<TEXT>>", text)
    prompt = prompt.replace("<<CATEGORIES>>", json.dumps(classification))

    # Request response from openai chat (try 3x to be safe)
    for attempt in range (3):
        try:
            response = chat.openai_llm(prompt).choices[0].message.content

            return json.loads(response)
        except json.JSONDecodeError:
            print("\n[get_additional_stabu_info]: LLM failed to return valid JSON, you should improve prompting instruction to return a valid JSON if this happens too frequently.")
            print("[get_additional_stabu_info]: The response that should have been parsed by json.loads(): \n", response, "\n\n")