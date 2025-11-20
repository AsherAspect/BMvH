import os

import pandas as pd

from dotenv import load_dotenv

from aspect_foundry.sql import SQLStorageHandler

# TODO: REWRITE THIS PART FOR BMVH
"""
This script is made to transform the SFI dataset into SQL, where each SFI code description is transformed into a list of requirements

Links:
    - SFI coding system: https://www.norsetechgroup.com/topics/sfi-coding-system/
    - SFI coding example: https://www.scribd.com/document/699242144/SFI-numbering-full-list

STABU structure: xx.yy
    - Main --> xx
        - Group --> yy

"""

#######################################################################################
# Uploading SFI .csv categories to SQL
#######################################################################################
def build_stabu_from_csv(sqlHandler, stabu_doc):
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
# Obtaining list of SFI categories from SQL
#######################################################################################
def get_stabu_list(sqlHandler, filter_category=None):
    # TODO: edit docs
    """
    Get an SFI category list that can be provided to the init function of a ScopeClassifier

    Args:
        sqlHandler (SQLStorageHandler): Credentials to connect to the required SQL instance to keep an overview of the files inside of the blob storage.
        filter_category (str): Use this arg only get the sub-categories of a certain SFI filter.

    Example:
        sqlHandler: SQLStorageHandler(server=server, user=user, password=password, port=port)
        filter_category: "10"

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
        SQL_categories_len = ("SELECT * FROM stabu WHERE LEN(category) = 2 AND category LIKE '%(filter_category)s^^^'" % {"filter_category":filter_category}).replace("^^^", "%")
    
    
    SQL_categories = sqlHandler.select(SQL_categories_len)
    
    for SQL_category in SQL_categories:
        category = {}

        category["category"] = SQL_category[0]
        category["name"] = SQL_category[1]

        SQL_category_len = ("SELECT * FROM stabu WHERE LEN(category) = 5 AND category LIKE '%(category)s^^^'" % {"category":category["category"][0]}).replace("^^^", "%")

        criteria = {}
        
        SQL_criteria = sqlHandler.select(SQL_category_len)
        for SQL_criterion in SQL_criteria:
            criteria[SQL_criterion[0]] = SQL_criterion[1]

        category["criteria"] = criteria

        categories.append(category)


    return categories



# TODO: onderstaande functions mogelijk mergen voor het eind

#######################################################################################
# Read text, compare to stabu's and create one large dict of classifications
#######################################################################################
def get_stabu_chunk(text, n, batch, scopeClassifier):

    # Filter out the STABU codes which had their conditions met
    def get_positive_stabu(classification):
        # Filter out criteria, with at least one positive classification
        positive_classification = {}
        for full_category, criteria_dict in classification.items():
            positive_classification[full_category] = {}
            for criterion, criterion_contents in criteria_dict.items():
                met = criterion_contents["met"]
                reasoning = criterion_contents["reasoning"]
                if int(met.split("/")[0]) > 0:
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