import os

from dotenv import load_dotenv

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from handling_stabu.stabu_helpers import build_stabu_from_csv

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

build_stabu_from_csv()