import os
from datetime import datetime as dt

MONGO_USER = os.getenv("MONGO_USER")
MONGO_USER_PW = os.getenv("MONGO_USER_PW")
MONGO_IP = os.getenv("MONGO_IP")
MONGO_PORT = os.getenv("MONGO_PORT")
MONGO_AUTH_SOURCE = os.getenv("MONGO_AUTH_SOURCE")
MONGO_URI = f"mongodb://{MONGO_USER}:{MONGO_USER_PW}@{MONGO_IP}:{MONGO_PORT}/default?authSource={MONGO_AUTH_SOURCE}"
SP500_URL = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
R1000_URL = "https://en.wikipedia.org/wiki/Russell_1000_Index"
IGNORE = ["BF.B", "BRK.B"]
INIT_TS = dt(1900,1,1)