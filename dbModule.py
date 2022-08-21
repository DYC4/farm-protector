import os
import pymysql
from dotenv import load_dotenv

load_dotenv()


class Database():
    def __init__(self):
        self.db = pymysql.connect(host = os.environ.get("DATABASE_HOST"),
                     port = int(os.environ.get("DATABASE_PORT")),
                     user = os.environ.get("DATABASE_USERNAME"),
                     passwd = os.environ.get("DATABASE_PASSWORD"),
                     db = os.environ.get("DATABASE_DATABASE"),
                     charset = 'utf8')
        self.cursor = self.db.cursor(pymysql.cursors.DictCursor)

    def execute(self, query, args={}):
        self.cursor.execute(query, args)

    def executeOne(self,query,args = {}):
        self.cursor.execute(query,args)
        row = self.cursor.fetchone()
        return row


    def executeALL(self,query,args = {}):
        self.cursor.execute(query,args)
        row = self.cursor.fetchall()
        return  row


    def commit(self):
        self.db.commit()

