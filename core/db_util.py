"""
Author: Logan Blue
Date: March 3, 2020

This class will encapsilate the database of choice (currently mongodb) for the acoustic
work. It will support loading, querying, and insertion into the database.
"""
#pylint: disable=trailing-whitespace, invalid-name, dangerous-default-value

import json
import os
from pathlib import Path

class DBObj:
    """This object wraps the interface to our database. Used for future flexibility is 
    we need to change the underlying database. """

    def __init__(self, db_name='exploration', collection_name='timit_train'):
        """Constructor for db obj, will connect to default port and to a 
        specified db and collection. By default the db = 'exploration' and 
        collection = 'timit_train'
        """
        self.backend = os.environ.get('WRY_STORAGE_BACKEND', 'mongo').lower()
        self.collection_name = collection_name
        if self.backend == 'file':
            out_dir = Path(os.environ.get('WRY_FEATURE_DIR', 'outputs/local_features'))
            out_dir.mkdir(parents=True, exist_ok=True)
            self.file_path = out_dir / f'{collection_name}.jsonl'
            self.table = None
            self.db = None
        else:
            import pymongo
            db_client = pymongo.MongoClient()               #connect to mongo
            self.db = db_client[db_name]                    #connect to correct db
            self.table = self.db[collection_name]           #access correct collection

    def insert(self, data):
        """This function will insert records into the table. 

        data - is assumed to be a pandas dataframe. 
        """
        if isinstance(data, dict):
            self.__insert_single(data)
        else:
            self.__insert_multi(data)

    def __insert_single(self, data):
        """This function will insert records into the table. 

        data - is assumed to be a pandas dataframe. 
        """
        if self.backend == 'file':
            with self.file_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(data, ensure_ascii=True) + '\n')
        else:
            self.table.insert_one(data)

    def __insert_multi(self, data):
        """This function will insert records into the table. 

        data - is assumed to be a pandas dataframe. 
        """
        #convert data into a list of dictionaries
        insertable_data = []
        for _, row in data.iterrows():
            insertable_data.append(row.to_dict())

        if self.backend == 'file':
            with self.file_path.open('a', encoding='utf-8') as f:
                for row in insertable_data:
                    f.write(json.dumps(row, ensure_ascii=True) + '\n')
        else:
            self.table.insertMany(insertable_data)

    def query(self, filters={}):
        """This function wraps the query/search functionality of the db. 
        The filters will be a dictionary of the column name, the condition, and 
        the operation that relates the column and the condition. By default, if 
        filters are not provided, this function will return the whole collection. 

        We expect the filters input to be in the mongo db style of querying. 
        This is done to simply the early development of the tool, if later versions
        of the tool require a different DB backend we will translate mongoDB style
        queries to the new databases standards here. 
        """
        if self.backend == 'file':
            if not self.file_path.exists():
                return []
            records = []
            with self.file_path.open('r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    records.append(json.loads(line))
            return records
        return self.table.find(filters)

    def distinct(self, field):
        """Return distinct values for a field."""
        if self.backend == 'file':
            if not self.file_path.exists():
                return []
            values = set()
            with self.file_path.open('r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if field in row:
                        values.add(row[field])
            return list(values)
        return list(self.table.distinct(field))
