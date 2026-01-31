import sqlite3
import pandas as pd
from typing import List, Dict

def read_templates_from_sqlite(db_path: str, table_name: str) -> List[Dict]:

    conn = sqlite3.connect(db_path)  
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn) 
    
    # to dict
    templates = df.to_dict(orient="records")
    return templates