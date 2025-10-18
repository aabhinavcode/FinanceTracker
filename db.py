

from sqlalchemy import create_engine

import pandas as pd
engine = create_engine('postgresql://neondb_owner:npg_cSfz9lxwFd5g@ep-bitter-mouse-a8yo6f70-pooler.eastus2.azure.neon.tech/neondb?sslmode=require&channel_binding=require')


# df_transactions.to_sql('transactions', engine, if_exists='replace', index=False)

def save_to_db(df, table_name):
    df.to_sql(table_name, engine, if_exists="append", index=False)
    print(f"✅ Saved {len(df)} rows to {table_name}")

def get_transactions(table_name="transactions"):
    """
    Read all rows from a table
    """
    return pd.read_sql(f"SELECT * FROM {table_name}", engine)
