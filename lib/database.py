import pandas as pd
import psycopg2
import getpass
from IPython.display import display_html

"""
Create connection to PSQL server
"""
def connect_psql():
	# Connect to PSQL
	# connection info
	conn_info = {
	    'sqluser': 'postgres',
	    'sqlpass': '',
	    'sqlhost': 'localhost',
	    'sqlport': 5432,
	    'dbname': '',
	    'schema_name': '',
	}
	    
	# Connect to the database
	print('Database: {}'.format(conn_info['dbname']))
	print('Username: {}'.format(conn_info["sqluser"]))

	conn_info["sqlpass"] = getpass.getpass('Password: ')
	con = psycopg2.connect(dbname=conn_info["dbname"],
		               host=conn_info["sqlhost"],
		               port=conn_info["sqlport"],
		               user=conn_info["sqluser"],
		               password=conn_info["sqlpass"])

	return con


def query_table_with_id(con, query, patientunitstayid):
    query = query.replace('{patientunitstayid}', str(patientunitstayid))
    return(pd.read_sql_query(query, con))


def display_pandas_df(*args):
    html_str=''
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)

    
def extract_table_by_time(df_table, column, dx_time, row_count):
    time_diff = abs(df_table[column] - dx_time)
    idx_min_time_diff = time_diff.sort_values(ascending=True).index[0:row_count]
    df_extract = df_table.iloc[idx_min_time_diff]
    
    return(df_extract)

