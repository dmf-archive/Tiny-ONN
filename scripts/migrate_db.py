import json
import os
import sqlite3
import sys
import pandas as pd

# Add the project root to the Python path to allow importing config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import QUANTIZATION_SCALE

def generate_param_name_map(modules_file="1.7B_model_modules.txt", output_dir="metadata"):
    param_names = []
    try:
        with open(modules_file, 'r', encoding='utf-16') as f: # Try UTF-16 first
            for line in f:
                name = line.strip()
                if name: # Ensure line is not empty
                    param_names.append(name)
    except FileNotFoundError:
        print(f"Error: Module list file '{modules_file}' not found.")
        return {}, {}
    except UnicodeDecodeError as e:
        print(f"Error decoding file '{modules_file}': {e}. Trying with 'utf-8' encoding as fallback.")
        try: # Fallback to utf-8 if utf-16 fails
            with open(modules_file, 'r', encoding='utf-8') as f:
                for line in f:
                    name = line.strip()
                    if name:
                        param_names.append(name)
        except Exception as e_fallback:
            print(f"Failed to read file with 'utf-8' encoding either: {e_fallback}")
            return {}, {}

    # Create a unique mapping from param_name to integer ID
    param_name_to_id = {name: i for i, name in enumerate(sorted(list(set(param_names))))}
    id_to_param_name = {i: name for name, i in param_name_to_id.items()}

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "param_name_map.json")

    with open(output_path, 'w') as f:
        json.dump(id_to_param_name, f, indent=4) # Only save id_to_param_name

    print(f"Param name map generated and saved to '{output_path}'.")
    print(f"Total unique param names: {len(id_to_param_name)}")
    return param_name_to_id, id_to_param_name

def migrate_database(old_db_path="tiny_onn_metrics.db", new_db_path="tiny_onn_metrics_new.db"):
    if not os.path.exists(old_db_path):
        print(f"Error: Old database '{old_db_path}' not found.")
        return

    # Generate or load param_name_map
    param_name_to_id, id_to_param_name = generate_param_name_map()
    if not param_name_to_id:
        print("Warning: Parameter name map could not be generated or loaded. Param names will not be converted to IDs.")

    print(f"Migrating data from '{old_db_path}' to '{new_db_path}'...")

    # 1. Read all data from the old database
    old_conn = sqlite3.connect(old_db_path)
    try:
        df = pd.read_sql_query("SELECT token_idx, param_name, block_idx, activation, grad_norm, absmax FROM block_metrics", old_conn)
    except pd.io.sql.DatabaseError as e:
        print(f"Error reading from old database: {e}")
        old_conn.close()
        return
    old_conn.close()

    if df.empty:
        print("No data found in the old database. Creating an empty new database.")
        # Create an empty new database with the correct schema
        new_conn = sqlite3.connect(new_db_path)
        cursor = new_conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS block_metrics (
            token_idx INTEGER,
            param_name TEXT,
            block_idx INTEGER,
            activation INTEGER,
            grad_norm INTEGER,
            absmax INTEGER,
            PRIMARY KEY (token_idx, param_name, block_idx)
        )
        """)
        new_conn.commit()
        new_conn.close()
        os.replace(new_db_path, old_db_path)
        print("Empty new database created and replaced old one.")
        return

    # 2. Convert param_name to ID and quantize the float values
    print(f"Converting param_name to ID and quantizing data with scale {QUANTIZATION_SCALE}...")
    if param_name_to_id:
        df['param_name'] = df['param_name'].map(param_name_to_id)
        df.dropna(subset=['param_name'], inplace=True) # Drop rows where mapping failed
        df['param_name'] = df['param_name'].astype(int) # Ensure it's integer type for DB
    
    df['activation'] = (df['activation'] * QUANTIZATION_SCALE).astype(int)
    df['grad_norm'] = (df['grad_norm'] * QUANTIZATION_SCALE).astype(int)
    df['absmax'] = (df['absmax'] * QUANTIZATION_SCALE).astype(int)

    # 3. Create a new database with the INTEGER schema and insert quantized data
    new_conn = sqlite3.connect(new_db_path)
    cursor = new_conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS block_metrics (
        token_idx INTEGER,
        param_name INTEGER, -- Changed to INTEGER for param_id
        block_idx INTEGER,
        activation INTEGER,
        grad_norm INTEGER,
        absmax INTEGER,
        PRIMARY KEY (token_idx, param_name, block_idx)
    )
    """)
    
    # Insert data in chunks to avoid memory issues for very large DFs
    chunk_size = 10000
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        data_to_insert = chunk.values.tolist()
        cursor.executemany("""
        INSERT OR REPLACE INTO block_metrics (token_idx, param_name, block_idx, activation, grad_norm, absmax)
        VALUES (?, ?, ?, ?, ?, ?)
        """, data_to_insert)
        new_conn.commit()
        print(f"Inserted {min(i + chunk_size, len(df))} / {len(df)} records.")

    new_conn.close()
    print(f"Data migration complete. New database saved to '{new_db_path}'.")

    # 4. Replace the old database file with the new one
    try:
        os.replace(new_db_path, old_db_path)
        print(f"Old database '{old_db_path}' replaced with new quantized database.")
    except OSError as e:
        print(f"Error replacing database file: {e}")

if __name__ == "__main__":
    migrate_database()
