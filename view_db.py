import sqlite3

conn = sqlite3.connect('ats_database.db')
cursor = conn.cursor()

# List all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables in database:")
for table in tables:
    print(f"  - {table[0]}")

# Show contents of 'resumes' table
print("\nResumes table:")
cursor.execute("SELECT id, user_id, filename, created_at FROM resumes;")
rows = cursor.fetchall()
for row in rows:
    print(f"  ID: {row[0]}, User: {row[1]}, File: {row[2]}, Created: {row[3]}")

conn.close()