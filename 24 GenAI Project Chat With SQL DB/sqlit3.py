import sqlite3

connection = sqlite3.connect("student.db")
cursor = connection.cursor()

table_info = """
CREATE TABLE IF NOT EXISTS STUDENT(
    NAME TEXT,
    CLASS TEXT,
    SECTION TEXT,
    MARKS INTEGER
)
"""

cursor.execute(table_info)

cursor.execute("INSERT INTO STUDENT VALUES ('Uditya', 'Machine Learning', 'A', 99)")
cursor.execute("INSERT INTO STUDENT VALUES ('Sai', 'Cyber Security', 'S', 90)")
cursor.execute("INSERT INTO STUDENT VALUES ('Adarsh', 'Machine', 'A', 100)")
cursor.execute("INSERT INTO STUDENT VALUES ('Ram', 'Data Learning', 'A', 50)")
cursor.execute("INSERT INTO STUDENT VALUES ('Raghu', 'Deep Learning', 'A', 96)")

connection.commit()

print("Inserted Records:")

data = cursor.execute("SELECT * FROM STUDENT")
for row in data:
    print(row)

connection.close()