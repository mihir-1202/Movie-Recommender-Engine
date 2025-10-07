import mysql.connector

db = mysql.connector.connect(host = 'localhost', user = 'root', password = 'mihir2005', database = 'world_layoffs')

my_cursor = db.cursor()
my_cursor.execute("SHOW DATABASES")

for x in my_cursor:
    print(x)