from sqlitedict import SqliteDict
import sqlite3
import codecs


def create_dictdb_from_file(file_path, db_path):
    db = SqliteDict(db_path, autocommit=True)

    file = codecs.open(file_path, 'r')
    line = file.readline()

    while line != '':
        splitted = line.split()
        line = file.readline()
        try:
            key, value = splitted[0], ' '.join(splitted[1:])
            db[key] = value
        except IndexError:
            continue

    file.close()
    db.close()


def create_db_from_dictdb(lookup_db_path, longabs_db_path, labels_db_path, db_name):
    connection = sqlite3.connect(db_name)
    cursor = connection.cursor()

    cursor.execute('''CREATE TABLE graph (node_id INTEGER PRIMARY KEY NOT NULL, long_abstracts TEXT, labels TEXT)''')

    connection.commit()

    lookup_db = SqliteDict(lookup_db_path, autocommit=False)
    longabs_db = SqliteDict(longabs_db_path, autocommit=False)
    labels_db = SqliteDict(labels_db_path, autocommit=False)

    intersection_nodes = lookup_db.keys()

    count = 0

    for node in intersection_nodes:
        longab = longabs_db[node]
        label = labels_db[node]
        id = lookup_db[node]

        cursor.execute('''INSERT INTO graph VALUES (?,?,?)''', (id, longab, label))

        if count%100000 == 0:
            print(count)
            connection.commit()

        count += 1

    connection.commit()

    connection.close()
    lookup_db.close()
    labels_db.close()
    longabs_db.close()
