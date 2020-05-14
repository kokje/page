import sqlite3


def init_db(schema, path):
    handle = None
    try:
        handle = sqlite3.connect(path)
        if schema:
            with open(schema, mode='r') as f:
                handle.cursor().execute(f.read())
                handle.commit()
    finally:
        if handle:
            handle.close()


def add_study(name, size, path):
    handle = None
    try:
        handle = sqlite3.connect(path)
        sql = """INSERT INTO study(name, size, active) VALUES(?,?,?)"""
        curr = handle.cursor()
        curr.execute(sql, (name, size, 1))
        handle.commit()
        return curr.lastrowid
    finally:
        if handle:
            handle.close()


def get_study_size(name, path):
    handle = None
    try:
        handle = sqlite3.connect(path)
        query = """SELECT * from study WHERE name=?"""
        curr = handle.cursor()
        curr.execute(query, (name,))
        record = curr.fetchone()
        return record[1]
    finally:
        if handle:
            handle.close()


def add_new_round(self, name, path):
    pass


def end_study(self, name, path):
    pass
