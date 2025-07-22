import sqlite3
import json

class AgentMemory:
    def __init__(self, db_path="memory.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS memory (id INTEGER PRIMARY KEY, data TEXT)"
        )

    def save(self, context):
        data = json.dumps(context)
        self.conn.execute("DELETE FROM memory")
        self.conn.execute("INSERT INTO memory (data) VALUES (?)", (data,))
        self.conn.commit()

    def load(self):
        cursor = self.conn.execute("SELECT data FROM memory ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return []
