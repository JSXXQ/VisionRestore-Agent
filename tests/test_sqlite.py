from visionrestore.storage.database import Database

def test_database_settings_roundtrip():
    db = Database()
    db.set_settings({"x": {"y": 1}})
    assert db.get_settings()["x"]["y"] == 1
