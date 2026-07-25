import os
os.environ["PASETO_SYMMETRIC_KEY"] = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
os.environ["DATABASE_URL"] = "sqlite:///./test.db"

from fastapi.testclient import TestClient
from app.main import app
from app.database import Base, engine
from app.auth import PASETO_KEY
from pyseto import encode
import datetime

Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)

client = TestClient(app)

now = datetime.datetime.now(datetime.timezone.utc)
exp = (now + datetime.timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
payload = {
    "iss": "dear-ai-gateway",
    "aud": "dear-ai-python-backend",
    "sub": "test_user",
    "exp": exp
}
token = encode(PASETO_KEY, payload).decode("utf-8")
headers = {"X-Internal-Auth": token}

res = client.post("/sessions", json={"title": "Test Session"}, headers=headers)
print("Create Session:", res.status_code, res.text)
if res.status_code == 200:
    session_id = res.json()["id"]
    res = client.post("/chats", json={"session_id": session_id, "role": "user", "content": "hi"}, headers=headers)
    print("Create Chat:", res.status_code, res.text)
    res = client.get(f"/chats?session_id={session_id}", headers=headers)
    print("Get Chats:", res.status_code, res.text)
