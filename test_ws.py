import json

payload = {
    "audio": "dummy",
    "voice_mode": True,
    "session_id": "test"
}
print(json.dumps(payload))
