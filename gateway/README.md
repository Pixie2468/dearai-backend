# Gateway Service

API gateway for the Dear AI backend. It validates incoming OIDC tokens, mints short-lived internal PASETO tokens, and reverse-proxies chat traffic to the backend service.

## Routes

- `GET /health` - Liveness check
- `/chat` and `/chat/` - Proxied to the backend (supports WebSocket upgrades)

## Configuration

Environment variables (service-specific):

| Variable | Required | Default | Notes |
| --- | --- | --- | --- |
| `ADDR` | no | `:8080` | Bind address for the gateway |
| `ENV` | no | `development` | `development`, `staging`, or `production` |
| `BACKEND_WS` | yes | — | Backend URL (`ws`, `wss`, `http`, or `https`) |
| `ISSUER_URL` | yes | — | OIDC issuer URL |
| `AUDIENCE_CLIENT_ID` | yes | — | OIDC client ID (audience) |
| `PASETO_SYMMETRIC_KEY` | yes | — | 32-byte hex string (64 hex chars) |
| `READ_TIMEOUT` | no | `30s` | Server read timeout |
| `WRITE_TIMEOUT` | no | `30s` | Server write timeout |
| `IDLE_TIMEOUT` | no | `120s` | Server idle timeout |
| `ALLOWED_ORIGINS` | no | — | Required in production; comma-separated list |

Example:

```bash
ADDR=":8080"
ENV=development
BACKEND_WS="ws://localhost:8000"
ISSUER_URL="https://your-issuer.example.com/"
AUDIENCE_CLIENT_ID="your-client-id"
PASETO_SYMMETRIC_KEY="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
READ_TIMEOUT="30s"
WRITE_TIMEOUT="30s"
IDLE_TIMEOUT="120s"
ALLOWED_ORIGINS="http://localhost:3000"
```

## Running locally

```bash
go run ./cmd/main.go
```

## Docker

```bash
docker build -t dearai-gateway .
docker run --rm -p 8080:8080 --env-file .env dearai-gateway
```

## Testing

```bash
go test ./...
```
