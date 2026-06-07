# Dear AI — Gateway

Public-facing API gateway written in Go. It is the **only** entry point clients talk to. It verifies OIDC JWTs, mints short-lived internal PASETO tokens, and reverse-proxies WebSocket traffic to the AI service.

---

## Responsibilities

| Concern | Detail |
|---------|--------|
| **OIDC verification** | Validates the client's JWT against the issuer's OIDC discovery endpoint. Enforces `email_verified = true` and the presence of `sub` + `email` claims. |
| **PASETO minting** | Generates a V4-local symmetric token (15 s TTL) carrying the user's immutable OIDC `sub` as the token subject. |
| **Header scrubbing** | Strips `Authorization` before forwarding — the AI service never sees the raw OIDC JWT. |
| **Reverse proxy** | Forwards `/chat` and `/chat/` to the AI service, including WebSocket upgrade negotiation. |

---

## Routes

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/health` | Public | Liveness check |
| `GET/WS` | `/chat` | OIDC JWT | Proxied to AI service WebSocket endpoint |
| `GET/WS` | `/chat/` | OIDC JWT | Same — avoids redirect that breaks WS upgrades |

---

## Auth Flow

```
Client ──[Authorization: Bearer <oidc-jwt>]──► Gateway
  1. ExtractToken   → parse Bearer scheme
  2. OIDCVerifier   → verify sig, issuer, audience, email_verified
  3. PasetoManager  → Generate(claims.Subject, 15s)
  4. Set X-Internal-Auth: <paseto>
  5. Delete Authorization header
  6. ReverseProxy   → forward to AI service
```

The `sub` claim (the immutable OIDC subject, not the email) is embedded in the PASETO so the AI service can scope the user's knowledge graph correctly.

---

## Configuration

All configuration is read from environment variables at startup. The service will **fail fast** if required variables are missing or invalid.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ADDR` | no | `:8080` | TCP listen address |
| `ENV` | no | `development` | `development`, `staging`, or `production` |
| `BACKEND_WS` | **yes** | — | AI service URL — accepts `ws://`, `wss://`, `http://`, `https://` |
| `OIDC_ISSUER` | **yes** | — | OIDC issuer URL (must be a valid HTTPS URL) |
| `OIDC_CLIENT_ID` | **yes** | — | OIDC audience / client ID |
| `PASETO_SYMMETRIC_KEY` | **yes** | — | 64-char hex string (32 raw bytes). Must match `ai_service`. |
| `READ_TIMEOUT` | no | `30s` | HTTP server read timeout |
| `WRITE_TIMEOUT` | no | `30s` | HTTP server write timeout |
| `IDLE_TIMEOUT` | no | `120s` | HTTP server idle connection timeout |
| `ALLOWED_ORIGINS` | no* | — | Comma-separated CORS origins. **Required in `production`.** |

> `ALLOWED_ORIGINS` is enforced at startup when `ENV=production`. Omitting it in production causes the server to refuse to start.

### Example `.env`

```bash
ADDR=":8080"
ENV=development

BACKEND_WS=ws://ai_service:8000

OIDC_ISSUER=https://your-oidc-provider.example.com/
OIDC_CLIENT_ID=your-client-id

PASETO_SYMMETRIC_KEY=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
```

---

## Project Structure

```
gateway/
├── cmd/
│   └── main.go           # Entrypoint — wires config, auth, proxy, server
└── internal/
    ├── auth/
    │   ├── oidc_verifier.go  # OIDC JWT verification (go-oidc/v3)
    │   └── paseto.go         # PASETO V4-local token generation + verification
    ├── config/
    │   └── config.go         # Env-driven config with strict validation
    ├── middleware/
    │   └── middleware.go     # RequireAuth — verifies JWT, mints PASETO
    ├── proxy/
    │   └── proxy.go          # WebSocket-safe httputil.ReverseProxy
    ├── server/
    │   └── router.go         # Route registration
    └── utils/
        ├── token.go          # Bearer token extraction
        └── response.go       # JSON response helpers
```

---

## Running Locally

```bash
# Copy and fill in the required variables
cp .env.example .env

# Run
go run ./cmd/main.go
```

The gateway starts on `:8080` by default.

---

## Docker

```bash
# Build
docker build -t dearai-gateway .

# Run
docker run --rm -p 8080:8080 --env-file .env dearai-gateway
```

---

## Tests

```bash
go test ./...
```

All packages have unit tests using interface stubs — no live OIDC provider or AI service is needed to run the suite.

| Package | Coverage |
|---------|----------|
| `internal/auth` | Key validation, TTL bounds, generate+verify round-trip |
| `internal/config` | All `Validate()` branches |
| `internal/middleware` | Missing token, bad OIDC, PASETO gen failure, success, subject forwarding |
| `internal/proxy` | Invalid URL rejection, host rewriting, `ws://` acceptance |
| `internal/server` | Health route, auth-gated chat route |
| `internal/utils` | Bearer extraction, JSON response helpers |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `aidanwoods.dev/go-paseto` | PASETO V4 token generation and verification |
| `github.com/coreos/go-oidc/v3` | OIDC discovery and JWT verification |
| `github.com/google/uuid` | JTI generation for token uniqueness |
| `github.com/kelseyhightower/envconfig` | Struct-based environment variable loading |
