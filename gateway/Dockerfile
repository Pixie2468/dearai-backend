# ────────────────────────────────────────────────────────────
# Stage 1: Build
# ────────────────────────────────────────────────────────────
FROM golang:1.26-alpine AS builder

WORKDIR /app

# Download dependencies first so this layer is cached when only
# source files change.
COPY go.mod go.sum ./
RUN go mod download

COPY . .

# Build a fully static binary.
# -trimpath    → strip local file paths from stack traces (security + reproducibility)
# -ldflags     → strip DWARF debug info and symbol table to shrink binary size
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build \
      -trimpath \
      -ldflags="-s -w" \
      -o /gateway \
      ./cmd/main.go

# ────────────────────────────────────────────────────────────
# Stage 2: Run — scratch-based for minimal attack surface
# ────────────────────────────────────────────────────────────
FROM gcr.io/distroless/static-debian12:nonroot

# Copy CA certificates from the builder so HTTPS calls to the
# OIDC discovery endpoint work at runtime.
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

# Copy the compiled binary.
COPY --from=builder /gateway /gateway

EXPOSE 8080

# Run as non-root (distroless nonroot image sets uid=65532 by default).
ENTRYPOINT ["/gateway"]