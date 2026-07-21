package server

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Pixie2468/dearai-backend/gateway/internal/auth"
	"github.com/Pixie2468/dearai-backend/gateway/internal/proxy"
)

type tokenVerifierStub struct {
	claims *auth.ExternalClaims
	err    error
}

func (v tokenVerifierStub) Verify(ctx context.Context, tokenString string) (*auth.ExternalClaims, error) {
	return v.claims, v.err
}

type tokenManagerStub struct {
	token string
	err   error
}

func (m tokenManagerStub) Generate(userID string, ttl time.Duration) (string, error) {
	return m.token, m.err
}

func (m tokenManagerStub) Verify(tokenString string) (*auth.InternalClaims, error) {
	return nil, errors.New("not implemented")
}

func TestRouterHealth(t *testing.T) {
	proxyHandler, err := proxy.NewProxy("http://example.com")
	if err != nil {
		t.Fatalf("unexpected proxy error: %v", err)
	}

	router := NewRouter(tokenVerifierStub{}, tokenManagerStub{}, proxyHandler)

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}

	var payload map[string]string
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	if payload["status"] == "" {
		t.Fatalf("expected status message")
	}
}

func TestRouterChatRoute(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))
	defer backend.Close()

	proxyHandler, err := proxy.NewProxy(backend.URL)
	if err != nil {
		t.Fatalf("unexpected proxy error: %v", err)
	}

	verifier := tokenVerifierStub{claims: &auth.ExternalClaims{Email: "user@example.com"}}
	manager := tokenManagerStub{token: "internal"}
	router := NewRouter(verifier, manager, proxyHandler)

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/chat", nil)
	req.Header.Set("Authorization", "Bearer ok")
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusNoContent {
		t.Fatalf("expected status %d, got %d", http.StatusNoContent, rec.Code)
	}
}
