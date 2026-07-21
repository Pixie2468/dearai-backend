package middleware

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Pixie2468/dearai-backend/gateway/internal/auth"
)

type tokenVerifierStub struct {
	claims *auth.ExternalClaims
	err    error
}

func (v tokenVerifierStub) Verify(ctx context.Context, tokenString string) (*auth.ExternalClaims, error) {
	return v.claims, v.err
}

type tokenManagerStub struct {
	token      string
	err        error
	capturedID string // set to the userID passed to Generate
}

func (m *tokenManagerStub) Generate(userID string, ttl time.Duration) (string, error) {
	m.capturedID = userID
	return m.token, m.err
}

func (m *tokenManagerStub) Verify(tokenString string) (*auth.InternalClaims, error) {
	return nil, errors.New("not implemented")
}

func TestRequireAuthMissingToken(t *testing.T) {
	called := false

	verifier := tokenVerifierStub{}
	manager := &tokenManagerStub{}
	h := RequireAuth(verifier, manager, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
	}))

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/", nil)

	h.ServeHTTP(rec, req)

	if called {
		t.Fatalf("handler should not be called")
	}
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("expected status %d, got %d", http.StatusUnauthorized, rec.Code)
	}
}

func TestRequireAuthVerifierError(t *testing.T) {
	called := false
	verifier := tokenVerifierStub{err: errors.New("bad token")}
	manager := &tokenManagerStub{}

	h := RequireAuth(verifier, manager, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
	}))

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	req.Header.Set("Authorization", "Bearer valid")

	h.ServeHTTP(rec, req)

	if called {
		t.Fatalf("handler should not be called")
	}
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("expected status %d, got %d", http.StatusUnauthorized, rec.Code)
	}
}

func TestRequireAuthTokenGenerationError(t *testing.T) {
	called := false
	verifier := tokenVerifierStub{claims: &auth.ExternalClaims{Subject: "oidc-sub-123", Email: "user@example.com"}}
	manager := &tokenManagerStub{err: errors.New("generate failed")}

	h := RequireAuth(verifier, manager, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
	}))

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	req.Header.Set("Authorization", "Bearer valid")

	h.ServeHTTP(rec, req)

	if called {
		t.Fatalf("handler should not be called")
	}
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("expected status %d, got %d", http.StatusInternalServerError, rec.Code)
	}
}

func TestRequireAuthSuccess(t *testing.T) {
	called := false
	verifier := tokenVerifierStub{claims: &auth.ExternalClaims{Subject: "oidc-sub-123", Email: "user@example.com"}}
	manager := &tokenManagerStub{token: "internal-token"}

	h := RequireAuth(verifier, manager, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		if got := r.Header.Get("X-Internal-Auth"); got != "internal-token" {
			t.Fatalf("expected internal token header, got %q", got)
		}
		if got := r.Header.Get("Authorization"); got != "" {
			t.Fatalf("expected Authorization header to be removed, got %q", got)
		}
		w.WriteHeader(http.StatusOK)
	}))

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	req.Header.Set("Authorization", "Bearer valid")

	h.ServeHTTP(rec, req)

	if !called {
		t.Fatalf("handler should be called")
	}
	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}
}

// TestRequireAuthUsesSubjectAsUserID verifies the middleware forwards the
// immutable OIDC 'sub' claim — NOT the mutable email — to the PASETO manager.
func TestRequireAuthUsesSubjectAsUserID(t *testing.T) {
	const oidcSub = "google-oauth2|987654321"
	verifier := tokenVerifierStub{claims: &auth.ExternalClaims{
		Subject: oidcSub,
		Email:   "user@example.com",
	}}
	manager := &tokenManagerStub{token: "tok"}

	h := RequireAuth(verifier, manager, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	req.Header.Set("Authorization", "Bearer valid")
	h.ServeHTTP(rec, req)

	if manager.capturedID != oidcSub {
		t.Fatalf("expected PASETO to be generated with sub=%q, got %q", oidcSub, manager.capturedID)
	}
}
