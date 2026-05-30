package proxy

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
)

func TestNewProxyRejectsInvalidURL(t *testing.T) {
	if _, err := NewProxy("://bad"); err == nil {
		t.Fatalf("expected error for invalid URL")
	}
}

func TestNewProxyRewritesTargetHost(t *testing.T) {
	var gotHost string
	var gotPath string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotHost = r.Host
		gotPath = r.URL.Path
		w.WriteHeader(http.StatusNoContent)
	}))
	defer backend.Close()

	proxyHandler, err := NewProxy(backend.URL)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	backendURL, err := url.Parse(backend.URL)
	if err != nil {
		t.Fatalf("failed to parse backend url: %v", err)
	}

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "http://gateway.local/chat", nil)
	proxyHandler.ServeHTTP(rec, req)

	if rec.Code != http.StatusNoContent {
		t.Fatalf("expected status %d, got %d", http.StatusNoContent, rec.Code)
	}
	if gotHost != backendURL.Host {
		t.Fatalf("expected host %q, got %q", backendURL.Host, gotHost)
	}
	if gotPath != "/chat" {
		t.Fatalf("expected path /chat, got %q", gotPath)
	}
}

func TestNewProxyAcceptsWebsocketTarget(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer backend.Close()

	backendURL, err := url.Parse(backend.URL)
	if err != nil {
		t.Fatalf("failed to parse backend url: %v", err)
	}

	proxyHandler, err := NewProxy("ws://" + backendURL.Host)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "http://gateway.local/chat", nil)
	proxyHandler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}
}
