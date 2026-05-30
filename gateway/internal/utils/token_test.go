package utils

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestExtractTokenMissingHeader(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/", nil)

	_, err := ExtractToken(req)
	if !errors.Is(err, ErrMissingAuthHeader) {
		t.Fatalf("expected ErrMissingAuthHeader, got %v", err)
	}
}

func TestExtractTokenInvalidFormat(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	req.Header.Set("Authorization", "Token abc")

	_, err := ExtractToken(req)
	if !errors.Is(err, ErrInvalidAuthFormat) {
		t.Fatalf("expected ErrInvalidAuthFormat, got %v", err)
	}
}

func TestExtractTokenSuccess(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	req.Header.Set("Authorization", "Bearer   abc123   ")

	token, err := ExtractToken(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if token != "abc123" {
		t.Fatalf("expected token abc123, got %q", token)
	}
}
