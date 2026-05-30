package auth

import (
	"context"
	"testing"
)

func TestNewOIDCVerifierRequiresFields(t *testing.T) {
	if _, err := NewOIDCVerifier(context.Background(), "", "client"); err == nil {
		t.Fatalf("expected error for missing issuer")
	}
	if _, err := NewOIDCVerifier(context.Background(), "https://issuer", ""); err == nil {
		t.Fatalf("expected error for missing clientID")
	}
}
