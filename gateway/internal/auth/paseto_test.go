package auth

import (
	"encoding/hex"
	"strings"
	"testing"
	"time"
)

func TestNewPasetoManagerValidatesKey(t *testing.T) {
	_, err := NewPasetoManager(PasetoConfig{SymmetricKey: "not-hex"})
	if err == nil {
		t.Fatalf("expected error for invalid key encoding")
	}

	_, err = NewPasetoManager(PasetoConfig{SymmetricKey: hex.EncodeToString([]byte("short"))})
	if err == nil {
		t.Fatalf("expected error for invalid key length")
	}
}

func TestPasetoGenerateValidateTTL(t *testing.T) {
	key := hex.EncodeToString(make([]byte, 32))
	manager, err := NewPasetoManager(PasetoConfig{
		SymmetricKey: key,
		MinTokenTTL:  10 * time.Second,
		MaxTokenTTL:  20 * time.Second,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if _, err := manager.Generate("user", 5*time.Second); err == nil {
		t.Fatalf("expected error for ttl below minimum")
	}
	if _, err := manager.Generate("user", 30*time.Second); err == nil {
		t.Fatalf("expected error for ttl above maximum")
	}
}

func TestPasetoGenerateAndVerify(t *testing.T) {
	key := hex.EncodeToString(make([]byte, 32))
	manager, err := NewPasetoManager(PasetoConfig{
		SymmetricKey: key,
		Issuer:       "issuer",
		Audience:     "audience",
		MinTokenTTL:  1 * time.Second,
		MaxTokenTTL:  10 * time.Second,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	token, err := manager.Generate("user-id", 2*time.Second)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	claims, err := manager.Verify(token)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if claims.UserID != "user-id" {
		t.Fatalf("expected user-id, got %q", claims.UserID)
	}
	if strings.TrimSpace(claims.TokenID) == "" {
		t.Fatalf("expected token id")
	}
}
