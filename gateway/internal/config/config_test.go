package config

import (
	"strings"
	"testing"
)

func baseConfig() *Config {
	return &Config{
		Environment:      "development",
		BackendWS:        "http://localhost:8080",
		IssuerURL:        "https://issuer.example.com",
		AudienceClientID: "audience",
		PasetoSymmetricKey: "dummy",
	}
}

func TestValidateEnvironment(t *testing.T) {
	cfg := baseConfig()
	cfg.Environment = "dev"

	if err := cfg.Validate(); err == nil {
		t.Fatalf("expected error for invalid environment")
	}
}

func TestValidateBackendScheme(t *testing.T) {
	cfg := baseConfig()
	cfg.BackendWS = "ftp://localhost:8080"

	if err := cfg.Validate(); err == nil {
		t.Fatalf("expected error for invalid backend scheme")
	}
}

func TestValidateBackendHost(t *testing.T) {
	cfg := baseConfig()
	cfg.BackendWS = "http:///missing-host"

	if err := cfg.Validate(); err == nil {
		t.Fatalf("expected error for missing backend host")
	}
}

func TestValidateIssuerURL(t *testing.T) {
	cfg := baseConfig()
	cfg.IssuerURL = "not-a-url"

	if err := cfg.Validate(); err == nil {
		t.Fatalf("expected error for invalid issuer URL")
	}
}

func TestValidateProductionRequiresOrigins(t *testing.T) {
	cfg := baseConfig()
	cfg.Environment = "production"
	cfg.AllowedOrigins = nil

	if err := cfg.Validate(); err == nil {
		t.Fatalf("expected error for missing allowed origins in production")
	}
}

func TestValidateAcceptsHTTPBackend(t *testing.T) {
	cfg := baseConfig()
	cfg.BackendWS = "https://backend.example.com"

	if err := cfg.Validate(); err != nil {
		t.Fatalf("expected valid config, got error: %v", err)
	}
}

func TestConfigStringRedactsSecrets(t *testing.T) {
	cfg := baseConfig()
	cfg.PasetoSymmetricKey = "secret"

	got := cfg.String()
	if got == "" {
		t.Fatalf("expected non-empty string")
	}
	if strings.Contains(got, "secret") {
		t.Fatalf("expected secret to be redacted")
	}
	if !strings.Contains(got, "[REDACTED]") {
		t.Fatalf("expected redacted marker in string")
	}
}

func TestConfigStringEmptySecret(t *testing.T) {
	cfg := baseConfig()
	cfg.PasetoSymmetricKey = ""

	if got := cfg.String(); got == "" {
		t.Fatalf("expected non-empty string")
	}
}
