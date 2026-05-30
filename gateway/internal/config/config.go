package config

import (
	"fmt"
	"net/url"
	"time"

	"github.com/kelseyhightower/envconfig"
)

type Config struct {
	Addr string `envconfig:"ADDR" default:":8080"`

	// Constrained to specific values in Validate()
	Environment string `envconfig:"ENV" default:"development"`

	BackendWS string `envconfig:"BACKEND_WS" required:"true"`

	IssuerURL        string `envconfig:"ISSUER_URL" required:"true"`
	AudienceClientID string `envconfig:"AUDIENCE_CLIENT_ID" required:"true"`

	// Sensitive Data
	PasetoSymmetricKey string `envconfig:"PASETO_SYMMETRIC_KEY" required:"true"`

	ReadTimeout  time.Duration `envconfig:"READ_TIMEOUT" default:"30s"`
	WriteTimeout time.Duration `envconfig:"WRITE_TIMEOUT" default:"30s"`
	IdleTimeout  time.Duration `envconfig:"IDLE_TIMEOUT" default:"120s"`

	AllowedOrigins []string `envconfig:"ALLOWED_ORIGINS"`
}

// String implements the fmt.Stringer interface.
// This ensures that if the Config struct is ever logged, secrets are safely redacted.
func (c Config) String() string {
	redacted := c
	if redacted.PasetoSymmetricKey != "" {
		redacted.PasetoSymmetricKey = "[REDACTED]"
	}
	// Return the struct representation with masked secrets
	type safeConfig Config
	return fmt.Sprintf("%+v", safeConfig(redacted))
}

// Load reads environment variables and returns a validated Config.
func Load() (*Config, error) {
	var cfg Config

	if err := envconfig.Process("", &cfg); err != nil {
		return nil, fmt.Errorf("failed to process environment variables: %w", err)
	}

	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("configuration validation failed: %w", err)
	}

	return &cfg, nil
}

// Validate ensures complex rules and formats are strictly adhered to.
func (c *Config) Validate() error {
	// 1. Strict Environment Constraints
	switch c.Environment {
	case "development", "staging", "production":
		// Valid environments
	default:
		return fmt.Errorf("invalid ENV '%s': must be development, staging, or production", c.Environment)
	}

	// 2. Backend URL Validation
	wsURL, err := url.Parse(c.BackendWS)
	if err != nil {
		return fmt.Errorf("invalid BACKEND_WS url: %w", err)
	}
	if wsURL.Scheme != "ws" && wsURL.Scheme != "wss" && wsURL.Scheme != "http" && wsURL.Scheme != "https" {
		return fmt.Errorf("BACKEND_WS scheme must be 'ws', 'wss', 'http', or 'https', got '%s'", wsURL.Scheme)
	}
	if wsURL.Host == "" {
		return fmt.Errorf("BACKEND_WS must include a valid host (e.g., ws://localhost:8080)")
	}

	// 3. OIDC Issuer Validation
	if _, err := url.ParseRequestURI(c.IssuerURL); err != nil {
		return fmt.Errorf("invalid ISSUER_URL format: %w", err)
	}

	// 4. CORS Safety Check
	if len(c.AllowedOrigins) == 0 && c.Environment == "production" {
		// Optional: Warn or fail if production boots without explicit CORS boundaries
		return fmt.Errorf("ALLOWED_ORIGINS cannot be empty in production")
	}

	return nil
}
