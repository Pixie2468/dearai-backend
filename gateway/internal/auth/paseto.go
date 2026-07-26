package auth

import (
	"encoding/hex"
	"errors"
	"fmt"
	"time"

	"aidanwoods.dev/go-paseto"
	"github.com/google/uuid"
)

// TokenManager defines the contract for generating and verifying internal tokens.
// This allows consumers to mock token logic in unit tests.
type TokenManager interface {
	Generate(userID string, ttl time.Duration) (string, error)
	Verify(tokenString string) (*InternalClaims, error)
}

type PasetoConfig struct {
	SymmetricKey string
	Issuer       string
	Audience     string
	MinTokenTTL  time.Duration
	MaxTokenTTL  time.Duration
}

type InternalClaims struct {
	UserID  string
	TokenID string
}

// pasetoManager is unexported to force the use of the interface.
type PasetoManager struct {
	key    paseto.V4SymmetricKey
	parser paseto.Parser
	config PasetoConfig
}

// NewPasetoManager initializes the token manager and pre-configures the parser rules.
func NewPasetoManager(cfg PasetoConfig) (TokenManager, error) {
	if cfg.SymmetricKey == "" {
		return nil, errors.New("missing symmetric key")
	}

	keyBytes, err := hex.DecodeString(cfg.SymmetricKey)
	if err != nil {
		return nil, fmt.Errorf("invalid symmetric key encoding: %w", err)
	}

	if len(keyBytes) != 32 {
		return nil, errors.New("symmetric key must be exactly 32 bytes")
	}

	key, err := paseto.V4SymmetricKeyFromBytes(keyBytes)
	if err != nil {
		return nil, fmt.Errorf("failed to create symmetric key: %w", err)
	}

	if cfg.MinTokenTTL == 0 {
		cfg.MinTokenTTL = 15 * time.Second
	}
	if cfg.MaxTokenTTL == 0 {
		cfg.MaxTokenTTL = 2 * time.Minute
	}

	// Pre-configure the parser with our strict business rules.
	// NewParser() automatically includes time-based rules (exp, nbf, iat).
	parser := paseto.NewParser()
	if cfg.Issuer != "" {
		parser.AddRule(paseto.IssuedBy(cfg.Issuer))
	}
	if cfg.Audience != "" {
		parser.AddRule(paseto.ForAudience(cfg.Audience))
	}

	return &PasetoManager{
		key:    key,
		parser: parser,
		config: cfg,
	}, nil
}

func (p *PasetoManager) Generate(userID string, ttl time.Duration) (string, error) {
	if userID == "" {
		return "", errors.New("missing user id")
	}
	if ttl < p.config.MinTokenTTL || ttl > p.config.MaxTokenTTL {
		return "", fmt.Errorf("token ttl %v is outside allowed bounds [%v, %v]", ttl, p.config.MinTokenTTL, p.config.MaxTokenTTL)
	}

	now := time.Now().UTC()
	token := paseto.NewToken()

	token.SetIssuedAt(now)
	token.SetNotBefore(now.Add(-2 * time.Second)) // Clock skew adjustment
	token.SetExpiration(now.Add(ttl))
	token.SetSubject(userID)

	// Add a unique token ID for potential future revocation needs
	token.SetJti(uuid.NewString())

	if p.config.Issuer != "" {
		token.SetIssuer(p.config.Issuer)
	}
	if p.config.Audience != "" {
		token.SetAudience(p.config.Audience)
	}

	return token.V4Encrypt(p.key, nil), nil
}

func (p *PasetoManager) Verify(tokenString string) (*InternalClaims, error) {
	if tokenString == "" {
		return nil, errors.New("missing token")
	}

	// ParseV4Local now automatically enforces time, issuer, and audience
	// based on the rules we set in NewPasetoManager.
	token, err := p.parser.ParseV4Local(p.key, tokenString, nil)
	if err != nil {
		return nil, fmt.Errorf("invalid or rejected token: %w", err)
	}

	userID, err := token.GetSubject()
	if err != nil || userID == "" {
		return nil, errors.New("invalid token: missing subject")
	}

	tokenID, _ := token.GetJti() // Ignore error, it will just be empty if not found

	return &InternalClaims{
		UserID:  userID,
		TokenID: tokenID,
	}, nil
}
