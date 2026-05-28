package auth

import (
	"context"
	"errors"
	"fmt"

	"github.com/coreos/go-oidc/v3/oidc"
)

// TokenVerifier defines the contract for verifying authentication tokens.
// Returning an interface enables easy mocking in consumer packages.
type TokenVerifier interface {
	Verify(ctx context.Context, tokenString string) (*ExternalClaims, error)
}

// ExternalClaims represents the expected payload from the OIDC provider.
type ExternalClaims struct {
	Subject       string `json:"sub"`            // Immutable unique identifier (Primary Key)
	Email         string `json:"email"`          // Mutable user attribute
	EmailVerified *bool  `json:"email_verified"` // Security flag
}

// oidcVerifier is the concrete implementation of TokenVerifier.
// It is unexported to force consumers to depend on the interface.
type OIDCVerifier struct {
	verifier *oidc.IDTokenVerifier
}

// NewOIDCVerifier initializes the provider and returns a TokenVerifier.
// NOTE: This performs a blocking network call to the issuer's OIDC discovery endpoint.
func NewOIDCVerifier(ctx context.Context, issuer, clientID string) (TokenVerifier, error) {
	if issuer == "" || clientID == "" {
		return nil, errors.New("issuer and clientID are required")
	}

	// Network I/O occurs here. Ensure ctx has a reasonable timeout.
	provider, err := oidc.NewProvider(ctx, issuer)
	if err != nil {
		return nil, fmt.Errorf("failed to discover OIDC configuration: %w", err)
	}

	return &OIDCVerifier{
		verifier: provider.Verifier(&oidc.Config{ClientID: clientID}),
	}, nil
}

// Verify parses the token, validates its signature/audience, and strictly enforces claims.
func (v *OIDCVerifier) Verify(ctx context.Context, tokenString string) (*ExternalClaims, error) {
	idToken, err := v.verifier.Verify(ctx, tokenString)
	if err != nil {
		return nil, fmt.Errorf("cryptographic token verification failed: %w", err)
	}

	var claims ExternalClaims
	if err := idToken.Claims(&claims); err != nil {
		return nil, fmt.Errorf("failed to unmarshal token claims: %w", err)
	}

	// Security: Ensure we have a valid, immutable subject
	if claims.Subject == "" {
		return nil, errors.New("invalid token: missing 'sub' claim")
	}

	// Security: Ensure email exists AND is verified by the Identity Provider
	if claims.Email == "" {
		return nil, errors.New("invalid token: missing 'email' claim")
	}
	if claims.EmailVerified == nil || !*claims.EmailVerified {
		return nil, errors.New("invalid token: email address is unverified")
	}

	return &claims, nil
}
