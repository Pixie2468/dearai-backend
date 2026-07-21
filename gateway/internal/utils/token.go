package utils

import (
	"errors"
	"net/http"
	"strings"
)

// Sentinel errors allow caller middleware to determine the exact failure reason
// and map it to the correct HTTP status code (e.g., 401 vs 400).
var (
	ErrMissingAuthHeader = errors.New("missing authorization header")
	ErrInvalidAuthFormat = errors.New("invalid authorization header format")
)

// ExtractToken safely retrieves a Bearer token from the HTTP request.
//
// It checks two sources in order:
//  1. Authorization: Bearer <token> header (preferred)
//  2. ?token=<token> query parameter (fallback for browser WebSocket clients
//     that cannot set custom headers on the upgrade request)
func ExtractToken(r *http.Request) (string, error) {
	authHeader := r.Header.Get("Authorization")

	// ── Fallback: query parameter ──
	// Browser WebSocket API does not support custom headers, so the test
	// client passes the JWT as ?token=<jwt> on the upgrade URL.
	if authHeader == "" {
		if qToken := r.URL.Query().Get("token"); qToken != "" {
			return qToken, nil
		}
		return "", ErrMissingAuthHeader
	}

	// ── Primary: Authorization header ──
	// strings.Cut safely splits at the first space, avoiding the panic/length
	// risks of strings.Split, and handles multiple trailing spaces gracefully.
	scheme, token, found := strings.Cut(authHeader, " ")

	// EqualFold performs case-insensitive comparison without memory allocation
	if !found || !strings.EqualFold(scheme, "Bearer") {
		return "", ErrInvalidAuthFormat
	}

	// Trim any accidental leading/trailing spaces from the token side
	token = strings.TrimSpace(token)
	if token == "" {
		return "", ErrInvalidAuthFormat
	}

	return token, nil
}
