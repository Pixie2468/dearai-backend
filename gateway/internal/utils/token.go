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
func ExtractToken(r *http.Request) (string, error) {
	authHeader := r.Header.Get("Authorization")
	if authHeader == "" {
		return "", ErrMissingAuthHeader
	}

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
