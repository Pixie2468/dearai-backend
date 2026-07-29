package middleware

import (
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/Pixie2468/dearai-backend/gateway/internal/auth"
	"github.com/Pixie2468/dearai-backend/gateway/internal/utils"
)

// cachedToken holds a PASETO token and when it was created.
type cachedToken struct {
	token     string
	createdAt time.Time
}

// pasetoCache is a global, concurrency-safe cache for internal PASETO tokens.
// Keyed by user subject (sub claim). Tokens are reused for up to
// pasetoReuseTTL to avoid repeated crypto work on rapid successive requests.
var pasetoCache sync.Map

// pasetoReuseTTL is how long a cached PASETO token is reused before
// generating a fresh one. Must be shorter than the token's own TTL (90s)
// to ensure backends never receive an expired token.
const pasetoReuseTTL = 60 * time.Second

func RequireAuth(
	oidcVerifier auth.TokenVerifier,
	pasetoManager auth.TokenManager,
	next http.Handler,
) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		token, err := utils.ExtractToken(r)
		if err != nil {
			utils.RespondError(w, http.StatusUnauthorized, "unauthorized: missing or malformed token")
			return
		}

		claims, err := oidcVerifier.Verify(r.Context(), token)
		if err != nil {
			log.Printf("OIDC verification failed: %v", err)
			utils.RespondError(w, http.StatusUnauthorized, "unauthorized: invalid token")
			return
		}

		// Attempt to reuse a cached PASETO token for this user.
		internalToken, ok := getCachedPaseto(claims.Subject)
		if !ok {
			internalToken, err = pasetoManager.Generate(claims.Subject, 90*time.Second)
			if err != nil {
				log.Printf("PASETO generation failed: %v", err)
				utils.RespondError(w, http.StatusInternalServerError, "internal server error")
				return
			}
			pasetoCache.Store(claims.Subject, cachedToken{
				token:     internalToken,
				createdAt: time.Now(),
			})
		}

		r.Header.Set("X-Internal-Auth", internalToken)
		r.Header.Del("Authorization")

		next.ServeHTTP(w, r)
	})
}

// getCachedPaseto returns a cached token if it exists and hasn't expired.
func getCachedPaseto(subject string) (string, bool) {
	val, ok := pasetoCache.Load(subject)
	if !ok {
		return "", false
	}
	cached := val.(cachedToken)
	if time.Since(cached.createdAt) > pasetoReuseTTL {
		pasetoCache.Delete(subject)
		return "", false
	}
	return cached.token, true
}

// CORS middleware handles Cross-Origin Resource Sharing
func CORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

		// Handle preflight OPTIONS request
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}

		next.ServeHTTP(w, r)
	})
}
