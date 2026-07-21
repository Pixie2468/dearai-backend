package middleware

import (
	"log"
	"net/http"
	"time"

	"github.com/Pixie2468/dearai-backend/gateway/internal/auth"
	"github.com/Pixie2468/dearai-backend/gateway/internal/utils"
)

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

		internalToken, err := pasetoManager.Generate(claims.Subject, 15*time.Second)
		if err != nil {
			log.Printf("PASETO generation failed: %v", err)
			utils.RespondError(w, http.StatusInternalServerError, "internal server error")
			return
		}

		r.Header.Set("X-Internal-Auth", internalToken)
		r.Header.Del("Authorization")

		next.ServeHTTP(w, r)
	})
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
