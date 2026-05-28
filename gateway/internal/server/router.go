package server

import (
	"net/http"

	"github.com/Pixie2468/dearai-backend/internal/auth"
	"github.com/Pixie2468/dearai-backend/internal/middleware"
	"github.com/Pixie2468/dearai-backend/internal/proxy"
	"github.com/Pixie2468/dearai-backend/internal/utils"
)

// NewRouter initializes the API Gateway routes using interface dependencies.
func NewRouter(
	verifier auth.TokenVerifier,
	pasetoManager auth.TokenManager,
	p *proxy.Proxy,
) http.Handler { // Return http.Handler to allow global middleware wrapping

	mux := http.NewServeMux()

	// 1. Initialize the middleware with our dependencies and wrap the proxy handler.
	// middleware.RequireAuth expects concrete types and the handler to wrap.
	// Use type assertions to convert the provided interfaces to the concrete types.
	authHandler := middleware.RequireAuth(verifier.(*auth.OIDCVerifier), pasetoManager.(*auth.PasetoManager), p)

	// 2. Register the wrapped proxy handler.
	// Note: We use a trailing slash "/chat/" so it acts as a prefix router,
	// capturing "/chat", "/chat/", and "/chat/subpath"
	mux.Handle("/chat/", authHandler)

	// 3. Register public routes (Using Go 1.22+ strict method routing)
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		utils.RespondJSON(w, http.StatusOK, map[string]string{
			"status": "Gateway is running",
		})
	})

	// 4. (Optional but recommended) Wrap the entire mux in global middleware
	// return middleware.Recoverer(middleware.Logger(mux))
	return mux
}
