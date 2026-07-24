package server

import (
	"net/http"

	"github.com/Pixie2468/dearai-backend/gateway/internal/auth"
	"github.com/Pixie2468/dearai-backend/gateway/internal/middleware"
	"github.com/Pixie2468/dearai-backend/gateway/internal/proxy"
	"github.com/Pixie2468/dearai-backend/gateway/internal/utils"
)

// NewRouter initializes the API Gateway routes using interface dependencies.
func NewRouter(
	verifier auth.TokenVerifier,
	pasetoManager auth.TokenManager,
	p *proxy.Proxy,
	chatProxy *proxy.Proxy,
	diaryProxy *proxy.Proxy,
	agentProxy *proxy.Proxy,
) http.Handler { // Return http.Handler to allow global middleware wrapping

	mux := http.NewServeMux()

	// 1. Initialize the middleware with our dependencies and wrap the proxy handler.
	authHandler := middleware.RequireAuth(verifier, pasetoManager, p)

	// 2. Register the wrapped proxy handler for both exact and subtree paths.
	// This avoids redirecting websocket upgrades from /chat -> /chat/.
	mux.Handle("/chat", authHandler)
	mux.Handle("/chat/", authHandler)

	// TTS endpoint — proxied to AI service with the same auth pipeline.
	mux.Handle("/tts", authHandler)

	// STT endpoint — proxied to AI service with the same auth pipeline.
	mux.Handle("/stt", authHandler)

	// Chat service endpoints
	mux.Handle("/api/chat", middleware.RequireAuth(verifier, pasetoManager, chatProxy))
	mux.Handle("/api/chat/", middleware.RequireAuth(verifier, pasetoManager, chatProxy))

	// Diary service endpoints
	mux.Handle("/api/diary", middleware.RequireAuth(verifier, pasetoManager, diaryProxy))
	mux.Handle("/api/diary/", middleware.RequireAuth(verifier, pasetoManager, diaryProxy))

	// Agent service endpoints
	mux.Handle("/api/agent", middleware.RequireAuth(verifier, pasetoManager, agentProxy))
	mux.Handle("/api/agent/", middleware.RequireAuth(verifier, pasetoManager, agentProxy))

	// 3. Register public routes (Using Go 1.22+ strict method routing)
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		utils.RespondJSON(w, http.StatusOK, map[string]string{
			"status": "Gateway is running",
		})
	})

	// 4. Wrap the entire mux in global middleware (CORS)
	return middleware.CORS(mux)
}
