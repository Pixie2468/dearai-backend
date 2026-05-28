package main

import (
	"context"
	"errors"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/Pixie2468/dearai-backend/internal/auth"
	"github.com/Pixie2468/dearai-backend/internal/config"
	"github.com/Pixie2468/dearai-backend/internal/proxy"
	"github.com/Pixie2468/dearai-backend/internal/server"
)

func main() {
	// 1. Load Configuration
	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	// 2. Setup Context with OS Signal Interception (Go 1.16+)
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	// 3. Initialize Auth Dependencies (with a strict startup timeout)
	startupCtx, cancelStartup := context.WithTimeout(ctx, 15*time.Second)
	verifier, pasetoManager := initAuth(startupCtx, cfg)
	cancelStartup() // Free resources early

	// 4. Initialize the Proxy
	revProxy, err := proxy.NewProxy(cfg.BackendWS)
	if err != nil {
		log.Fatalf("Failed to initialize proxy: %v", err)
	}

	// 5. Wire the Router (Expecting interfaces from previous refactors)
	mux := server.NewRouter(verifier, pasetoManager, revProxy)

	srv := &http.Server{
		Addr:         cfg.Addr,
		Handler:      mux,
		ReadTimeout:  cfg.ReadTimeout,
		WriteTimeout: cfg.WriteTimeout,
		IdleTimeout:  cfg.IdleTimeout,
	}

	// 6. Start the Server Asynchronously
	go func() {
		log.Printf("Gateway starting on %s in %s mode...", cfg.Addr, cfg.Environment)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("Server crashed unexpectedly: %v", err)
		}
	}()

	// 7. Block until OS signal is received
	<-ctx.Done()
	log.Println("Interrupt signal received. Shutting down gracefully...")

	// 8. Graceful Shutdown (Drain active connections)
	shutdownCtx, cancelShutdown := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancelShutdown()

	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Fatalf("Server forced to shutdown: %v", err)
	}

	log.Println("Server exited cleanly")
}

// initAuth returns interfaces (auth.TokenVerifier, auth.TokenManager) to maintain loose coupling.
func initAuth(ctx context.Context, cfg *config.Config) (auth.TokenVerifier, auth.TokenManager) {

	// Ensure the OIDC discovery doesn't hang forever
	oidcVerifier, err := auth.NewOIDCVerifier(ctx, cfg.IssuerURL, cfg.AudienceClientID)
	if err != nil {
		log.Fatalf("Failed to initialize OIDC verifier (is the network up?): %v", err)
	}

	// Note: Consider moving Issuer and Audience to config.Config in the future
	pasetoManager, err := auth.NewPasetoManager(auth.PasetoConfig{
		SymmetricKey: cfg.PasetoSymmetricKey,
		Issuer:       "dear-ai-gateway",
		Audience:     "dear-ai-python-backend",
		MinTokenTTL:  10 * time.Second,
		MaxTokenTTL:  2 * time.Minute,
	})
	if err != nil {
		log.Fatalf("Failed to initialize PASETO manager: %v", err)
	}

	return oidcVerifier, pasetoManager
}
