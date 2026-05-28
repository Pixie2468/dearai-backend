package proxy

import (
	"log"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"time"
)

// Proxy aliases httputil.ReverseProxy for cleaner consumer usage.
type Proxy = httputil.ReverseProxy

// NewProxy initializes a robust reverse proxy for the given target.
func NewProxy(target string) (*Proxy, error) {
	targetURL, err := url.Parse(target)
	if err != nil {
		return nil, err
	}

	proxy := httputil.NewSingleHostReverseProxy(targetURL)

	// 1. Inject a robust, production-ready Transport.
	// This prevents infinite hangs if the backend fails.
	proxy.Transport = &http.Transport{
		Proxy: http.ProxyFromEnvironment,
		DialContext: (&net.Dialer{
			Timeout:   10 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext,
		ForceAttemptHTTP2:     true,
		MaxIdleConns:          100,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
		ResponseHeaderTimeout: 10 * time.Second, // Fails fast if backend hangs processing
	}

	// 2. Properly implement the Rewrite hook (Go 1.20+)
	proxy.Rewrite = func(pr *httputil.ProxyRequest) {
		// SetURL routes the request scheme/host/path correctly to the backend
		pr.SetURL(targetURL)

		// SetXForwarded automatically sets X-Forwarded-For, X-Forwarded-Host, and X-Forwarded-Proto
		pr.SetXForwarded()

		// Mutate the OUTGOING request (pr.Out) so the backend sees the expected Host
		pr.Out.Host = targetURL.Host
	}

	// 3. Graceful Error Handling
	proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
		// In production, consider rate-limiting this log or using structured logging (slog)
		log.Printf("[Proxy Error] failed to reach backend %s: %v", targetURL.Host, err)
		http.Error(w, "backend service unavailable", http.StatusBadGateway)
	}

	return proxy, nil
}
