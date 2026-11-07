package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"
)

type config struct {
	port    string
	apiKey  string
	base    string
	maxSess int
	prewarm int
	reuse   bool
	convTTL int
}

func loadConfig() config {
	cfg := config{
		port:    getenv("PORT", "8080"),
		apiKey:  os.Getenv("API_KEY"),
		base:    strings.TrimRight(getenv("UPSTREAM", "https://playground.liquid.ai"), "/"),
		maxSess: getenvInt("POOL_MAX", 8),
		prewarm: getenvInt("PREWARM", 2),
		reuse:   getenv("REUSE", "1") != "0",
		convTTL: getenvInt("CONV_TTL_MIN", 60),
	}
	if cfg.maxSess < 1 {
		cfg.maxSess = 1
	}
	if cfg.prewarm < 0 {
		cfg.prewarm = 0
	}
	if cfg.prewarm > cfg.maxSess {
		cfg.prewarm = cfg.maxSess
	}
	return cfg
}

func getenv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func getenvInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

// requireAuth enforces the optional proxy API key (Bearer token).
func requireAuth(next http.HandlerFunc, apiKey string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if apiKey != "" {
			token := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
			if token == "" {
				token = r.Header.Get("x-api-key")
			}
			if token != apiKey {
				writeOpenAIError(w, http.StatusUnauthorized, "invalid_api_key", "invalid or missing API key")
				return
			}
		}
		next(w, r)
	}
}

func main() {
	cfg := loadConfig()
	pool := NewSessionPool(cfg.base, cfg.maxSess)
	ps := &proxyServer{
		pool:  pool,
		base:  cfg.base,
		reuse: cfg.reuse,
		convs: newConvCache(256, time.Duration(cfg.convTTL)*time.Minute),
	}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", requireAuth(ps.handleChatCompletions, cfg.apiKey))
	mux.HandleFunc("GET /v1/models", requireAuth(ps.handleModels, cfg.apiKey))
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		pool.mu.Lock()
		free, total := len(pool.free), pool.total
		pool.mu.Unlock()
		writeJSON(w, http.StatusOK, map[string]any{
			"status": "ok", "sessions_free": free, "sessions_total": total,
		})
	})

	if cfg.prewarm > 0 {
		go pool.Prewarm(cfg.prewarm)
	}

	srv := &http.Server{
		Addr:              ":" + cfg.port,
		Handler:           mux,
		ReadHeaderTimeout: 15 * time.Second,
	}

	done := make(chan os.Signal, 1)
	signal.Notify(done, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-done
		log.Printf("shutting down")
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_ = srv.Shutdown(ctx)
	}()

	log.Printf("liquid-proxy listening on :%s (upstream %s, pool max %d, auth %s, chat reuse %s)",
		cfg.port, cfg.base, cfg.maxSess,
		map[bool]string{true: "enabled", false: "open"}[cfg.apiKey != ""],
		map[bool]string{true: "on", false: "off"}[cfg.reuse])
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("server: %v", err)
	}
}
