// Package server exposes MenuMind over HTTP: instant hybrid search,
// grounded Ask streaming, stats and health.
package server

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/gorredinesh21/menumind/internal/ask"
	"github.com/gorredinesh21/menumind/internal/corpus"
	"github.com/gorredinesh21/menumind/internal/embed"
	"github.com/gorredinesh21/menumind/internal/search"
	"github.com/gorredinesh21/menumind/web"
)

// Server wires the index, embedder and asker to HTTP.
type Server struct {
	Index *search.Index
	Emb   *embed.Client
	Ask   *ask.Client
	Log   *slog.Logger
}

// New builds the server.
func New(idx *search.Index, e *embed.Client, a *ask.Client, log *slog.Logger) *Server {
	return &Server{Index: idx, Emb: e, Ask: a, Log: log}
}

// Handler returns the route mux.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	static, _ := fs.Sub(web.Assets, "files")
	mux.Handle("/", http.FileServer(http.FS(static)))
	mux.HandleFunc("GET /healthz", s.health) // internal/startup probes
	mux.HandleFunc("GET /live", s.health)    // public: GFE intercepts /healthz on ingress
	mux.HandleFunc("GET /stats", s.stats)
	mux.HandleFunc("GET /api/search", s.searchH)
	mux.HandleFunc("POST /api/ask", s.askH)
	return s.logMid(mux)
}

func (s *Server) logMid(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		if r.URL.Path != "/healthz" {
			s.Log.Info("http", "method", r.Method, "path", r.URL.Path, "ms", time.Since(start).Milliseconds())
		}
	})
}

func (s *Server) health(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"status": "ok",
		"dishes": len(corpus.Get().Items),
		"vectors": s.Index.VectorCount(),
		"semantic": s.Emb.Enabled(),
		"generation": s.Ask.Enabled(),
	})
}

func (s *Server) stats(w http.ResponseWriter, r *http.Request) {
	c := corpus.Get()
	areas := map[string]int{}
	cuisines := map[string]int{}
	for _, it := range c.Items {
		areas[it.Area]++
		cuisines[it.Cuisine]++
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"dishes": len(c.Items), "restaurants": len(c.Restaurants),
		"areas": areas, "cuisines": cuisines,
		"vectors": s.Index.VectorCount(),
	})
}

func filtersFromQuery(r *http.Request) search.Filters {
	q := r.URL.Query()
	f := search.Filters{
		VegOnly:   q.Get("veg") == "1" || q.Get("veg") == "true",
		MinRating: parseFloat(q.Get("min_rating")),
		Cuisine:   q.Get("cuisine"),
	}
	f.MaxPrice = int(parseFloat(q.Get("max_price")))
	return f
}

func parseFloat(s string) float64 {
	v, _ := strconv.ParseFloat(strings.TrimSpace(s), 64)
	return v
}

func (s *Server) searchH(w http.ResponseWriter, r *http.Request) {
	q := strings.TrimSpace(r.URL.Query().Get("q"))
	if q == "" {
		http.Error(w, "q required", http.StatusBadRequest)
		return
	}
	k := int(parseFloat(r.URL.Query().Get("k")))
	if k <= 0 || k > 40 {
		k = 12
	}
	f := filtersFromQuery(r)

	var vec []float32
	semantic := false
	if s.Emb.Enabled() {
		ctx, cancel := context.WithTimeout(r.Context(), 8*time.Second)
		v, err := s.Emb.Query(ctx, q)
		cancel()
		if err != nil {
			s.Log.Warn("query embed failed, lexical-only", "err", err)
		} else {
			vec, semantic = v, true
		}
	}
	hits := s.Index.Query(q, vec, f, k)
	writeJSON(w, http.StatusOK, map[string]any{
		"query": q, "semantic": semantic, "count": len(hits), "results": hits,
	})
}

type askRequest struct {
	Question string `json:"question"`
	VegOnly  bool   `json:"veg_only"`
	MaxPrice int    `json:"max_price"`
	MinRating float64 `json:"min_rating"`
	Cuisine  string `json:"cuisine"`
}

// askH retrieves dishes for the question and streams a grounded answer.
func (s *Server) askH(w http.ResponseWriter, r *http.Request) {
	var req askRequest
	if err := json.NewDecoder(io.LimitReader(r.Body, 8192)).Decode(&req); err != nil {
		http.Error(w, "bad json", http.StatusBadRequest)
		return
	}
	q := strings.TrimSpace(req.Question)
	if q == "" || len(q) > 500 {
		http.Error(w, "question required (≤500 chars)", http.StatusBadRequest)
		return
	}
	f := search.Filters{VegOnly: req.VegOnly, MaxPrice: req.MaxPrice, MinRating: req.MinRating, Cuisine: req.Cuisine}

	var vec []float32
	if s.Emb.Enabled() {
		ctx, cancel := context.WithTimeout(r.Context(), 8*time.Second)
		v, err := s.Emb.Query(ctx, q)
		cancel()
		if err == nil {
			vec = v
		}
	}
	hits := s.Index.Query(q, vec, f, 8)

	fl, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")

	// sources first so the UI can render dish cards while the LLM writes
	src, _ := json.Marshal(map[string]any{"type": "sources", "results": hits})
	fmt.Fprintf(w, "data: %s\n\n", src)
	fl.Flush()

	if !s.Ask.Enabled() {
		// deterministic degraded mode: a readable summary without the LLM
		emitAll(w, fl, lexicalSummary(q, hits))
		return
	}

	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()
	err := s.Ask.Stream(ctx, q, ask.Context(hits), func(delta string) bool {
		b, _ := json.Marshal(map[string]any{"type": "delta", "text": delta})
		fmt.Fprintf(w, "data: %s\n\n", b)
		fl.Flush()
		return true
	})
	if err != nil {
		s.Log.Warn("ask stream failed", "err", err)
		em, _ := json.Marshal(map[string]any{"type": "error", "text": "LLM unavailable — showing retrieval only"})
		fmt.Fprintf(w, "data: %s\n\n", em)
		fl.Flush()
	}
	done, _ := json.Marshal(map[string]any{"type": "done"})
	fmt.Fprintf(w, "data: %s\n\n", done)
	fl.Flush()
}

func emitAll(w http.ResponseWriter, fl http.Flusher, text string) {
	for _, chunk := range chunkBy(text, 24) {
		b, _ := json.Marshal(map[string]any{"type": "delta", "text": chunk})
		fmt.Fprintf(w, "data: %s\n\n", b)
		fl.Flush()
	}
	done, _ := json.Marshal(map[string]any{"type": "done"})
	fmt.Fprintf(w, "data: %s\n\n", done)
	fl.Flush()
}

func chunkBy(s string, n int) []string {
	var out []string
	for len(s) > n {
		out = append(out, s[:n])
		s = s[n:]
	}
	if s != "" {
		out = append(out, s)
	}
	return out
}

// lexicalSummary is the no-LLM degraded answer built from retrieval alone.
func lexicalSummary(q string, hits []search.Hit) string {
	if len(hits) == 0 {
		return "No matching dishes found for \"" + q + "\". Try broader words like 'spicy', 'veg', 'biryani' or 'dessert'."
	}
	var b strings.Builder
	b.WriteString("Retrieval-only mode (LLM offline). Top matches:\n")
	for i, h := range hits {
		if i >= 5 {
			break
		}
		fmt.Fprintf(&b, "%d. %s (₹%d, ⭐%.1f) at %s — %s\n",
			i+1, h.Item.Name, h.Item.Price, h.Item.Rating, h.Item.RestName, h.Item.Description)
	}
	return b.String()
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}
