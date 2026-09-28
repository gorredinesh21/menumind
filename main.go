// MenuMind — semantic menu search + grounded RAG answers over a Bangalore
// menu corpus, in pure Go: BM25 + vector cosine fused with RRF, streaming
// cited LLM answers, graceful degradation at every layer.
package main

import (
	"context"
	embedio "embed"
	"flag"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"time"

	"github.com/gorredinesh21/menumind/internal/ask"
	"github.com/gorredinesh21/menumind/internal/corpus"
	"github.com/gorredinesh21/menumind/internal/embed"
	"github.com/gorredinesh21/menumind/internal/search"
	"github.com/gorredinesh21/menumind/internal/server"
)

//go:embed data/embeddings.json
var cacheFS embedio.FS

func main() {
	addr := flag.String("addr", envOr("PORT", "8080"), "listen address")
	hfKey := flag.String("hf-token", os.Getenv("HF_TOKEN"), "HuggingFace token (empty = lexical-only mode)")
	embedModel := flag.String("embed-model", envOr("HF_EMBED_MODEL", "BAAI/bge-small-en-v1.5"), "embedding model")
	llmModel := flag.String("llm-model", envOr("HF_LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct"), "chat model")
	base := flag.String("ai-base", envOr("AI_BASE_URL", "https://router.huggingface.co"), "OpenAI-compatible base URL")
	flag.Parse()

	log := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	slog.SetDefault(log)

	c := corpus.Get()
	idx := search.New(c.Items)
	log.Info("index built", "dishes", len(c.Items), "restaurants", len(c.Restaurants))

	// load the committed embedding cache (falls back to embedded copy),
	// attach vectors; semantic search turns on only if vectors + key exist.
	loadCache := func() embed.CacheFile {
		if b, err := os.ReadFile("data/embeddings.json"); err == nil {
			cf, err := embed.DecodeCache(b)
			if err == nil {
				return cf
			}
			log.Warn("disk cache unreadable, trying embedded", "err", err)
		}
		b, err := cacheFS.ReadFile("data/embeddings.json")
		if err != nil {
			return nil
		}
		cf, err := embed.DecodeCache(b)
		if err != nil {
			log.Warn("embedded cache unreadable", "err", err)
			return nil
		}
		return cf
	}
	if cf := loadCache(); len(cf) > 0 {
		idx.SetVectors(cf)
		log.Info("vectors attached", "count", len(cf))
	} else {
		log.Warn("no embedding cache — semantic channel off, BM25-only")
	}

	emb := embed.New(*base, *hfKey, *embedModel)
	asker := ask.New(*base, *hfKey, *llmModel)

	srv := server.New(idx, emb, asker, log)
	httpSrv := &http.Server{
		Addr:              ":" + trimColon(*addr),
		Handler:           srv.Handler(),
		ReadHeaderTimeout: 5 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	go func() {
		log.Info("menumind listening", "addr", httpSrv.Addr,
			"semantic", emb.Enabled() && idx.VectorCount() > 0,
			"generation", asker.Enabled(), "dishes", len(c.Items))
		if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Error("server error", "err", err)
			stop()
		}
	}()
	<-ctx.Done()
	log.Info("shutting down")
	shCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = httpSrv.Shutdown(shCtx)
	log.Info("bye")
}

func envOr(k, d string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return d
}

func trimColon(s string) string {
	if len(s) > 0 && s[0] == ':' {
		return s[1:]
	}
	return s
}
