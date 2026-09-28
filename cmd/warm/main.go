// Command warm builds the corpus embedding cache (data/embeddings.json) with
// a bounded concurrent worker pool. Run once; the file is committed so deploys
// boot instantly with semantic search enabled.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"golang.org/x/sync/errgroup"

	"github.com/gorredinesh21/menumind/internal/corpus"
	"github.com/gorredinesh21/menumind/internal/embed"
)

func main() {
	out := flag.String("out", "data/embeddings.json", "cache output path")
	key := flag.String("hf-token", os.Getenv("HF_TOKEN"), "HuggingFace token")
	model := flag.String("model", "BAAI/bge-small-en-v1.5", "embedding model")
	workers := flag.Int("workers", 8, "concurrent embedding calls")
	flag.Parse()

	if *key == "" {
		log.Fatal("HF_TOKEN required to warm the cache")
	}
	c := corpus.Get()
	cl := embed.New("https://router.huggingface.co", *key, *model)

	start := time.Now()
	items := c.Items
	type result struct {
		id string
		v  []float32
	}
	results := make([]result, len(items))
	g, ctx := errgroup.WithContext(context.Background())
	g.SetLimit(*workers)
	for i, it := range items {
		i, it := i, it
		g.Go(func() error {
			vec, err := cl.One(ctx, it.Text())
			if err != nil {
				return fmt.Errorf("%s (%s): %w", it.ID, it.Name, err)
			}
			results[i] = result{it.ID, vec}
			if (i+1)%100 == 0 {
				log.Printf("embedded %d/%d", i+1, len(items))
			}
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		log.Fatalf("warm failed: %v", err)
	}
	cf := embed.CacheFile{}
	for _, r := range results {
		cf[r.id] = r.v
	}
	if err := os.MkdirAll("data", 0o755); err != nil {
		log.Fatal(err)
	}
	if err := embed.SaveCache(*out, cf); err != nil {
		log.Fatal(err)
	}
	log.Printf("warmed %d vectors in %s → %s", len(cf), time.Since(start).Round(time.Millisecond), *out)
}
