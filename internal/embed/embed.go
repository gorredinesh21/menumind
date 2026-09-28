// Package embed talks to the bge-small-en-v1.5 feature-extraction endpoint,
// with a disk cache (committed to the repo) so boots are instant, an
// in-memory query LRU, and graceful degradation to lexical-only search.
package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"sync"
	"time"
)

// Client embeds text.
type Client struct {
	BaseURL string // https://router.huggingface.co
	APIKey  string
	Model   string // BAAI/bge-small-en-v1.5
	HTTP    *http.Client

	mu    sync.Mutex
	cache map[string][]float32 // query cache (bounded)
	order []string
	max   int
}

// New builds a client with a bounded in-memory query cache.
func New(base, key, model string) *Client {
	return &Client{
		BaseURL: base, APIKey: key, Model: model,
		HTTP:  &http.Client{Timeout: 30 * time.Second},
		cache: map[string][]float32{}, max: 512,
	}
}

// Enabled reports whether an API key is configured.
func (c *Client) Enabled() bool { return c != nil && c.APIKey != "" }

// One embeds a single text (no cache — used for corpus warmup).
func (c *Client) One(ctx context.Context, text string) ([]float32, error) {
	url := fmt.Sprintf("%s/hf-inference/models/%s/pipeline/feature-extraction",
		trimSlash(c.BaseURL), c.Model)
	body, _ := json.Marshal(map[string]any{"inputs": text, "options": map[string]bool{"wait_for_model": true}})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+c.APIKey)
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return nil, fmt.Errorf("embed request: %w", err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("embed status %d: %.150s", resp.StatusCode, raw)
	}
	return decodeVec(raw)
}

// decodeVec accepts 1D [384] or 2D token-level [n][384] responses,
// mean-pooling the 2D case into a single vector.
func decodeVec(raw []byte) ([]float32, error) {
	var flat []float32
	if err := json.Unmarshal(raw, &flat); err == nil {
		if len(flat) == 0 {
			return nil, fmt.Errorf("empty embedding")
		}
		return flat, nil
	}
	var nested [][]float32
	if err := json.Unmarshal(raw, &nested); err != nil {
		return nil, fmt.Errorf("unexpected embedding shape")
	}
	if len(nested) == 0 {
		return nil, fmt.Errorf("empty embedding")
	}
	out := make([]float32, len(nested[0]))
	for _, tok := range nested {
		for i, x := range tok {
			if i < len(out) {
				out[i] += x
			}
		}
	}
	for i := range out {
		out[i] /= float32(len(nested))
	}
	return out, nil
}

// Query embeds a user query through the bounded LRU cache.
func (c *Client) Query(ctx context.Context, text string) ([]float32, error) {
	if !c.Enabled() {
		return nil, fmt.Errorf("embeddings disabled (no key)")
	}
	c.mu.Lock()
	if v, ok := c.cache[text]; ok {
		c.mu.Unlock()
		return v, nil
	}
	c.mu.Unlock()
	v, err := c.One(ctx, text)
	if err != nil {
		return nil, err
	}
	c.mu.Lock()
	if _, ok := c.cache[text]; !ok {
		c.cache[text] = v
		c.order = append(c.order, text)
		if len(c.order) > c.max {
			oldest := c.order[0]
			c.order = c.order[1:]
			delete(c.cache, oldest)
		}
	}
	c.mu.Unlock()
	return v, nil
}

// CacheFile is the committed corpus-embedding cache (item id → vector).
type CacheFile map[string][]float32

// LoadCache reads embeddings.json from disk.
func LoadCache(path string) (CacheFile, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return DecodeCache(b)
}

// DecodeCache parses cache bytes.
func DecodeCache(b []byte) (CacheFile, error) {
	var cf CacheFile
	if err := json.Unmarshal(b, &cf); err != nil {
		return nil, err
	}
	return cf, nil
}

// SaveCache writes embeddings.json atomically.
func SaveCache(path string, cf CacheFile) error {
	b, err := json.Marshal(cf)
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func trimSlash(s string) string {
	for len(s) > 0 && s[len(s)-1] == '/' {
		s = s[:len(s)-1]
	}
	return s
}
