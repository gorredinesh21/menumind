// Package ask does grounded generation: retrieve dishes, then stream an LLM
// answer that may ONLY use the retrieved context, with [D1]-style citations.
package ask

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/gorredinesh21/menumind/internal/search"
)

// Client streams grounded answers.
type Client struct {
	BaseURL string
	APIKey  string
	Model   string
	HTTP    *http.Client
}

// New builds the ask client.
func New(base, key, model string) *Client {
	return &Client{BaseURL: base, APIKey: key, Model: model,
		HTTP: &http.Client{Timeout: 90 * time.Second}}
}

// Enabled reports whether generation is configured.
func (c *Client) Enabled() bool { return c != nil && c.APIKey != "" }

// Context builds the numbered dish context block from retrieval hits.
func Context(hits []search.Hit) string {
	var b strings.Builder
	for i, h := range hits {
		fmt.Fprintf(&b, "[D%d] %s — %s. %s cuisine, %s at %s (%s). ₹%d, ⭐%.1f, %s.\n",
			i+1, h.Item.Name, h.Item.Description,
			h.Item.Cuisine, h.Item.Category, h.Item.RestName, h.Item.Area,
			h.Item.Price, h.Item.Rating, vegTag(h.Item.Veg))
	}
	return b.String()
}

func vegTag(v bool) string {
	if v {
		return "veg"
	}
	return "non-veg"
}

// Stream calls the model with the grounded prompt and emits answer tokens via
// the callback. tokens are raw text deltas; the callback returns false to stop.
func (c *Client) Stream(ctx context.Context, question, dishCtx string, emit func(delta string) bool) error {
	system := `You are MenuMind, a food discovery assistant for Bangalore menus.
Answer the user's question using ONLY the numbered dishes provided in the context ([D1], [D2], …).
Cite every dish you mention with its [D#] tag. If the context cannot answer the question, say exactly what's missing and suggest the closest dishes instead.
Be concise: 2-4 short sentences or a tight list. Include prices (₹) where relevant. Never invent dishes, restaurants or prices.`
	user := "CONTEXT (retrieved dishes):\n" + dishCtx + "\nQUESTION: " + question

	body, _ := json.Marshal(map[string]any{
		"model": c.Model,
		"messages": []map[string]string{
			{"role": "system", "content": system},
			{"role": "user", "content": user},
		},
		"max_tokens":  400,
		"temperature": 0.2,
		"stream":      true,
	})
	url := strings.TrimRight(c.BaseURL, "/") + "/v1/chat/completions"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+c.APIKey)
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return fmt.Errorf("llm request: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return fmt.Errorf("llm status %d: %.200s", resp.StatusCode, raw)
	}

	// parse OpenAI-style SSE: lines "data: {...}" and terminal "data: [DONE]"
	r := bufio.NewReader(resp.Body)
	for {
		line, err := r.ReadString('\n')
		if err != nil {
			break
		}
		line = strings.TrimRight(line, "\r\n")
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			break
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if json.Unmarshal([]byte(payload), &chunk) != nil || len(chunk.Choices) == 0 {
			continue
		}
		if d := chunk.Choices[0].Delta.Content; d != "" {
			if !emit(d) {
				return nil // consumer stopped early
			}
		}
	}
	return nil
}
