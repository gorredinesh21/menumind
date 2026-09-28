package embed

import (
	"encoding/json"
	"testing"
)

func TestDecodeVec1D(t *testing.T) {
	raw, _ := json.Marshal([]float32{0.1, 0.2, 0.3})
	v, err := decodeVec(raw)
	if err != nil || len(v) != 3 || v[0] != 0.1 {
		t.Fatalf("1D decode failed: %v %v", v, err)
	}
}

func TestDecodeVec2DMeanPools(t *testing.T) {
	raw, _ := json.Marshal([][]float32{{1, 2, 3}, {3, 4, 5}})
	v, err := decodeVec(raw)
	if err != nil {
		t.Fatal(err)
	}
	if len(v) != 3 || v[0] != 2 || v[1] != 3 || v[2] != 4 {
		t.Fatalf("mean pooling wrong: %v", v)
	}
}

func TestDecodeVecGarbage(t *testing.T) {
	if _, err := decodeVec([]byte(`{"error":"x"}`)); err == nil {
		t.Fatal("object shape must error")
	}
	if _, err := decodeVec([]byte(`[]`)); err == nil {
		t.Fatal("empty nested must error")
	}
}

func TestQueryCacheRoundTrip(t *testing.T) {
	c := New("http://localhost:1", "k", "m")
	c.cache["hello"] = []float32{1}
	v, err := c.Query(nil, "hello")
	if err != nil || v[0] != 1 {
		t.Fatalf("cache hit failed: %v %v", v, err)
	}
}

func TestDisabledClient(t *testing.T) {
	c := New("http://x", "", "m")
	if c.Enabled() {
		t.Fatal("empty key must disable")
	}
	if _, err := c.Query(nil, "q"); err == nil {
		t.Fatal("disabled client must error")
	}
}

func TestCacheFileRoundTrip(t *testing.T) {
	cf := CacheFile{"a": {1, 2}, "b": {3}}
	b, err := json.Marshal(cf)
	if err != nil {
		t.Fatal(err)
	}
	got, err := DecodeCache(b)
	if err != nil || len(got) != 2 || got["a"][1] != 2 {
		t.Fatalf("cache round-trip failed: %v %v", got, err)
	}
}
