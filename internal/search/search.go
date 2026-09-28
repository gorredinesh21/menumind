// Package search implements MenuMind's hybrid retrieval: hand-written BM25
// over tokenized dish text, cosine similarity over embedded vectors, fused
// with Reciprocal Rank Fusion, then filtered. Pure Go, no dependencies.
package search

import (
	"math"
	"sort"
	"strings"
	"sync"
	"unicode"

	"github.com/gorredinesh21/menumind/internal/corpus"
)

// Filters narrows the candidate pool before ranking.
type Filters struct {
	VegOnly   bool
	MaxPrice  int // rupees, 0 = any
	MinRating float64
	Cuisine   string // substring match
}

// Hit is one ranked dish with its score breakdown (shown in the UI).
type Hit struct {
	Item     corpus.MenuItem `json:"item"`
	RRF      float64         `json:"rrf"`
	BM25Rank int             `json:"bm25_rank"` // 1-based, 0 = not in top-K
	VecRank  int             `json:"vec_rank"`
	BM25     float64         `json:"bm25_score"`
	Cosine   float64         `json:"cosine"`
}

// Index is the immutable search index over the corpus.
type Index struct {
	items []corpus.MenuItem
	docs  [][]string       // tokenized docs, aligned with items
	tf    []map[string]int // term frequency per doc
	df    map[string]int   // document frequency
	avgDL float64
	n     int

	mu    sync.RWMutex
	vecs  map[string][]float32 // item id → normalized vector
}

// New builds the lexical index; vectors are attached later via SetVectors.
func New(items []corpus.MenuItem) *Index {
	idx := &Index{
		items: append([]corpus.MenuItem(nil), items...),
		docs:  make([][]string, len(items)),
		tf:    make([]map[string]int, len(items)),
		df:    map[string]int{},
		vecs:  map[string][]float32{},
		n:     len(items),
	}
	total := 0
	for i, it := range items {
		toks := Tokenize(it.Text())
		idx.docs[i] = toks
		m := map[string]int{}
		for _, t := range toks {
			m[t]++
		}
		idx.tf[i] = m
		for t := range m {
			idx.df[t]++
		}
		total += len(toks)
	}
	if idx.n > 0 {
		idx.avgDL = float64(total) / float64(idx.n)
	}
	return idx
}

// Tokenize lowercases and keeps letter/digit runs (ASCII + unicode letters,
// so transliterated dish names survive; ₹ and dashes split).
func Tokenize(s string) []string {
	var out []string
	var cur []rune
	for _, r := range strings.ToLower(s) {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			cur = append(cur, r)
		} else if len(cur) > 0 {
			out = append(out, string(cur))
			cur = nil
		}
	}
	if len(cur) > 0 {
		out = append(out, string(cur))
	}
	return out
}

// BM25 scores docs for the query. k1/b are the classic Robertson defaults.
func (idx *Index) BM25(query string) []float64 {
	const k1, b = 1.5, 0.75
	terms := Tokenize(query)
	scores := make([]float64, idx.n)
	if len(terms) == 0 {
		return scores
	}
	idf := make(map[string]float64, len(terms))
	for _, t := range terms {
		n := idx.df[t]
		if n == 0 {
			idf[t] = math.Log(1 + (float64(idx.n)-float64(n)+0.5)/(float64(n)+0.5))
			continue
		}
		idf[t] = math.Log(1 + (float64(idx.n)-float64(n)+0.5)/(float64(n)+0.5))
	}
	for i := 0; i < idx.n; i++ {
		dl := float64(len(idx.docs[i]))
		var s float64
		for _, t := range terms {
			f := float64(idx.tf[i][t])
			if f == 0 {
				continue
			}
			norm := f * (k1 + 1) / (f + k1*(1-b+b*dl/idx.avgDL))
			s += idf[t] * norm
		}
		scores[i] = s
	}
	return scores
}

// SetVectors attaches normalized embedding vectors (item id → vector).
func (idx *Index) SetVectors(v map[string][]float32) {
	norm := make(map[string][]float32, len(v))
	for id, vec := range v {
		norm[id] = Normalize(vec)
	}
	idx.mu.Lock()
	idx.vecs = norm
	idx.mu.Unlock()
}

// VectorCount reports how many items have vectors attached.
func (idx *Index) VectorCount() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.vecs)
}

// Normalize scales a vector to unit length (zero vectors pass through).
func Normalize(v []float32) []float32 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum == 0 {
		return v
	}
	inv := float32(1 / math.Sqrt(sum))
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = x * inv
	}
	return out
}

// Cosine computes the dot product of two unit vectors.
func Cosine(a, b []float32) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var s float64
	for i := 0; i < n; i++ {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}

// Query runs hybrid retrieval: BM25 list + vector list fused by RRF.
// queryVec may be nil (lexical-only mode, e.g. embeddings unavailable).
// Returns top-k hits with per-channel ranks for transparency.
func (idx *Index) Query(query string, queryVec []float32, f Filters, topK int) []Hit {
	type ranked struct {
		pos  int
		rank int
	}
	const rrfK = 60.0
	const channelTop = 50

	// --- lexical channel ---
	bm := idx.BM25(query)
	order := make([]int, idx.n)
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool {
		if bm[order[a]] != bm[order[b]] {
			return bm[order[a]] > bm[order[b]]
		}
		return order[a] < order[b] // stable tiebreak
	})
	bmRanked := map[int]int{} // item pos → rank (1-based)
	for r, pos := range order {
		if r >= channelTop || bm[pos] <= 0 {
			break
		}
		bmRanked[pos] = r + 1
	}

	// --- vector channel ---
	vecRanked := map[int]int{}
	if queryVec != nil {
		q := Normalize(queryVec)
		idx.mu.RLock()
		type scored struct {
			pos int
			s   float64
		}
		var ss []scored
		for pos, it := range idx.items {
			v, ok := idx.vecs[it.ID]
			if !ok {
				continue
			}
			if s := Cosine(q, v); s > 0.15 { // noise floor
				ss = append(ss, scored{pos, s})
			}
		}
		idx.mu.RUnlock()
		sort.Slice(ss, func(a, b int) bool {
			if ss[a].s != ss[b].s {
				return ss[a].s > ss[b].s
			}
			return ss[a].pos < ss[b].pos
		})
		for r, sc := range ss {
			if r >= channelTop {
				break
			}
			vecRanked[sc.pos] = r + 1
		}
	}

	// --- fuse + filter ---
	fuse := map[int]float64{}
	for pos, r := range bmRanked {
		fuse[pos] += 1 / (rrfK + float64(r))
	}
	for pos, r := range vecRanked {
		fuse[pos] += 1 / (rrfK + float64(r))
	}
	var hits []Hit
	for pos, s := range fuse {
		it := idx.items[pos]
		if !passes(it, f) {
			continue
		}
		hits = append(hits, Hit{
			Item: it, RRF: s,
			BM25Rank: bmRanked[pos], VecRank: vecRanked[pos],
			BM25: bm[pos],
		})
	}
	sort.Slice(hits, func(a, b int) bool {
		if hits[a].RRF != hits[b].RRF {
			return hits[a].RRF > hits[b].RRF
		}
		return hits[a].Item.Rating > hits[b].Item.Rating
	})
	if topK > 0 && len(hits) > topK {
		hits = hits[:topK]
	}
	// fill cosine scores for display
	if queryVec != nil {
		q := Normalize(queryVec)
		idx.mu.RLock()
		for i := range hits {
			if v, ok := idx.vecs[hits[i].Item.ID]; ok {
				hits[i].Cosine = math.Round(Cosine(q, v)*1000) / 1000
			}
		}
		idx.mu.RUnlock()
	}
	return hits
}

func passes(it corpus.MenuItem, f Filters) bool {
	if f.VegOnly && !it.Veg {
		return false
	}
	if f.MaxPrice > 0 && it.Price > f.MaxPrice {
		return false
	}
	if f.MinRating > 0 && it.Rating < f.MinRating {
		return false
	}
	if f.Cuisine != "" && !strings.Contains(strings.ToLower(it.Cuisine), strings.ToLower(f.Cuisine)) {
		return false
	}
	return true
}

var _ = corpus.Get
