package main

// Conversation reuse for harness clients (ZCode etc.): OpenAI clients resend
// the whole history every turn, but the upstream keeps chat context on its
// side (parentMessageId chaining). We hash the conversation prefix; when a
// request extends a known conversation we send only the last message as a
// delta instead of re-sending everything.

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"sync"
	"time"
)

// ---------------------------------------------------------------------------
// Canonical message hashing

type canonToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type canonMessage struct {
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content,omitempty"`
	ToolCalls  []canonToolCall `json:"tool_calls,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
}

// normalizeContentJSON re-marshals content through typed structs so that the
// client's echo of our reply hashes identically to what we produced.
func normalizeContentJSON(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return []byte("null")
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		b, _ := json.Marshal(s)
		return b
	}
	var parts []struct {
		Type     string `json:"type"`
		Text     string `json:"text"`
		ImageURL *struct {
			URL string `json:"url"`
		} `json:"image_url"`
	}
	if err := json.Unmarshal(raw, &parts); err == nil {
		b, _ := json.Marshal(parts)
		return b
	}
	return raw
}

func normalizeToolCalls(raw json.RawMessage) []canonToolCall {
	if len(raw) == 0 {
		return nil
	}
	var calls []canonToolCall
	if err := json.Unmarshal(raw, &calls); err != nil {
		return nil
	}
	return calls
}

// canonMessage returns the canonical JSON form of one message. Both incoming
// history messages and our own replies are canonicalized through this, so a
// client echoing our answer back produces the same hash input.
func makeCanonMessage(role string, content json.RawMessage, toolCalls json.RawMessage, toolCallID string) string {
	cm := canonMessage{
		Role:       role,
		Content:    normalizeContentJSON(content),
		ToolCalls:  normalizeToolCalls(toolCalls),
		ToolCallID: toolCallID,
	}
	b, _ := json.Marshal(cm)
	return string(b)
}

// hashChain folds one canonical message into the running conversation hash.
func hashChain(prev string, canon string) string {
	h := sha256.Sum256([]byte(prev + "\x00" + canon))
	return hex.EncodeToString(h[:])
}

// seedChain starts a conversation hash chain bound to the session's user, so
// identical texts from different conversations never collide.
func seedChain(userID string) string {
	h := sha256.Sum256([]byte("liquid-conv-seed\x00" + userID))
	return hex.EncodeToString(h[:])
}

// ---------------------------------------------------------------------------
// Conversation cache (LRU + TTL)

type convEntry struct {
	mu sync.Mutex

	userID        string
	chatID        string
	nextParentID  string // message id the next delta should branch from
	lastMessageID string // fallback when an exchange produced no assistant id
	lastUsed      time.Time
}

type convCache struct {
	mu      sync.Mutex
	entries map[string]*convEntry // key: chain hash of a known conversation state
	max     int
	ttl     time.Duration
}

func newConvCache(max int, ttl time.Duration) *convCache {
	return &convCache{entries: map[string]*convEntry{}, max: max, ttl: ttl}
}

// get returns a live (non-expired) entry for the chain hash.
func (c *convCache) get(key string) *convEntry {
	c.mu.Lock()
	defer c.mu.Unlock()
	e, ok := c.entries[key]
	if !ok {
		return nil
	}
	if time.Since(e.lastUsed) > c.ttl {
		delete(c.entries, key)
		return nil
	}
	e.lastUsed = time.Now()
	return e
}

// put registers a conversation state under the chain hash.
func (c *convCache) put(key string, e *convEntry) {
	e.lastUsed = time.Now()
	c.mu.Lock()
	defer c.mu.Unlock()
	c.evictLocked()
	c.entries[key] = e
}

func (c *convCache) evictLocked() {
	now := time.Now()
	for k, e := range c.entries {
		if now.Sub(e.lastUsed) > c.ttl {
			delete(c.entries, k)
		}
	}
	for len(c.entries) >= c.max {
		// evict least recently used
		var oldestKey string
		var oldest time.Time
		first := true
		for k, e := range c.entries {
			if first || e.lastUsed.Before(oldest) {
				oldestKey, oldest, first = k, e.lastUsed, false
			}
		}
		if first {
			break
		}
		delete(c.entries, oldestKey)
	}
}

// invalidateUser drops every entry bound to a session user (session died).
func (c *convCache) invalidateUser(userID string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	for k, e := range c.entries {
		if e.userID == userID {
			delete(c.entries, k)
		}
	}
}

// foldChain folds a list of canonical messages into a running hash.
func foldChain(seed string, canonList []string) string {
	h := seed
	for _, c := range canonList {
		h = hashChain(h, c)
	}
	return h
}

// replyCanonical is the canonical form of OUR reply message — the same shape
// the client echoes back as an assistant message.
func replyCanonical(content *string, calls []ToolCall) string {
	var ccm []canonToolCall
	for _, c := range calls {
		var tc canonToolCall
		tc.ID = c.ID
		tc.Type = "function"
		tc.Function.Name = c.Name
		tc.Function.Arguments = c.Arguments
		ccm = append(ccm, tc)
	}
	var raw json.RawMessage = []byte("null")
	if content != nil {
		raw, _ = json.Marshal(*content)
	}
	cm := canonMessage{Role: "assistant", Content: raw, ToolCalls: ccm}
	b, _ := json.Marshal(cm)
	return string(b)
}
