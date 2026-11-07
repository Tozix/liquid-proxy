package main

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"
	"unicode"
)

// ---------------------------------------------------------------------------
// Model catalog (slugs verified against the live playground on 2026-11-07)

type modelDef struct {
	ID   string // exposed to clients
	Slug string // upstream modelSlug
	Name string
}

var catalog = []modelDef{
	{"lfm2.5-24b-a2b", "lfm2.5-24b-a2b-20261027", "LFM2.5-24B-A2B"},
	{"lfm3-1.7b", "lfm3-1.7b-20261027", "LFM3-1.7B"},
	{"lfm3-vl-2b", "lfm3-vl-2b-20261027", "LFM3-VL-2B"},
	{"lfm2.5-2.6b", "lfm2.5-2.6b-20260811", "LFM2.5-2.6B"},
	{"lfm2.5-vl-3b", "lfm2.5-vl-3b-20260811", "LFM2.5-VL-3B"},
	{"lfm2.5-1.2b-jp", "lfm2.5-1.2b-jp-202606-20260604", "LFM2.5-1.2B-JP"},
}

// resolveModel maps a client-supplied model name to an upstream slug.
// Unknown names fall back to the default slug (logged).
func resolveModel(input string) string {
	in := strings.ToLower(strings.TrimSpace(input))
	for _, m := range catalog {
		if in == m.Slug || in == m.ID {
			return m.Slug
		}
	}
	if in == "" || in == "default" {
		return defaultModelSlug
	}
	log.Printf("[model] unknown model %q, falling back to %s", input, defaultModelSlug)
	return defaultModelSlug
}

// ---------------------------------------------------------------------------
// OpenAI wire types

type chatMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

type completionRequest struct {
	Model         string        `json:"model"`
	Messages      []chatMessage `json:"messages"`
	Stream        bool          `json:"stream"`
	StreamOptions *struct {
		IncludeUsage bool `json:"include_usage"`
	} `json:"stream_options"`
	// Tool/function calling and sampling parameters are accepted but ignored.
}

// imagePart is one decoded image awaiting upload to the playground.
type imagePart struct {
	data []byte
	mime string
}

const (
	maxImagesPerMessage = 4
	maxImageBytes       = 10 << 20 // 10 MB
)

// extractContent splits an OpenAI content field into text and images.
// Content may be a string or an array of typed parts.
// Image sources: data: URIs and http(s) URLs.
func extractContent(raw json.RawMessage) (text string, images []imagePart, err error) {
	if len(raw) == 0 {
		return "", nil, nil
	}
	var s string
	if uerr := json.Unmarshal(raw, &s); uerr == nil {
		return s, nil, nil
	}
	var parts []struct {
		Type     string `json:"type"`
		Text     string `json:"text"`
		ImageURL *struct {
			URL    string `json:"url"`
			Detail string `json:"detail"`
		} `json:"image_url"`
	}
	if perr := json.Unmarshal(raw, &parts); perr != nil {
		return "", nil, fmt.Errorf("unsupported content format")
	}
	var b strings.Builder
	for _, p := range parts {
		switch p.Type {
		case "text":
			b.WriteString(p.Text)
		case "image_url":
			if p.ImageURL == nil || p.ImageURL.URL == "" {
				return "", nil, fmt.Errorf("image_url part with empty url")
			}
			img, ierr := decodeImageSource(p.ImageURL.URL)
			if ierr != nil {
				return "", nil, ierr
			}
			images = append(images, img)
		default:
			return "", nil, fmt.Errorf("unsupported content part type %q", p.Type)
		}
	}
	return b.String(), images, nil
}

// decodeImageSource materializes an image from a data: URI or http(s) URL.
func decodeImageSource(src string) (imagePart, error) {
	if strings.HasPrefix(src, "data:") {
		// data:<mime>;base64,<payload>
		rest := strings.TrimPrefix(src, "data:")
		comma := strings.Index(rest, ",")
		if comma < 0 {
			return imagePart{}, fmt.Errorf("bad data URI")
		}
		meta, payload := rest[:comma], rest[comma+1:]
		meta = strings.TrimSuffix(meta, ";base64")
		mime := strings.SplitN(meta, ";", 2)[0]
		if mime == "" {
			mime = "application/octet-stream"
		}
		data, derr := base64.StdEncoding.DecodeString(payload)
		if derr != nil {
			return imagePart{}, fmt.Errorf("bad base64 in data URI: %v", derr)
		}
		if len(data) > maxImageBytes {
			return imagePart{}, fmt.Errorf("image larger than %d MB", maxImageBytes>>20)
		}
		return imagePart{data: data, mime: mime}, nil
	}
	if !strings.HasPrefix(src, "http://") && !strings.HasPrefix(src, "https://") {
		return imagePart{}, fmt.Errorf("image url must be data: URI or http(s) URL")
	}
	req, err := http.NewRequest(http.MethodGet, src, nil)
	if err != nil {
		return imagePart{}, fmt.Errorf("bad image url: %v", err)
	}
	req.Header.Set("User-Agent", userAgent)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return imagePart{}, fmt.Errorf("cannot download image: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return imagePart{}, fmt.Errorf("cannot download image: status %d", resp.StatusCode)
	}
	data, err := io.ReadAll(io.LimitReader(resp.Body, maxImageBytes+1))
	if err != nil {
		return imagePart{}, fmt.Errorf("cannot download image: %v", err)
	}
	if len(data) > maxImageBytes {
		return imagePart{}, fmt.Errorf("image larger than %d MB", maxImageBytes>>20)
	}
	mime := resp.Header.Get("Content-Type")
	if i := strings.Index(mime, ";"); i >= 0 {
		mime = strings.TrimSpace(mime[:i])
	}
	if mime == "" {
		mime = http.DetectContentType(data)
	}
	return imagePart{data: data, mime: mime}, nil
}

func isVisionSlug(slug string) bool {
	return strings.Contains(slug, "-vl-")
}

// serializeMessages folds an OpenAI conversation into one upstream prompt and
// collects all referenced images.
func serializeMessages(msgs []chatMessage) (string, []imagePart, error) {
	var b strings.Builder
	var images []imagePart
	for i, m := range msgs {
		text, imgs, err := extractContent(m.Content)
		if err != nil {
			return "", nil, fmt.Errorf("message %d: %w", i, err)
		}
		images = append(images, imgs...)
		switch m.Role {
		case "system", "developer":
			b.WriteString(text)
		case "assistant":
			b.WriteString("Assistant: " + text)
		case "tool", "function":
			b.WriteString("Tool result: " + text)
		default: // user and anything else
			b.WriteString("User: " + text)
		}
		b.WriteString("\n\n")
	}
	return strings.TrimRight(b.String(), "\n"), images, nil
}

func newID(prefix string) string {
	b := make([]byte, 12)
	_, _ = rand.Read(b)
	return prefix + hex.EncodeToString(b)
}

func estTokens(s string) int {
	n := (len(s) + 3) / 4
	if n < 1 {
		n = 1
	}
	return n
}

type usageInfo struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeOpenAIError(w http.ResponseWriter, status int, typ, msg string) {
	writeJSON(w, status, map[string]any{
		"error": map[string]any{"message": msg, "type": typ, "code": nil},
	})
}

func writeUpstreamError(w http.ResponseWriter, err error) {
	switch e := err.(type) {
	case errSessionDead:
		writeOpenAIError(w, http.StatusBadGateway, "upstream_error", "liquid session rejected: "+e.Error())
	case errUpstream:
		writeOpenAIError(w, http.StatusBadGateway, "upstream_error", e.Error())
	default:
		writeOpenAIError(w, http.StatusBadGateway, "upstream_error", err.Error())
	}
}

// ---------------------------------------------------------------------------
// Handlers

type proxyServer struct {
	pool *SessionPool
	base string
}

// handleChatCompletions implements POST /v1/chat/completions.
//
// Upstream auth failures are detected at StartChatStream time (headers are
// checked before the body is consumed), so retries happen before any byte is
// written to the client.
func (ps *proxyServer) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(io.LimitReader(r.Body, 8<<20))
	if err != nil {
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error", "cannot read body")
		return
	}
	var req completionRequest
	if err := json.Unmarshal(body, &req); err != nil {
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error", "bad JSON: "+err.Error())
		return
	}
	if len(req.Messages) == 0 {
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error", "messages must not be empty")
		return
	}
	prompt, images, err := serializeMessages(req.Messages)
	if err != nil {
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	slug := resolveModel(req.Model)
	promptTokens := 4
	for _, m := range req.Messages {
		if t, _, terr := extractContent(m.Content); terr == nil {
			promptTokens += estTokens(t) + 4
		}
	}
	if len(images) > maxImagesPerMessage {
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error",
			fmt.Sprintf("too many images: %d (max %d per request)", len(images), maxImagesPerMessage))
		return
	}
	if len(images) > 0 && !isVisionSlug(slug) {
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error",
			"model "+slug+" does not accept images; use a vision model: lfm3-vl-2b or lfm2.5-vl-3b")
		return
	}

	const maxAttempts = 2
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		session, err := ps.pool.Acquire(r.Context())
		if err != nil {
			writeOpenAIError(w, http.StatusServiceUnavailable, "server_error", "no upstream session available: "+err.Error())
			return
		}

		// images: upload via this session before starting the stream
		imageUuids := []string{}
		uploadFailed := false
		for i, img := range images {
			key, uerr := session.UploadImage(r.Context(), ps.base, img.data, img.mime)
			if uerr != nil {
				if isDead(uerr) && attempt < maxAttempts {
					ps.pool.Drop(session)
					uploadFailed = true
					break
				}
				writeOpenAIError(w, http.StatusBadGateway, "upstream_error", fmt.Sprintf("image %d upload: %v", i, uerr))
				ps.pool.put(session)
				return
			}
			imageUuids = append(imageUuids, key)
		}
		if uploadFailed {
			continue
		}

		chatID, cerr := session.createChat(r.Context(), ps.base)
		if cerr != nil {
			ps.settle(session, cerr)
			if isDead(cerr) && attempt < maxAttempts {
				continue
			}
			writeUpstreamError(w, cerr)
			return
		}
		streamResp, serr := session.StartChatStream(r.Context(), ps.base, chatID, slug, prompt, imageUuids)
		if serr != nil {
			// the chat exists but is useless — remove it before releasing
			ps.cleanupChat(session, chatID)
			ps.settle(session, serr)
			if isDead(serr) && attempt < maxAttempts {
				continue
			}
			writeUpstreamError(w, serr)
			return
		}
		// upstream accepted the request; the stream itself may still turn out
		// empty (known upstream flake) — retryable before any byte is sent
		if req.Stream {
			if ps.streamToClient(w, r, streamResp, slug, req, promptTokens) {
				if attempt < maxAttempts && r.Context().Err() == nil {
					log.Printf("[retry] empty stream (chat %s), retrying on a fresh chat", chatID)
					ps.cleanupChat(session, chatID)
					ps.pool.put(session)
					continue
				}
				ps.writeEmptyStreamFallback(w, modelNameFor(req, slug))
				ps.cleanupChat(session, chatID)
				ps.pool.put(session)
				return
			}
		} else {
			if ps.blockToClient(w, r, streamResp, slug, req, promptTokens) {
				if attempt < maxAttempts && r.Context().Err() == nil {
					log.Printf("[retry] empty completion (chat %s), retrying on a fresh chat", chatID)
					ps.cleanupChat(session, chatID)
					ps.pool.put(session)
					continue
				}
				ps.writeEmptyCompletion(w, slug, req, promptTokens)
				ps.cleanupChat(session, chatID)
				ps.pool.put(session)
				return
			}
		}
		// the client already has the full response; housekeeping before release
		ps.cleanupChat(session, chatID)
		ps.pool.put(session)
		return
	}
}

func modelNameFor(req completionRequest, slug string) string {
	if req.Model != "" {
		return req.Model
	}
	return slug
}

// writeEmptyStreamFallback emits a minimal valid SSE stream when every
// attempt returned an empty upstream stream (headers were never sent).
func (ps *proxyServer) writeEmptyStreamFallback(w http.ResponseWriter, model string) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(http.StatusOK)
	send := func(delta map[string]any, finish any) {
		chunk := map[string]any{
			"id": newID("chatcmpl-"), "object": "chat.completion.chunk",
			"created": time.Now().Unix(), "model": model,
			"choices": []any{map[string]any{"index": 0, "delta": delta, "finish_reason": finish}},
		}
		data, _ := json.Marshal(chunk)
		_, _ = fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}
	send(map[string]any{"role": "assistant", "content": ""}, nil)
	send(map[string]any{}, "stop")
	_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
}

// writeEmptyCompletion returns a valid empty completion when every attempt
// was empty (nothing has been written to the client yet).
func (ps *proxyServer) writeEmptyCompletion(w http.ResponseWriter, slug string, req completionRequest, promptTokens int) {
	writeJSON(w, http.StatusOK, map[string]any{
		"id": newID("chatcmpl-"), "object": "chat.completion",
		"created": time.Now().Unix(), "model": modelNameFor(req, slug),
		"choices": []any{map[string]any{
			"index":         0,
			"message":       map[string]any{"role": "assistant", "content": ""},
			"finish_reason": "stop",
		}},
		"usage":              usageInfo{PromptTokens: promptTokens, CompletionTokens: 1, TotalTokens: promptTokens + 1},
		"system_fingerprint": slug,
	})
}

// cleanupChat best-effort deletes the upstream chat. Runs after the response
// has been fully written, on the same session (before it returns to the pool,
// so token rotation cannot invalidate the delete).
func (ps *proxyServer) cleanupChat(s *Session, chatID string) {
	if chatID == "" {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := s.DeleteChat(ctx, ps.base, chatID); err != nil {
		log.Printf("[cleanup] delete chat %s: %v", chatID, err)
	} else {
		log.Printf("[cleanup] chat %s deleted", chatID)
	}
}

// settle returns the session to the pool, dropping it if the session broke.
func (ps *proxyServer) settle(s *Session, err error) {
	if err != nil && isDead(err) {
		ps.pool.Drop(s)
		return
	}
	ps.pool.put(s)
}

type streamResult struct {
	reasoning strings.Builder
	content   strings.Builder
	raw       strings.Builder
	final     streamFinal
}

// consumeInto feeds the upstream stream through the splitter, collecting the
// final text for usage stats.
func consumeInto(resp *http.Response, onDelta func(reasoning, content string)) (*streamResult, error) {
	splitter := NewThinkSplitter()
	res := &streamResult{}
	err := consumeDataStream(resp.Body,
		func(chunk string) error {
			res.raw.WriteString(chunk)
			reasoning, content := splitter.Feed(chunk)
			onDelta(reasoning, content)
			return nil
		},
		func(streamMeta) {},
		func(f streamFinal) { res.final = f },
	)
	// flush bytes still buffered in the splitter
	reasoning, content := splitter.Finish()
	onDelta(reasoning, content)
	return res, err
}

// streamToClient proxies an accepted upstream stream as OpenAI SSE.
// It returns true when the upstream stream was empty (no content before
// EOF/final) AND nothing has been sent to the client yet — the caller may
// then retry the whole completion on a fresh chat.
// Session release and chat cleanup happen in the caller.
func (ps *proxyServer) streamToClient(w http.ResponseWriter, r *http.Request,
	resp *http.Response, slug string, req completionRequest, promptTokens int) bool {

	flusher, ok := w.(http.Flusher)
	if !ok {
		resp.Body.Close()
		writeOpenAIError(w, http.StatusInternalServerError, "server_error", "streaming unsupported by transport")
		return false
	}

	modelName := modelNameFor(req, slug)
	chunkID := newID("chatcmpl-")

	// SSE gate: headers are written only when the first real (non-whitespace)
	// content arrives, so empty-stream flake modes can retry silently.
	committed := false
	var bufRea, bufCon strings.Builder
	var raw strings.Builder
	seenText := false
	splitter := NewThinkSplitter()

	writeHeaders := func() {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.Header().Set("X-Accel-Buffering", "no")
		w.WriteHeader(http.StatusOK)
		flusher.Flush()
	}
	firstDelta := true
	send := func(delta map[string]any, finishReason any, usage *usageInfo) {
		if firstDelta && delta != nil {
			delta["role"] = "assistant"
			firstDelta = false
		}
		chunk := map[string]any{
			"id":      chunkID,
			"object":  "chat.completion.chunk",
			"created": time.Now().Unix(),
			"model":   modelName,
			"choices": []any{map[string]any{
				"index":         0,
				"delta":         delta,
				"finish_reason": finishReason,
			}},
		}
		if usage != nil {
			chunk["choices"] = []any{}
			chunk["usage"] = usage
		}
		data, _ := json.Marshal(chunk)
		_, _ = fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}
	commit := func() {
		if committed {
			return
		}
		committed = true
		writeHeaders()
	}

	res := &streamResult{}
	cerr := consumeDataStream(resp.Body,
		func(chunk string) error {
			raw.WriteString(chunk)
			if !seenText && strings.IndexFunc(chunk, func(rn rune) bool { return !unicode.IsSpace(rn) }) >= 0 {
				seenText = true
			}
			rea, con := splitter.Feed(chunk)
			if !committed {
				bufRea.WriteString(rea)
				bufCon.WriteString(con)
				if seenText {
					commit()
					if bufRea.Len() > 0 {
						send(map[string]any{"reasoning_content": bufRea.String()}, nil, nil)
						bufRea.Reset()
					}
					if bufCon.Len() > 0 {
						send(map[string]any{"content": bufCon.String()}, nil, nil)
						bufCon.Reset()
					}
				}
				return nil
			}
			if rea != "" {
				send(map[string]any{"reasoning_content": rea}, nil, nil)
			}
			if con != "" {
				send(map[string]any{"content": con}, nil, nil)
			}
			return nil
		},
		func(streamMeta) {},
		func(f streamFinal) { res.final = f },
	)
	resp.Body.Close()

	// flush splitter tail into the right place (buffered or live)
	tailRea, tailCon := splitter.Finish()
	if !committed {
		bufRea.WriteString(tailRea)
		bufCon.WriteString(tailCon)
		if strings.TrimSpace(raw.String()) != "" {
			// model produced something whitespace-trimmed-nonempty late in the
			// stream — commit and replay everything that accumulated
			commit()
			if bufRea.Len() > 0 {
				send(map[string]any{"reasoning_content": bufRea.String()}, nil, nil)
				bufRea.Reset()
			}
			if bufCon.Len() > 0 {
				send(map[string]any{"content": bufCon.String()}, nil, nil)
				bufCon.Reset()
			}
		}
	} else {
		if tailRea != "" {
			send(map[string]any{"reasoning_content": tailRea}, nil, nil)
		}
		if tailCon != "" {
			send(map[string]any{"content": tailCon}, nil, nil)
		}
	}

	if !committed {
		// transport-level empty stream (metadata only / whitespace / read
		// error before any content): nothing was sent, caller can retry.
		if cerr != nil {
			log.Printf("[stream] upstream read error before any content: %v", cerr)
		}
		return true
	}

	if cerr != nil {
		log.Printf("[stream] upstream error mid-stream: %v", cerr)
		send(map[string]any{"content": "\n[proxy error: " + cerr.Error() + "]"}, nil, nil)
		send(map[string]any{}, "stop", nil)
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
		return false
	}

	finalText := res.final.FinalContent
	if finalText == "" {
		finalText = raw.String()
	}
	completionTokens := estTokens(finalText)
	usage := &usageInfo{
		PromptTokens:     promptTokens,
		CompletionTokens: completionTokens,
		TotalTokens:      promptTokens + completionTokens,
	}
	send(map[string]any{}, "stop", nil)
	if req.StreamOptions != nil && req.StreamOptions.IncludeUsage {
		send(nil, nil, usage)
	}
	_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
	return false
}

// blockToClient waits for the full upstream stream and returns one OpenAI
// JSON completion. It returns true when the result was empty and nothing was
// written yet (caller may retry). Session release and chat cleanup happen in
// the caller.
func (ps *proxyServer) blockToClient(w http.ResponseWriter, r *http.Request,
	resp *http.Response, slug string, req completionRequest, promptTokens int) bool {

	res, cerr := consumeInto(resp, func(string, string) {})
	resp.Body.Close()
	if cerr != nil {
		writeUpstreamError(w, cerr)
		return false
	}
	if isEmptyStreamResult(res) {
		return true
	}

	raw := res.final.FinalContent
	if raw == "" {
		raw = res.raw.String()
	}
	reasoning, content := splitThinkFull(raw)
	message := map[string]any{"role": "assistant", "content": content}
	if reasoning != "" {
		message["reasoning_content"] = reasoning
	}
	completionTokens := estTokens(raw)
	modelOut := req.Model
	if modelOut == "" {
		modelOut = slug
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"id":      newID("chatcmpl-"),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   modelOut,
		"choices": []any{map[string]any{
			"index":         0,
			"message":       message,
			"finish_reason": "stop",
		}},
		"usage": usageInfo{
			PromptTokens:     promptTokens,
			CompletionTokens: completionTokens,
			TotalTokens:      promptTokens + completionTokens,
		},
		"system_fingerprint": slug,
	})
	// push the response out so the follow-up delete adds no visible latency
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
	return false
}

// isEmptyStreamResult reports a transport-level empty stream: upstream closed
// or finished without producing any text chunk at all.
func isEmptyStreamResult(res *streamResult) bool {
	return strings.TrimSpace(res.raw.String()) == "" && strings.TrimSpace(res.final.FinalContent) == ""
}

// handleModels implements GET /v1/models.
func (ps *proxyServer) handleModels(w http.ResponseWriter, r *http.Request) {
	created := time.Unix(0, 0).Unix()
	data := make([]map[string]any, 0, len(catalog))
	for _, m := range catalog {
		data = append(data, map[string]any{
			"id": m.ID, "object": "model", "created": created, "owned_by": "liquid-ai",
			"slug": m.Slug, "name": m.Name,
		})
	}
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": data})
}
