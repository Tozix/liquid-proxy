package main

// Upstream client for playground.liquid.ai: anonymous session management and
// the chat data-stream API. Sessions are checked out exclusively because the
// upstream rotates the session token on auth calls and invalidates the
// previous token once a newer one is used.

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"strings"
	"sync"
)

const (
	defaultModelSlug = "lfm2.5-24b-a2b-20261027"
	signupCallback   = "https://playground.liquid.ai/sign-in?callbackUrl=%2F"
	userAgent        = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/156.0.0.0 Safari/537.36"
)

// ---------------------------------------------------------------------------
// Errors

type errSessionDead struct {
	status int
	body   string
}

func (e errSessionDead) Error() string {
	return fmt.Sprintf("upstream auth failed: %d %.200s", e.status, e.body)
}

func isDead(err error) bool {
	_, ok := err.(errSessionDead)
	return ok
}

type errUpstream struct {
	status int
	body   string
}

func (e errUpstream) Error() string {
	return fmt.Sprintf("upstream %d: %.300s", e.status, e.body)
}

// ---------------------------------------------------------------------------
// Session

type Session struct {
	hc     *http.Client
	jar    map[string]string
	userID string
}

func newSession(ctx context.Context, base string, hc *http.Client) (*Session, error) {
	s := &Session{hc: hc, jar: map[string]string{}}

	// 1. CSRF token (server also sets the __Host-authjs.csrf-token cookie).
	body, err := s.do(ctx, http.MethodGet, base+"/api/auth/csrf", nil, "", nil)
	if err != nil {
		return nil, fmt.Errorf("csrf: %w", err)
	}
	var csrf struct {
		CSRFToken string `json:"csrfToken"`
	}
	if err := json.Unmarshal(body, &csrf); err != nil || csrf.CSRFToken == "" {
		return nil, fmt.Errorf("csrf: bad response %q", string(body))
	}

	// 2. Anonymous credentials sign-in (same request the web app performs).
	form := url.Values{
		"email":       {""},
		"password":    {""},
		"isAnonymous": {"true"},
		"csrfToken":   {csrf.CSRFToken},
		"callbackUrl": {signupCallback},
		"json":        {"true"},
	}
	if _, err := s.do(ctx, http.MethodPost, base+"/api/auth/callback/credentials",
		[]byte(form.Encode()), "application/x-www-form-urlencoded",
		map[string]string{"x-auth-return-redirect": "1"}); err != nil {
		return nil, fmt.Errorf("signin: %w", err)
	}
	if s.jar["__Secure-authjs.session-token"] == "" {
		return nil, errors.New("signin: no session cookie issued")
	}

	// 3. Resolve the user id (this call may rotate the token; jar is updated).
	body, err = s.do(ctx, http.MethodGet, base+"/api/auth/session", nil, "", nil)
	if err != nil {
		return nil, fmt.Errorf("session: %w", err)
	}
	var sess struct {
		User *struct {
			ID string `json:"id"`
		} `json:"user"`
	}
	if err := json.Unmarshal(body, &sess); err != nil || sess.User == nil || sess.User.ID == "" {
		return nil, fmt.Errorf("session: not authenticated, body %.200s", string(body))
	}
	s.userID = sess.User.ID
	return s, nil
}

// do performs one buffered upstream request against the session's cookie jar
// and applies every Set-Cookie from the response (token rotation lives there).
func (s *Session) do(ctx context.Context, method, rawURL string, body []byte, contentType string, extra map[string]string) ([]byte, error) {
	resp, err := s.open(ctx, method, rawURL, body, contentType, extra)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, 16<<20))
	if err != nil {
		return nil, err
	}
	return data, nil
}

// open sends one upstream request and validates the response status. The
// caller is responsible for reading/closing resp.Body when status is 2xx.
func (s *Session) open(ctx context.Context, method, rawURL string, body []byte, contentType string, extra map[string]string) (*http.Response, error) {
	var req *http.Request
	var err error
	if body != nil {
		req, err = http.NewRequestWithContext(ctx, method, rawURL, bytes.NewReader(body))
	} else {
		req, err = http.NewRequestWithContext(ctx, method, rawURL, nil)
	}
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", userAgent)
	req.Header.Set("Accept", "*/*")
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	for k, v := range extra {
		req.Header.Set(k, v)
	}
	if cookie := s.cookieHeader(); cookie != "" {
		req.Header.Set("Cookie", cookie)
	}

	resp, err := s.hc.Do(req)
	if err != nil {
		return nil, err
	}
	s.applyCookies(resp)

	if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		resp.Body.Close()
		return nil, errSessionDead{status: resp.StatusCode, body: string(snippet)}
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		resp.Body.Close()
		return nil, errUpstream{status: resp.StatusCode, body: string(snippet)}
	}
	return resp, nil
}

func (s *Session) applyCookies(resp *http.Response) {
	for _, c := range resp.Cookies() {
		if c.Value == "" {
			delete(s.jar, c.Name)
			continue
		}
		s.jar[c.Name] = c.Value
	}
}

func (s *Session) cookieHeader() string {
	parts := make([]string, 0, len(s.jar))
	for k, v := range s.jar {
		parts = append(parts, k+"="+v)
	}
	return strings.Join(parts, "; ")
}

// ---------------------------------------------------------------------------
// Chat API

type chatInfo struct {
	NewChat struct {
		ID string `json:"id"`
	} `json:"newChat"`
}

func (s *Session) createChat(ctx context.Context, base string) (string, error) {
	payload, _ := json.Marshal(map[string]string{"userId": s.userID})
	data, err := s.do(ctx, http.MethodPost, base+"/api/chats", payload, "application/json", nil)
	if err != nil {
		return "", err
	}
	var info chatInfo
	if err := json.Unmarshal(data, &info); err != nil || info.NewChat.ID == "" {
		return "", fmt.Errorf("create chat: bad response %.200s", string(data))
	}
	return info.NewChat.ID, nil
}

// DeleteChat soft-deletes the chat on the upstream (best-effort housekeeping:
// the proxy creates one chat per request and removes it after the response).
func (s *Session) DeleteChat(ctx context.Context, base, chatID string) error {
	_, err := s.do(ctx, http.MethodDelete,
		fmt.Sprintf("%s/api/chats/%s", base, url.PathEscape(chatID)), nil, "", nil)
	return err
}

// UploadImage registers an image with the playground and uploads it to the
// returned presigned storage URL. Returns the assetKey to reference in
// imageUuids of a chat stream request. Flow reversed from the web app.
func (s *Session) UploadImage(ctx context.Context, base string, data []byte, mime string) (string, error) {
	sum := sha256.Sum256(data)
	hash := hex.EncodeToString(sum[:])
	ext := "bin"
	switch mime {
	case "image/png":
		ext = "png"
	case "image/jpeg", "image/jpg":
		ext = "jpg"
	case "image/webp":
		ext = "webp"
	case "image/gif":
		ext = "gif"
	}
	payload, _ := json.Marshal([]map[string]any{{
		"fileName":    "image." + ext,
		"contentType": mime,
		"size":        len(data),
		"hash":        hash,
		"extension":   ext,
	}})
	respBody, err := s.do(ctx, http.MethodPost, base+"/api/images/upload", payload, "application/json", nil)
	if err != nil {
		return "", fmt.Errorf("presign: %w", err)
	}
	var infos []struct {
		AssetKey      string `json:"assetKey"`
		UploadURL     string `json:"uploadUrl"`
		AlreadyExists bool   `json:"alreadyExists"`
	}
	if err := json.Unmarshal(respBody, &infos); err != nil || len(infos) == 0 || infos[0].AssetKey == "" {
		return "", fmt.Errorf("presign: bad response %.200s", string(respBody))
	}
	info := infos[0]
	if !info.AlreadyExists && info.UploadURL != "" {
		// Presigned PUT to the external storage host — no session cookies.
		req, err := http.NewRequestWithContext(ctx, http.MethodPut, info.UploadURL, bytes.NewReader(data))
		if err != nil {
			return "", err
		}
		req.Header.Set("Content-Type", mime)
		putResp, err := s.hc.Do(req)
		if err != nil {
			return "", fmt.Errorf("storage put: %w", err)
		}
		defer putResp.Body.Close()
		io.Copy(io.Discard, io.LimitReader(putResp.Body, 4096))
		if putResp.StatusCode < 200 || putResp.StatusCode >= 300 {
			return "", fmt.Errorf("storage put: status %d", putResp.StatusCode)
		}
	}
	return info.AssetKey, nil
}

type streamMeta struct {
	UserMessageID      string `json:"userMessageId"`
	AssistantMessageID string `json:"assistantMessageId"`
}

type streamFinal struct {
	ID              string  `json:"id"`
	FinalContent    string  `json:"finalContent"`
	TokensPerSecond float64 `json:"tokensPerSecond"`
	MsToFirstToken  float64 `json:"msToFirstToken"`
}

// StartChatStream sends the message turn and returns once upstream accepted
// it (2xx). Auth failures surface here, before any byte of the body is read,
// so callers can retry cleanly on a fresh session. Close resp.Body after
// consumeDataStream.
func (s *Session) StartChatStream(ctx context.Context, base, chatID, modelSlug, content string, imageUuids []string) (*http.Response, error) {
	if imageUuids == nil {
		imageUuids = []string{}
	}
	payload, _ := json.Marshal(map[string]any{
		"content":         content,
		"modelSlug":       modelSlug,
		"parentMessageId": nil,
		"addons":          []string{},
		"mode":            "completion",
		"imageUuids":      imageUuids,
	})
	return s.open(ctx, http.MethodPost,
		fmt.Sprintf("%s/api/chats/%s/messages/stream", base, url.PathEscape(chatID)),
		payload, "application/json", nil)
}

// consumeDataStream parses a data-stream v1 body incrementally and returns
// when the upstream stream ends (EOF). It does not close the body.
func consumeDataStream(body io.Reader, onText func(string) error, onMeta func(streamMeta), onFinal func(streamFinal)) error {
	reader := bufio.NewReaderSize(body, 64<<10)
	for {
		line, err := reader.ReadString('\n')
		if line != "" {
			line = strings.TrimRight(line, "\r\n")
			if line != "" {
				if perr := dispatchStreamLine(line, onText, onMeta, onFinal); perr != nil {
					return perr
				}
			}
		}
		if err != nil {
			if errors.Is(err, io.EOF) {
				return nil
			}
			return err
		}
	}
}

func dispatchStreamLine(line string, onText func(string) error, onMeta func(streamMeta), onFinal func(streamFinal)) error {
	idx := strings.Index(line, ":")
	if idx < 0 {
		return nil
	}
	code, payload := line[:idx], line[idx+1:]
	switch code {
	case "0":
		var text string
		if err := json.Unmarshal([]byte(payload), &text); err != nil {
			return nil
		}
		if onText != nil {
			return onText(text)
		}
	case "2":
		var meta streamMeta
		if err := json.Unmarshal([]byte(payload), &meta); err == nil && onMeta != nil {
			onMeta(meta)
		}
	case "8":
		var finals []streamFinal
		if err := json.Unmarshal([]byte(payload), &finals); err == nil && len(finals) > 0 && onFinal != nil {
			onFinal(finals[0])
		}
	default:
		// other control codes of the v1 data stream are not needed
	}
	return nil
}

// ---------------------------------------------------------------------------
// Session pool

type SessionPool struct {
	base string
	hc   *http.Client

	mu    sync.Mutex
	cond  *sync.Cond
	free  []*Session
	total int
	max   int
}

func NewSessionPool(base string, max int) *SessionPool {
	p := &SessionPool{
		base: base,
		hc:   &http.Client{}, // no overall timeout; request ctx governs streams
		max:  max,
	}
	p.cond = sync.NewCond(&p.mu)
	return p
}

// Prewarm creates up to n sessions in parallel; failures are logged only.
func (p *SessionPool) Prewarm(n int) {
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			s, err := newSession(context.Background(), p.base, p.hc)
			if err != nil {
				log.Printf("[pool] prewarm failed: %v", err)
				return
			}
			p.mu.Lock()
			p.total++
			p.mu.Unlock()
			log.Printf("[pool] prewarmed session user=%s", maskID(s.userID))
			p.put(s)
		}()
	}
	wg.Wait()
}

// Acquire returns a free session, creating one if capacity allows; otherwise
// it blocks until another request releases its session or ctx is cancelled.
func (p *SessionPool) Acquire(ctx context.Context) (*Session, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	for {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		if len(p.free) > 0 {
			s := p.free[len(p.free)-1]
			p.free = p.free[:len(p.free)-1]
			return s, nil
		}
		if p.total < p.max {
			p.total++
			p.mu.Unlock()
			s, err := newSession(ctx, p.base, p.hc)
			p.mu.Lock()
			if err != nil {
				p.total--
				p.cond.Broadcast()
				return nil, err
			}
			log.Printf("[pool] created session user=%s", maskID(s.userID))
			return s, nil
		}
		p.cond.Wait()
	}
}

func (p *SessionPool) put(s *Session) {
	p.mu.Lock()
	p.free = append(p.free, s)
	p.cond.Broadcast()
	p.mu.Unlock()
}

// Drop discards a broken session; capacity is freed for a replacement.
func (p *SessionPool) Drop(s *Session) {
	log.Printf("[pool] dropping session user=%s", maskID(s.userID))
	p.mu.Lock()
	p.total--
	p.cond.Broadcast()
	p.mu.Unlock()
}

func maskID(id string) string {
	if len(id) <= 8 {
		return id
	}
	return id[:8] + "…"
}
