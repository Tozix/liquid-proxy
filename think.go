package main

import "strings"

// ThinkSplitter converts the raw model output of LFM reasoning models into
// (reasoning, content) deltas on the fly. Upstream reasoning arrives inline
// as "<think>...</think>" before the answer.
type ThinkSplitter struct {
	buf     []byte
	inThink bool
	decided bool // saw a real <think> or ruled it out
}

const (
	tagOpen  = "<think>"
	tagClose = "</think>"
)

func NewThinkSplitter() *ThinkSplitter { return &ThinkSplitter{} }

func isSpaceByte(b byte) bool {
	return b == ' ' || b == '\n' || b == '\t' || b == '\r'
}

// Feed consumes the next raw chunk and returns every (reasoning, content)
// delta that became decidable after it.
func (t *ThinkSplitter) Feed(p string) (reasoning, content string) {
	t.buf = append(t.buf, p...)
	var rea, con strings.Builder
	for {
		if !t.decided {
			i := 0
			for i < len(t.buf) && isSpaceByte(t.buf[i]) {
				i++
			}
			head := t.buf[i:]
			if len(head) < len(tagOpen) && hasPrefixOf(head, tagOpen) {
				break // still ambiguous, wait for more bytes
			}
			t.decided = true
			if len(head) >= len(tagOpen) && string(head[:len(tagOpen)]) == tagOpen {
				t.inThink = true
				t.buf = append([]byte(nil), t.buf[i+len(tagOpen):]...)
				continue
			}
			// no reasoning block: everything is content
		}
		if t.inThink {
			idx := strings.Index(string(t.buf), tagClose)
			if idx < 0 {
				// emit all but a possible partial-tag suffix, then wait
				keep := longestSuffixLen(string(t.buf), tagClose)
				if keep >= len(t.buf) {
					break
				}
				rea.Write(t.buf[:len(t.buf)-keep])
				t.buf = append([]byte(nil), t.buf[len(t.buf)-keep:]...)
				break
			}
			rea.Write(t.buf[:idx])
			t.buf = append([]byte(nil), t.buf[idx+len(tagClose):]...)
			t.inThink = false
			// drop newlines immediately after </think>
			t.buf = []byte(strings.TrimLeft(string(t.buf), "\r\n"))
			continue
		}
		con.Write(t.buf)
		t.buf = t.buf[:0]
		break
	}
	return rea.String(), con.String()
}

// Finish flushes the internal buffer at end of stream and returns any
// remaining (reasoning, content).
func (t *ThinkSplitter) Finish() (reasoning, content string) {
	out := string(t.buf)
	t.buf = nil
	if t.inThink {
		return out, ""
	}
	return "", out
}

// hasPrefixOf reports whether b could still become a prefix of tag.
func hasPrefixOf(b []byte, tag string) bool {
	if len(b) >= len(tag) {
		return false
	}
	return strings.HasPrefix(tag, string(b))
}

// longestSuffixLen returns the length of the longest suffix of s that is a
// proper prefix of tag (so a tag spanning chunk borders is not missed).
func longestSuffixLen(s, tag string) int {
	for n := len(tag) - 1; n > 0; n-- {
		if len(s) >= n && strings.HasSuffix(s, tag[:n]) {
			return n
		}
	}
	return 0
}

// splitThinkFull splits a complete raw completion into (reasoning, content).
func splitThinkFull(raw string) (reasoning, content string) {
	s := strings.TrimLeft(raw, " \r\n\t")
	if !strings.HasPrefix(s, tagOpen) {
		return "", raw
	}
	rest := s[len(tagOpen):]
	idx := strings.Index(rest, tagClose)
	if idx < 0 {
		return strings.TrimPrefix(rest, "\n"), ""
	}
	reasoning = rest[:idx]
	content = strings.TrimLeft(rest[idx+len(tagClose):], " \r\n\t")
	return reasoning, content
}
