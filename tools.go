package main

// OpenAI function-calling support. The playground API has no tools parameter,
// but LFM models are trained for tool calling: given tool definitions in the
// prompt they emit native calls as
//
//	<|tool_call_start|>[tool_name(param='value', n=42)]<|tool_call_end|>
//
// We inject the tool list into the prompt, strip those tokens from the
// output, parse the Python-literal arguments and translate everything into
// OpenAI tool_calls.

import (
	"encoding/json"
	"fmt"
	"log"
	"strings"
	"unicode"
)

const (
	tcStart = "<|tool_call_start|>"
	tcEnd   = "<|tool_call_end|>"
)

type toolDef struct {
	Type     string `json:"type"`
	Function struct {
		Name        string          `json:"name"`
		Description string          `json:"description"`
		Parameters  json.RawMessage `json:"parameters"`
	} `json:"function"`
}

// ToolCall is a parsed model tool call, ready for the OpenAI wire format.
type ToolCall struct {
	ID        string
	Name      string
	Arguments string // compact JSON object
}

// toolsPreamble renders the tool definitions for the prompt so the model
// falls back into its trained tool-calling format.
func toolsPreamble(tools []toolDef) (string, error) {
	if len(tools) == 0 {
		return "", nil
	}
	list := make([]map[string]any, 0, len(tools))
	for _, t := range tools {
		if t.Function.Name == "" {
			continue
		}
		entry := map[string]any{
			"name":        t.Function.Name,
			"description": t.Function.Description,
		}
		if len(t.Function.Parameters) > 0 {
			var params any
			if err := json.Unmarshal(t.Function.Parameters, &params); err == nil {
				entry["parameters"] = params
			}
		}
		list = append(list, entry)
	}
	schema, err := json.Marshal(list)
	if err != nil {
		return "", err
	}
	var b strings.Builder
	b.WriteString("# Tools\n\n")
	b.WriteString("You can call tools. Available tools (JSON):\n")
	b.WriteString(string(schema))
	b.WriteString("\n\nTo call one, output exactly (special tokens included):\n")
	b.WriteString(tcStart + "[tool_name(param='value', flag=true)]" + tcEnd + "\n")
	b.WriteString("IMPORTANT: tool names are case-sensitive — copy the name EXACTLY as given above ")
	b.WriteString("(e.g. if the tool is listed as \"Bash\", call Bash, not bash; if \"ReadFile\", not read_file).\n")
	b.WriteString("Multiple calls are allowed in one reply. Arguments use Python literals: strings in quotes, numbers, true/false, None, lists, dicts.\n")
	b.WriteString("After you receive the tool result you may answer the user or call more tools. Do not invent results.\n")
	return b.String(), nil
}

// ---------------------------------------------------------------------------
// Tool name normalization (models often lowercase or snake_case tool names)

// nameIndex maps every plausible spelling of a declared tool name onto the
// exact name the client registered, so a call never fails with "tool not found".
type nameIndex map[string]string

func buildNameIndex(tools []toolDef) nameIndex {
	idx := nameIndex{}
	for _, t := range tools {
		exact := t.Function.Name
		if exact == "" {
			continue
		}
		for _, variant := range nameVariants(exact) {
			if _, taken := idx[variant]; !taken {
				idx[variant] = exact
			}
		}
	}
	return idx
}

// nameVariants produces the spellings we accept for one declared name:
// the exact name, lowercased, snake_cased, kebab-cased, and unseparated.
func nameVariants(name string) []string {
	base := []string{name, strings.ToLower(name)}
	snake := toSnakeCase(name)
	kebab := strings.ReplaceAll(snake, "_", "-")
	bare := strings.ReplaceAll(snake, "_", "")
	return append(base, snake, kebab, bare)
}

// toSnakeCase converts CamelCase / PascalCase to snake_case.
func toSnakeCase(s string) string {
	var b strings.Builder
	runes := []rune(s)
	for i, r := range runes {
		if unicode.IsUpper(r) {
			if i > 0 {
				prev := runes[i-1]
				nextLower := i+1 < len(runes) && unicode.IsLower(runes[i+1])
				if unicode.IsLower(prev) || unicode.IsDigit(prev) || (unicode.IsUpper(prev) && nextLower) {
					b.WriteRune('_')
				}
			}
			b.WriteRune(unicode.ToLower(r))
			continue
		}
		if r == '-' || r == ' ' {
			b.WriteRune('_')
			continue
		}
		b.WriteRune(r)
	}
	return b.String()
}

// normalize resolves a model-emitted name to the declared one, if possible.
func (idx nameIndex) normalize(emitted string) string {
	if len(idx) == 0 {
		return emitted
	}
	if exact, ok := idx[emitted]; ok {
		return exact
	}
	lower := strings.ToLower(strings.TrimSpace(emitted))
	if exact, ok := idx[lower]; ok {
		return exact
	}
	if exact, ok := idx[toSnakeCase(emitted)]; ok {
		return exact
	}
	// last resort: suffix/prefix tolerant match (read_file -> read)
	snake := toSnakeCase(emitted)
	for variant, exact := range idx {
		if len(variant) >= 3 && (strings.HasSuffix(snake, variant) || strings.HasPrefix(snake, variant)) {
			log.Printf("[tools] fuzzy-matched %q -> %q", emitted, exact)
			return exact
		}
	}
	// unknown: pass through so the client reports it, but leave a trace
	log.Printf("[tools] WARNING unknown tool name %q (declared: %v)", emitted, idx.declaredNames())
	return emitted
}

// declaredNames lists the original (non-normalized) tool names for logging.
func (idx nameIndex) declaredNames() []string {
	seen := map[string]bool{}
	var out []string
	for _, exact := range idx {
		if !seen[exact] {
			seen[exact] = true
			out = append(out, exact)
		}
	}
	return out
}

// ---------------------------------------------------------------------------
// Python-literal value parser (strings, numbers, bools, None, lists, dicts)

type pyScanner struct {
	s []rune
	i int
}

func (p *pyScanner) skipWS() {
	for p.i < len(p.s) && unicode.IsSpace(p.s[p.i]) {
		p.i++
	}
}

func (p *pyScanner) readString(quote rune) (string, error) {
	var b strings.Builder
	p.i++ // opening quote
	for p.i < len(p.s) {
		r := p.s[p.i]
		if r == '\\' && p.i+1 < len(p.s) {
			p.i++
			switch p.s[p.i] {
			case 'n':
				b.WriteRune('\n')
			case 't':
				b.WriteRune('\t')
			case 'r':
				b.WriteRune('\r')
			default:
				b.WriteRune(p.s[p.i])
			}
			p.i++
			continue
		}
		if r == quote {
			p.i++
			return b.String(), nil
		}
		b.WriteRune(r)
		p.i++
	}
	return "", fmt.Errorf("unterminated string")
}

func (p *pyScanner) readNumber() (any, error) {
	start := p.i
	if p.i < len(p.s) && (p.s[p.i] == '-' || p.s[p.i] == '+') {
		p.i++
	}
	for p.i < len(p.s) && (unicode.IsDigit(p.s[p.i]) || p.s[p.i] == '.') {
		p.i++
	}
	lit := string(p.s[start:p.i])
	if strings.Contains(lit, ".") {
		var f float64
		if _, err := fmt.Sscanf(lit, "%g", &f); err != nil {
			return nil, fmt.Errorf("bad number %q", lit)
		}
		return f, nil
	}
	var n int
	if _, err := fmt.Sscanf(lit, "%d", &n); err != nil {
		return nil, fmt.Errorf("bad number %q", lit)
	}
	return n, nil
}

func (p *pyScanner) readValue() (any, error) {
	p.skipWS()
	if p.i >= len(p.s) {
		return nil, fmt.Errorf("unexpected end of arguments")
	}
	switch r := p.s[p.i]; {
	case r == '\'' || r == '"':
		return p.readString(r)
	case r == '[':
		p.i++
		var arr []any
		for {
			p.skipWS()
			if p.i < len(p.s) && p.s[p.i] == ']' {
				p.i++
				return arr, nil
			}
			v, err := p.readValue()
			if err != nil {
				return nil, err
			}
			arr = append(arr, v)
			p.skipWS()
			if p.i < len(p.s) && p.s[p.i] == ',' {
				p.i++
				continue
			}
			if p.i < len(p.s) && p.s[p.i] == ']' {
				p.i++
				return arr, nil
			}
			return nil, fmt.Errorf("expected , or ] in list")
		}
	case r == '{':
		p.i++
		obj := map[string]any{}
		for {
			p.skipWS()
			if p.i < len(p.s) && p.s[p.i] == '}' {
				p.i++
				return obj, nil
			}
			var key string
			if p.i < len(p.s) && (p.s[p.i] == '\'' || p.s[p.i] == '"') {
				key, _ = p.readString(p.s[p.i])
			} else {
				start := p.i
				for p.i < len(p.s) && (unicode.IsLetter(p.s[p.i]) || unicode.IsDigit(p.s[p.i]) || p.s[p.i] == '_') {
					p.i++
				}
				key = string(p.s[start:p.i])
			}
			p.skipWS()
			if p.i >= len(p.s) || p.s[p.i] != ':' {
				return nil, fmt.Errorf("expected : in dict")
			}
			p.i++
			v, err := p.readValue()
			if err != nil {
				return nil, err
			}
			obj[key] = v
			p.skipWS()
			if p.i < len(p.s) && p.s[p.i] == ',' {
				p.i++
				continue
			}
			if p.i < len(p.s) && p.s[p.i] == '}' {
				p.i++
				return obj, nil
			}
			return nil, fmt.Errorf("expected , or } in dict")
		}
	case unicode.IsDigit(r) || r == '-' || r == '+':
		return p.readNumber()
	default:
		start := p.i
		for p.i < len(p.s) && unicode.IsLetter(p.s[p.i]) {
			p.i++
		}
		word := string(p.s[start:p.i])
		switch strings.ToLower(word) {
		case "true":
			return true, nil
		case "false":
			return false, nil
		case "none", "null":
			return nil, nil
		case "":
			return nil, fmt.Errorf("unexpected character %q", r)
		default:
			return word, nil // bareword: keep as string
		}
	}
}

// parsePyArgs parses "k1=v1, k2='v2'" into a JSON object string. A bare
// object literal — "{...}" or "{'k': 'v'}" — is accepted too: models often
// emit JSON-style arguments although the native format is Python kwargs.
func parsePyArgs(s string) (string, error) {
	trimmed := strings.TrimSpace(s)
	if strings.HasPrefix(trimmed, "{") {
		p := &pyScanner{s: []rune(trimmed)}
		v, err := p.readValue()
		if err == nil {
			if obj, ok := v.(map[string]any); ok {
				out, merr := json.Marshal(obj)
				if merr == nil {
					return string(out), nil
				}
			}
		}
		// fall through to kwargs parsing for malformed literals
	}
	p := &pyScanner{s: []rune(s)}
	obj := map[string]any{}
	first := true
	for {
		p.skipWS()
		if p.i >= len(p.s) {
			break
		}
		if !first {
			if p.s[p.i] != ',' {
				return "", fmt.Errorf("expected , between arguments at %d", p.i)
			}
			p.i++
			p.skipWS()
			if p.i >= len(p.s) {
				break
			}
		}
		start := p.i
		for p.i < len(p.s) && (unicode.IsLetter(p.s[p.i]) || unicode.IsDigit(p.s[p.i]) || p.s[p.i] == '_') {
			p.i++
		}
		key := string(p.s[start:p.i])
		if key == "" {
			return "", fmt.Errorf("empty argument name at %d", start)
		}
		p.skipWS()
		if p.i >= len(p.s) || p.s[p.i] != '=' {
			return "", fmt.Errorf("expected = after %q", key)
		}
		p.i++
		v, err := p.readValue()
		if err != nil {
			return "", err
		}
		obj[key] = v
		first = false
	}
	out, err := json.Marshal(obj)
	if err != nil {
		return "", err
	}
	return string(out), nil
}

// ---------------------------------------------------------------------------
// Extraction from model output

// matchBracket returns the index of the bracket closing s[open]
// (quote-aware), or -1.
func matchBracket(s string, open int) int {
	depth := 0
	var quote rune
	for i := open; i < len(s); i++ {
		r := rune(s[i])
		if quote != 0 {
			if r == '\\' {
				i++
			} else if r == quote {
				quote = 0
			}
			continue
		}
		switch r {
		case '\'', '"':
			quote = r
		case '[', '(', '{':
			depth++
		case ']', ')', '}':
			depth--
			if depth == 0 {
				return i
			}
		}
	}
	return -1
}

func isIdent(s string) bool {
	if s == "" {
		return false
	}
	for i, r := range s {
		if !(unicode.IsLetter(r) || r == '_' || (i > 0 && unicode.IsDigit(r))) {
			return false
		}
	}
	return true
}

// parseCallGroups parses "name(args), name2(args2)" sequences (quote-aware).
// Returns false when the text does not look like call groups at all.
func parseCallGroups(s string) ([]ToolCall, bool) {
	p := &pyScanner{s: []rune(s)}
	var calls []ToolCall
	for {
		p.skipWS()
		if p.i >= len(p.s) {
			break
		}
		start := p.i
		for p.i < len(p.s) && (unicode.IsLetter(p.s[p.i]) || unicode.IsDigit(p.s[p.i]) || p.s[p.i] == '_') {
			p.i++
		}
		name := string(p.s[start:p.i])
		if !isIdent(name) {
			return nil, false
		}
		p.skipWS()
		if p.i >= len(p.s) || p.s[p.i] != '(' {
			return nil, false
		}
		p.i++
		argStart := p.i
		depth := 1
		var quote rune
		for p.i < len(p.s) {
			r := p.s[p.i]
			if quote != 0 {
				if r == '\\' {
					p.i++
				} else if r == quote {
					quote = 0
				}
				p.i++
				continue
			}
			switch r {
			case '\'', '"':
				quote = r
			case '(', '[', '{':
				depth++
			case ')', ']', '}':
				depth--
			}
			if depth == 0 {
				break
			}
			p.i++
		}
		if depth != 0 {
			return nil, false
		}
		argsStr := string(p.s[argStart:p.i])
		p.i++ // closing ')'
		argsJSON, err := parsePyArgs(argsStr)
		if err != nil {
			fallback, _ := json.Marshal(map[string]string{"_raw": strings.TrimSpace(argsStr)})
			argsJSON = string(fallback)
		}
		calls = append(calls, ToolCall{Name: name, Arguments: argsJSON})
		p.skipWS()
		if p.i < len(p.s) && p.s[p.i] == ',' {
			p.i++
			continue
		}
		break
	}
	// trailing prose after the groups is tolerated (models sometimes add it)
	return calls, true
}

// parseCallBlock parses the text between one <|tool_call_start|>/<|tool_call_end|>
// pair. The native format wraps one or more calls in a bracket list:
// [name(args), name2(args2)] — a bare no-argument "[name]" is also accepted.
func parseCallBlock(body string) []ToolCall {
	b := strings.TrimSpace(body)
	if b == "" {
		return nil
	}
	if strings.HasPrefix(b, "[") {
		if close := matchBracket(b, 0); close > 0 {
			inner := strings.TrimSpace(b[1:close])
			if inner != "" {
				// a no-argument call like [ping]
				if !strings.Contains(inner, "(") && isIdent(inner) {
					return []ToolCall{{Name: inner, Arguments: "{}"}}
				}
				if calls, ok := parseCallGroups(inner); ok {
					return calls
				}
			}
			// fall through: maybe the block was a single call without the list
		}
	}
	if calls, ok := parseCallGroups(b); ok {
		return calls
	}
	return nil
}

// extractToolCalls strips tool-call blocks from model output and parses them.
// Remaining plain text is returned as content.
func extractToolCalls(text string) (content string, calls []ToolCall) {
	var b strings.Builder
	rest := text
	for {
		start := strings.Index(rest, tcStart)
		if start < 0 {
			b.WriteString(rest)
			break
		}
		b.WriteString(rest[:start])
		after := rest[start+len(tcStart):]
		end := strings.Index(after, tcEnd)
		var body string
		if end < 0 {
			body = after
			rest = ""
		} else {
			body = after[:end]
			rest = after[end+len(tcEnd):]
		}
		for _, call := range parseCallBlock(body) {
			call.ID = newID("call_")
			calls = append(calls, call)
		}
		if end < 0 {
			break
		}
	}
	return b.String(), calls
}

// ---------------------------------------------------------------------------
// Streaming filter: pulls tool-call blocks out of the content delta flow

type toolCallFilter struct {
	buf      strings.Builder // accumulated call body (inside a block)
	hold     string          // pending bytes that may be a start/end token prefix
	inCall   bool
	seenAny  bool
	finished bool
}

func NewToolCallFilter() *toolCallFilter { return &toolCallFilter{} }

// Feed consumes a content delta and returns cleaned content plus any
// completed tool calls.
func (f *toolCallFilter) Feed(delta string) (string, []ToolCall) {
	if delta == "" {
		return "", nil
	}
	f.seenAny = true
	var out strings.Builder
	var calls []ToolCall
	pending := f.hold + delta
	f.hold = ""
	for {
		if !f.inCall {
			idx := strings.Index(pending, tcStart)
			if idx < 0 {
				// emit everything except a possible partial-token suffix
				keep := longestSuffixLen(pending, tcStart)
				out.WriteString(pending[:len(pending)-keep])
				f.hold = pending[len(pending)-keep:]
				break
			}
			out.WriteString(pending[:idx])
			pending = pending[idx+len(tcStart):]
			f.inCall = true
			continue
		}
		// inside a call block: accumulate until the end token
		end := strings.Index(pending, tcEnd)
		if end < 0 {
			keep := longestSuffixLen(pending, tcEnd)
			cut := len(pending) - keep
			f.buf.WriteString(pending[:cut])
			pending = pending[cut:]
			break
		}
		f.buf.WriteString(pending[:end])
		pending = pending[end+len(tcEnd):]
		body := f.buf.String()
		f.buf.Reset()
		f.inCall = false
		for _, call := range parseCallBlock(body) {
			call.ID = newID("call_")
			calls = append(calls, call)
		}
	}
	return out.String(), calls
}

// Finish flushes the filter at end of stream. Unterminated blocks are
// recovered as calls when possible, otherwise passed through as content.
func (f *toolCallFilter) Finish() (string, []ToolCall) {
	if f.finished {
		return "", nil
	}
	f.finished = true
	var out strings.Builder
	var calls []ToolCall
	pending := f.hold
	f.hold = ""
	if f.inCall {
		f.buf.WriteString(pending)
		pending = f.buf.String()
		f.buf.Reset()
		f.inCall = false
		for _, call := range parseCallBlock(pending) {
			call.ID = newID("call_")
			calls = append(calls, call)
		}
	}
	out.WriteString(pending)
	return out.String(), calls
}
