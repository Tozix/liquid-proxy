package main

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestParsePyArgsBasic(t *testing.T) {
	out, err := parsePyArgs(`command='ls -la /tmp', timeout=30`)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal([]byte(out), &m); err != nil {
		t.Fatal(err)
	}
	if m["command"] != "ls -la /tmp" || m["timeout"] != float64(30) {
		t.Fatalf("got %v", m)
	}
}

func TestParsePyArgsTypes(t *testing.T) {
	out, err := parsePyArgs(`s="double quoted", b=true, f=False, n=None, arr=[1, 'two', true], d={'k': 5}`)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal([]byte(out), &m); err != nil {
		t.Fatal(err)
	}
	if m["s"] != "double quoted" || m["b"] != true || m["f"] != false || m["n"] != nil {
		t.Fatalf("scalars: %v", m)
	}
	arr, ok := m["arr"].([]any)
	if !ok || len(arr) != 3 || arr[0] != float64(1) || arr[1] != "two" || arr[2] != true {
		t.Fatalf("arr: %v", m["arr"])
	}
	d, ok := m["d"].(map[string]any)
	if !ok || d["k"] != float64(5) {
		t.Fatalf("dict: %v", m["d"])
	}
}

func TestParsePyArgsEmpty(t *testing.T) {
	out, err := parsePyArgs("  ")
	if err != nil {
		t.Fatal(err)
	}
	if out != "{}" {
		t.Fatalf("got %q", out)
	}
}

func TestExtractToolCallsSingle(t *testing.T) {
	in := "<|tool_call_start|>[bash(command='ls -la', timeout=5)]<|tool_call_end|>"
	content, calls := extractToolCalls(in)
	if content != "" {
		t.Fatalf("content should be empty, got %q", content)
	}
	if len(calls) != 1 || calls[0].Name != "bash" {
		t.Fatalf("calls: %+v", calls)
	}
	var m map[string]any
	if err := json.Unmarshal([]byte(calls[0].Arguments), &m); err != nil {
		t.Fatalf("arguments not JSON: %v (%q)", err, calls[0].Arguments)
	}
	if m["command"] != "ls -la" || m["timeout"] != float64(5) {
		t.Fatalf("args: %v", m)
	}
}

func TestExtractToolCallsWithTextAndMulti(t *testing.T) {
	in := "Смотрю каталог.\n<|tool_call_start|>[ls(path='/etc'), read(file='hosts')]<|tool_call_end|>\nГотово."
	content, calls := extractToolCalls(in)
	if len(calls) != 2 || calls[0].Name != "ls" || calls[1].Name != "read" {
		t.Fatalf("calls: %+v", calls)
	}
	var m map[string]any
	_ = json.Unmarshal([]byte(calls[1].Arguments), &m)
	if m["file"] != "hosts" {
		t.Fatalf("args2: %v", m)
	}
	if !strings.Contains(content, "Смотрю каталог.") || !strings.Contains(content, "Готово.") {
		t.Fatalf("content: %q", content)
	}
	if strings.Contains(content, "tool_call") {
		t.Fatalf("tokens leaked into content: %q", content)
	}
}

func TestExtractToolCallsNoArgs(t *testing.T) {
	_, calls := extractToolCalls("<|tool_call_start|>[ping()]<|tool_call_end|>")
	if len(calls) != 1 || calls[0].Name != "ping" {
		t.Fatalf("calls: %+v", calls)
	}
	if calls[0].Arguments != "{}" {
		t.Fatalf("args: %q", calls[0].Arguments)
	}
}

func TestExtractToolCallsNone(t *testing.T) {
	content, calls := extractToolCalls("Обычный ответ без тулз.")
	if calls != nil || content != "Обычный ответ без тулз." {
		t.Fatalf("content %q calls %+v", content, calls)
	}
}

func TestExtractToolCallsEscapedQuotes(t *testing.T) {
	_, calls := extractToolCalls(`<|tool_call_start|>[bash(command='echo "hi \'there"')]<|tool_call_end|>`)
	if len(calls) != 1 {
		t.Fatalf("calls: %+v", calls)
	}
	var m map[string]any
	if err := json.Unmarshal([]byte(calls[0].Arguments), &m); err != nil {
		t.Fatalf("bad json: %v (%q)", err, calls[0].Arguments)
	}
	if !strings.Contains(m["command"].(string), `echo "hi 'there`) {
		t.Fatalf("command: %v", m["command"])
	}
}

func TestFilterStreamingChunks(t *testing.T) {
	f := NewToolCallFilter()
	parts := []string{"Провер", "ка.", "<|tool_call_st", "art|>[get_w", "eather(city='Tok", "yo', units='metric')] Ends: ", "<|tool_call_end|>", " Потом ответ."}
	var content strings.Builder
	var calls []ToolCall
	for _, p := range parts {
		c, cl := f.Feed(p)
		content.WriteString(c)
		calls = append(calls, cl...)
	}
	tail, cl := f.Finish()
	content.WriteString(tail)
	calls = append(calls, cl...)

	if len(calls) != 1 || calls[0].Name != "get_weather" {
		t.Fatalf("calls: %+v", calls)
	}
	var m map[string]any
	if err := json.Unmarshal([]byte(calls[0].Arguments), &m); err != nil {
		t.Fatalf("bad json: %v (%q)", err, calls[0].Arguments)
	}
	if m["city"] != "Tokyo" || m["units"] != "metric" {
		t.Fatalf("args: %v", m)
	}
	if !strings.Contains(content.String(), "Проверка.") || !strings.Contains(content.String(), " Потом ответ.") {
		t.Fatalf("content: %q", content.String())
	}
	if strings.Contains(content.String(), "tool_call") {
		t.Fatalf("tokens leaked: %q", content.String())
	}
}

func TestFilterPlainTextPassthrough(t *testing.T) {
	f := NewToolCallFilter()
	var content strings.Builder
	var calls []ToolCall
	for _, p := range []string{"Просто ", "текст ", "без тулз < и > символов."} {
		c, cl := f.Feed(p)
		content.WriteString(c)
		calls = append(calls, cl...)
	}
	tail, _ := f.Finish()
	content.WriteString(tail)
	if len(calls) != 0 {
		t.Fatalf("calls: %+v", calls)
	}
	if content.String() != "Просто текст без тулз < и > символов." {
		t.Fatalf("content: %q", content.String())
	}
}

func TestToolsPreamble(t *testing.T) {
	tools := []toolDef{}
	t1 := tools[0:0]
	_ = t1
	var td toolDef
	td.Type = "function"
	td.Function.Name = "bash"
	td.Function.Description = "Run a command"
	td.Function.Parameters = json.RawMessage(`{"type":"object","properties":{"command":{"type":"string"}}}`)
	pre, err := toolsPreamble([]toolDef{td})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(pre, `"bash"`) || !strings.Contains(pre, tcStart) || !strings.Contains(pre, "# Tools") {
		t.Fatalf("preamble: %q", pre)
	}
}

func TestToSnakeCase(t *testing.T) {
	cases := map[string]string{
		"Bash":          "bash",
		"ReadFile":      "read_file",
		"WebFetch":      "web_fetch",
		"TodoWrite":     "todo_write",
		"XMLParser":     "xml_parser",
		"already_snake": "already_snake",
		"kebab-case":    "kebab_case",
	}
	for in, want := range cases {
		if got := toSnakeCase(in); got != want {
			t.Errorf("toSnakeCase(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestNameIndexNormalize(t *testing.T) {
	var td1, td2, td3 toolDef
	td1.Function.Name = "Bash"
	td2.Function.Name = "ReadFile"
	td3.Function.Name = "WebFetch"
	idx := buildNameIndex([]toolDef{td1, td2, td3})

	cases := map[string]string{
		"Bash":        "Bash", // exact
		"bash":        "Bash", // lowercase -> declared
		"BASh":        "Bash", // lower fallback
		"read_file":   "ReadFile",
		"readFile":    "ReadFile",
		"read-file":   "ReadFile",
		"readfile":    "ReadFile",
		"webfetch":    "WebFetch",
		"web_fetch":   "WebFetch",
		"UnknownTool": "UnknownTool", // untouched
	}
	for in, want := range cases {
		if got := idx.normalize(in); got != want {
			t.Errorf("normalize(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestNameIndexEmpty(t *testing.T) {
	idx := buildNameIndex(nil)
	if got := idx.normalize("bash"); got != "bash" {
		t.Errorf("empty index should pass through, got %q", got)
	}
}
