package main

import (
	"encoding/json"
	"testing"
)

type feedStep struct {
	in  string
	rea string
	con string
}

func runSplitter(steps []feedStep, finishRea, finishCon string) error {
	t := NewThinkSplitter()
	for i, st := range steps {
		rea, con := t.Feed(st.in)
		if rea != st.rea || con != st.con {
			return errStep(i, st, rea, con)
		}
	}
	rea, con := t.Finish()
	if rea != finishRea || con != finishCon {
		return errStep(-1, feedStep{in: "<finish>"}, rea, con)
	}
	return nil
}

func errStep(i int, want feedStep, gotRea, gotCon string) error {
	return errString{i: i, want: want, gotRea: gotRea, gotCon: gotCon}
}

type errString struct {
	i      int
	want   feedStep
	gotRea string
	gotCon string
}

func (e errString) Error() string {
	return "step " + itoa(e.i) + ": want (" + q(e.want.rea) + "," + q(e.want.con) + ") got (" + q(e.gotRea) + "," + q(e.gotCon) + ")"
}

func q(s string) string { return "\"" + s + "\"" }

func itoa(i int) string {
	if i == -1 {
		return "finish"
	}
	digits := "0123456789"
	if i < 10 {
		return digits[i : i+1]
	}
	return digits[i/10:i/10+1] + digits[i%10:i%10+1]
}

func TestSplitterPlain(t *testing.T) {
	steps := []feedStep{{in: "Hello world", con: "Hello world"}}
	if err := runSplitter(steps, "", ""); err != nil {
		t.Fatal(err)
	}
}

func TestSplitterWholeThink(t *testing.T) {
	steps := []feedStep{{
		in:  "<think>\nabc\n</think>\n\nAnswer",
		rea: "\nabc\n",
		con: "Answer",
	}}
	if err := runSplitter(steps, "", ""); err != nil {
		t.Fatal(err)
	}
}

func TestSplitterChunkedThink(t *testing.T) {
	steps := []feedStep{
		{in: "<thi"},
		{in: "nk>\nrea", rea: "\nrea"},
		{in: "son</th", rea: "son"},
		{in: "ink>\n\nans", con: "ans"},
		{in: "wer", con: "wer"},
	}
	if err := runSplitter(steps, "", ""); err != nil {
		t.Fatal(err)
	}
}

func TestSplitterNoThinkTagLike(t *testing.T) {
	steps := []feedStep{{in: "<b>bold</b>", con: "<b>bold</b>"}}
	if err := runSplitter(steps, "", ""); err != nil {
		t.Fatal(err)
	}
}

func TestSplitterWhitespaceBeforeThink(t *testing.T) {
	steps := []feedStep{{
		in:  "\n<think>a</think>b",
		rea: "a",
		con: "b",
	}}
	if err := runSplitter(steps, "", ""); err != nil {
		t.Fatal(err)
	}
}

func TestSplitterUnclosedThink(t *testing.T) {
	steps := []feedStep{{in: "<think>abc", rea: "abc"}}
	if err := runSplitter(steps, "", ""); err != nil {
		t.Fatal(err)
	}
}

func TestSplitterThinkThenFinishNewlines(t *testing.T) {
	// exactly like the playground stream: "<think>…</think>\n\nI'm LFM,"
	steps := []feedStep{{
		in:  "<think>reasoning</think>\n\nI'm LFM,",
		rea: "reasoning",
		con: "I'm LFM,",
	}}
	if err := runSplitter(steps, "", ""); err != nil {
		t.Fatal(err)
	}
}

func TestSplitThinkFull(t *testing.T) {
	rea, con := splitThinkFull("<think>\nthought\n</think>\n\nFinal answer")
	if rea != "\nthought\n" || con != "Final answer" {
		t.Fatalf("want (\\nthought\\n, Final answer), got (%q, %q)", rea, con)
	}
	rea, con = splitThinkFull("<think>only reasoning")
	if rea != "only reasoning" || con != "" {
		t.Fatalf("unclosed: got (%q, %q)", rea, con)
	}
	rea, con = splitThinkFull("no think here")
	if rea != "" || con != "no think here" {
		t.Fatalf("plain: got (%q, %q)", rea, con)
	}
}

func TestSerializeMessages(t *testing.T) {
	msgs := []chatMessage{
		{Role: "system", Content: jsonRaw("Be brief.")},
		{Role: "user", Content: jsonRaw("Hi")},
		{Role: "assistant", Content: jsonRaw("Hello!")},
		{Role: "user", Content: jsonRaw("How are you?")},
	}
	got, _, err := serializeMessages(msgs)
	if err != nil {
		t.Fatal(err)
	}
	want := "Be brief.\n\nUser: Hi\n\nAssistant: Hello!\n\nUser: How are you?"
	if got != want {
		t.Fatalf("serialize: want %q got %q", want, got)
	}
}

func jsonRaw(s string) json.RawMessage {
	b, _ := json.Marshal(s)
	return b
}
