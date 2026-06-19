// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package fix

import (
	"errors"
	"strconv"
	"strings"
	"testing"
)

// soh renders a human-written FIX string (| as the delimiter) into real SOH
// bytes, so test fixtures stay readable.
func soh(s string) []byte { return []byte(strings.ReplaceAll(s, "|", string(SOH))) }

// TestSerializeComputesBodyLengthAndChecksum proves Serialize emits a spec-correct
// frame: 8= first, 9= second with the EXACT body byte count, 10= last with the
// correct mod-256 checksum, all SOH-delimited.
func TestSerializeComputesBodyLengthAndChecksum(t *testing.T) {
	m := NewMessage(MsgTypeNewOrderSingle).
		Set(TagClOrdID, "ORD1").
		Set(TagSymbol, "LUX/LUSD").
		Set(TagSide, SideBuy).
		Set(TagOrderQty, "12").
		Set(TagOrdType, OrdTypeLimit).
		Set(TagPrice, "101")

	raw := m.Serialize()
	s := string(raw)

	if !strings.HasPrefix(s, "8=FIX.4.4"+string(SOH)+"9=") {
		t.Fatalf("frame must start 8=FIX.4.4<SOH>9=, got %q", s)
	}
	if s[len(s)-1] != SOH {
		t.Fatalf("frame must end with SOH")
	}

	// Independently recompute BodyLength: bytes after the 9=..SOH up to and
	// including the SOH before 10=.
	fields := strings.Split(s[:len(s)-1], string(SOH))
	// fields[0]=8=.., fields[1]=9=.., last=10=..
	blStr := strings.TrimPrefix(fields[1], "9=")
	declaredBL, err := strconv.Atoi(blStr)
	if err != nil {
		t.Fatalf("BodyLength not numeric: %q", fields[1])
	}
	head := "8=FIX.4.4" + string(SOH) + "9=" + blStr + string(SOH)
	csField := fields[len(fields)-1] + string(SOH) // "10=xxx" + SOH
	body := s[len(head) : len(s)-len(csField)]
	if len(body) != declaredBL {
		t.Fatalf("BodyLength=%d but body is %d bytes: body=%q", declaredBL, len(body), body)
	}

	// Independently recompute checksum over everything up to the 10= field.
	prefix := s[:len(s)-len(csField)]
	var sum int
	for _, b := range []byte(prefix) {
		sum += int(b)
	}
	wantCS := sum % 256
	gotCSStr := strings.TrimPrefix(fields[len(fields)-1], "10=")
	gotCS, _ := strconv.Atoi(gotCSStr)
	if gotCS != wantCS {
		t.Fatalf("checksum=%s want %03d", gotCSStr, wantCS)
	}
	if len(gotCSStr) != 3 {
		t.Fatalf("checksum must be 3 digits, got %q", gotCSStr)
	}
}

// TestParseRoundTrip proves a serialized message parses back to the same semantic
// fields (framing tags stripped), with integrity validated.
func TestParseRoundTrip(t *testing.T) {
	orig := NewMessage(MsgTypeNewOrderSingle).
		Set(TagSenderCompID, "CLIENT").
		Set(TagTargetCompID, "LXDEX").
		SetInt(TagMsgSeqNum, 2).
		Set(TagClOrdID, "ORD42").
		Set(TagSymbol, "LUX/LETH").
		Set(TagSide, SideSell).
		Set(TagOrderQty, "5").
		Set(TagOrdType, OrdTypeLimit).
		Set(TagPrice, "0.05")

	parsed, err := Parse(orig.Serialize())
	if err != nil {
		t.Fatalf("Parse round-trip: %v", err)
	}
	if parsed.MsgType() != MsgTypeNewOrderSingle {
		t.Fatalf("MsgType=%q want D", parsed.MsgType())
	}
	for _, tc := range []struct {
		tag  int
		want string
	}{
		{TagSenderCompID, "CLIENT"},
		{TagTargetCompID, "LXDEX"},
		{TagClOrdID, "ORD42"},
		{TagSymbol, "LUX/LETH"},
		{TagSide, SideSell},
		{TagOrderQty, "5"},
		{TagPrice, "0.05"},
	} {
		got, ok := parsed.Get(tc.tag)
		if !ok || got != tc.want {
			t.Errorf("tag %d = %q (ok=%v), want %q", tc.tag, got, ok, tc.want)
		}
	}
	if seq, err := parsed.GetInt(TagMsgSeqNum); err != nil || seq != 2 {
		t.Errorf("MsgSeqNum=%d err=%v want 2", seq, err)
	}
}

// TestParseRejectsBadChecksum proves a frame with a wrong CheckSum (tag 10) is
// rejected with ErrChecksum — corruption never reaches the matcher.
func TestParseRejectsBadChecksum(t *testing.T) {
	good := string(NewMessage(MsgTypeHeartbeat).Serialize())
	// Flip the last checksum digit.
	idx := strings.LastIndex(good, "10=")
	bad := good[:idx+3] + "999" + string(SOH)
	_, err := Parse([]byte(bad))
	if !errors.Is(err, ErrChecksum) {
		t.Fatalf("want ErrChecksum, got %v", err)
	}
}

// TestParseRejectsBadBodyLength proves a frame whose BodyLength (tag 9) lies is
// rejected with ErrBodyLength.
func TestParseRejectsBadBodyLength(t *testing.T) {
	good := string(NewMessage(MsgTypeNewOrderSingle).Set(TagClOrdID, "X").Serialize())
	// Rewrite 9=<n> to 9=<n+1> and recompute the checksum so ONLY the body length
	// is wrong (otherwise the checksum guard would fire first).
	fields := strings.Split(strings.TrimSuffix(good, string(SOH)), string(SOH))
	blStr := strings.TrimPrefix(fields[1], "9=")
	n, _ := strconv.Atoi(blStr)
	fields[1] = "9=" + strconv.Itoa(n+1)
	// Rebuild without the old checksum, recompute.
	prefix := strings.Join(fields[:len(fields)-1], string(SOH)) + string(SOH)
	var sum int
	for _, b := range []byte(prefix) {
		sum += int(b)
	}
	bad := prefix + "10=" + pad3(sum%256) + string(SOH)
	_, err := Parse([]byte(bad))
	if !errors.Is(err, ErrBodyLength) {
		t.Fatalf("want ErrBodyLength, got %v", err)
	}
}

// TestParseRejectsMalformed covers structural errors: empty, no trailing SOH, bad
// field, missing 8/9/10, missing MsgType.
func TestParseRejectsMalformed(t *testing.T) {
	cases := map[string][]byte{
		"empty":           {},
		"no trailing SOH": soh("8=FIX.4.4|9=5|35=0|10=000"),
		"bad field":       soh("8=FIX.4.4|9=5|garbage|10=000|"),
		"not begin 8":     soh("35=0|9=5|10=000|"),
		"wrong version":   soh("8=FIX.4.2|9=5|35=0|10=000|"),
	}
	for name, raw := range cases {
		if _, err := Parse(raw); err == nil {
			t.Errorf("%s: expected error, got nil", name)
		}
	}
}

// TestSetIgnoresFramingTags proves a caller cannot inject framing tags (8/9/10)
// into the body — they are derived by Serialize only.
func TestSetIgnoresFramingTags(t *testing.T) {
	m := NewMessage(MsgTypeHeartbeat).
		Set(TagBeginString, "FIX.4.2").
		Set(TagBodyLength, "999").
		Set(TagCheckSum, "111")
	raw := string(m.Serialize())
	if !strings.HasPrefix(raw, "8=FIX.4.4"+string(SOH)) {
		t.Fatalf("BeginString must stay FIX.4.4, got %q", raw)
	}
	if strings.Contains(raw, "9=999") || strings.Contains(raw, "10=111") {
		t.Fatalf("framing tags must not be settable from body: %q", raw)
	}
	// And it still parses (integrity intact).
	if _, err := Parse(m.Serialize()); err != nil {
		t.Fatalf("Parse after attempted framing injection: %v", err)
	}
}

func pad3(n int) string {
	s := strconv.Itoa(n)
	for len(s) < 3 {
		s = "0" + s
	}
	return s
}
