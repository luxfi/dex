// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

// Package fix is a focused, dependency-free FIX Latest codec and session
// acceptor — the SERVER side of the FIX trading path.
//
// VERSION. The session layer is FIXT.1.1 (tag 8) and the application layer is
// FIX Latest — FIX.5.0SP2 plus Extension Pack 307, named on the Logon by
// DefaultApplVerID (1137) and ApplExtID (1156). FIX 5.0 split the two layers
// precisely so the application dialect can advance without touching the session,
// and FIX Trading Community recommends FIX Latest application messages over the
// FIX4 (8=FIX.4.4) profile this gateway used to speak. No client depended on the
// old BeginString, so the wire moved with the semantics rather than straddling
// both.
//
// It is ONE of four protocol front-ends onto the SAME matcher: the D-Chain DEX
// (pkg/dchain) served over the FROZEN dex_* ZAP surface (pkg/zapwire). ZAP
// clients speak dex_* directly; the 0x9010 V4 precompile speaks it via
// engine_zap; the WS server speaks it via a JSON translation; and FIX speaks it
// via this codec + acceptor. NONE of them owns a matcher — the acceptor decodes a
// FIX NewOrderSingle, calls the SAME dex_place / dex_submit the others call, and
// renders the consensus-computed result back as a FIX ExecutionReport. DRY,
// orthogonal: the matcher lives in ONE place and FIX is purely a wire translation.
//
// message.go is the wire codec: a FIX message is an ordered list of
// tag=value fields delimited by SOH (0x01). The codec computes and VALIDATES the
// two integrity fields every FIX message carries:
//
//   - BodyLength (tag 9): the byte count from the field AFTER BodyLength up to and
//     INCLUDING the SOH before CheckSum (tag 10). It is the second field, right
//     after BeginString (tag 8).
//   - CheckSum (tag 10): (sum of every byte up to and including the SOH before the
//     CheckSum field) mod 256, rendered as a zero-padded 3-digit string. It is
//     always the LAST field.
//
// Framing is identical across FIX versions, and this codec is the ONE definition
// of it in the tree — emit spec-correct, parse spec-correct, integrity-checked.
package fix

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
)

// SOH is the FIX field delimiter (Start Of Header, ASCII 0x01).
const SOH = '\x01'

// The version this gateway speaks. FIXT.1.1 is the session layer (tag 8); the
// application layer is named by ApplVerID + ApplExtID, together FIX Latest.
const (
	BeginString = "FIXT.1.1" // tag 8: session layer
	ApplVerID   = "9"        // tags 1128/1137: FIX.5.0SP2, the FIX Latest base
	ApplExtID   = "307"      // tag 1156: Extension Pack — SP2 + EP307 is FIX Latest
)

// FIX tag numbers used by this gateway. Only the tags the order path needs are
// named; the codec carries arbitrary tags generically.
const (
	TagBeginString   = 8   // session layer, e.g. "FIXT.1.1"
	TagBodyLength    = 9   // body length (integrity)
	TagMsgType       = 35  // message type, e.g. "D" NewOrderSingle
	TagSenderCompID  = 49  // sender identity
	TagTargetCompID  = 56  // target identity
	TagMsgSeqNum     = 34  // sequence number
	TagSendingTime   = 52  // sending timestamp
	TagCheckSum      = 10  // checksum (integrity), always last
	TagClOrdID       = 11  // client order id
	TagOrderID       = 37  // exchange order id
	TagExecID        = 17  // execution id
	TagOrdStatus     = 39  // order status
	TagExecType      = 150 // execution type
	TagSymbol        = 55  // instrument
	TagSide          = 54  // 1=buy 2=sell
	TagOrderQty      = 38  // quantity
	TagOrdType       = 40  // 1=market 2=limit
	TagPrice         = 44  // limit price
	TagTimeInForce   = 59  // 0=day 3=IOC ...
	TagTransactTime  = 60  // transaction time
	TagLastQty       = 32  // qty of this fill
	TagLastPx        = 31  // price of this fill
	TagLeavesQty     = 151 // unfilled remaining qty
	TagCumQty        = 14  // cumulative filled qty
	TagAvgPx         = 6   // average fill price
	TagText          = 58  // free text (reject reason)
	TagOrigClOrdID   = 41  // original ClOrdID (cancel)
	TagTestReqID     = 112 // test request id (heartbeat challenge)
	TagEncryptMethod = 98  // logon: encryption method (0=none)
	TagHeartBtInt    = 108 // logon: heartbeat interval (seconds)

	TagDefaultApplVerID = 1137 // logon: application version for the whole session
	TagApplExtID        = 1156 // logon: application Extension Pack number
)

// MsgType values (tag 35).
const (
	MsgTypeHeartbeat         = "0"
	MsgTypeTestRequest       = "1"
	MsgTypeReject            = "3"
	MsgTypeLogout            = "5"
	MsgTypeExecutionReport   = "8"
	MsgTypeLogon             = "A"
	MsgTypeNewOrderSingle    = "D"
	MsgTypeOrderCancelReq    = "F"
	MsgTypeOrderCancelReject = "9"
)

// Side values (tag 54).
const (
	SideBuy  = "1"
	SideSell = "2"
)

// OrdType values (tag 40).
const (
	OrdTypeMarket = "1"
	OrdTypeLimit  = "2"
)

// ExecType (150) says WHAT JUST HAPPENED. A fill is ExecTypeTrade — FIX Latest
// retired the FIX 4.x pair 1=PartialFill / 2=Fill, which duplicated order state
// into an event field. The resulting state belongs to OrdStatus alone, so read
// the two together: ExecTypeTrade + OrdStatusPartial is a fill that left the
// order working; ExecTypeTrade + OrdStatusFilled is the fill that completed it.
const (
	ExecTypeNew      = "0"
	ExecTypeTrade    = "F"
	ExecTypeCanceled = "4"
	ExecTypeRejected = "8"
)

// OrdStatus (39) says WHERE THE ORDER STANDS after that event. These five are
// the whole lifecycle this gateway can report.
const (
	OrdStatusNew      = "0"
	OrdStatusPartial  = "1"
	OrdStatusFilled   = "2"
	OrdStatusCanceled = "4"
	OrdStatusRejected = "8"
)

// ErrMalformed is returned when bytes are not a parseable FIX frame.
var ErrMalformed = errors.New("fix: malformed message")

// ErrChecksum is returned when a frame's CheckSum (tag 10) does not match.
var ErrChecksum = errors.New("fix: checksum mismatch")

// ErrBodyLength is returned when a frame's BodyLength (tag 9) does not match.
var ErrBodyLength = errors.New("fix: body length mismatch")

// field is one tag=value pair, preserving wire order.
type field struct {
	tag int
	val string
}

// Message is an ordered FIX message: a list of tag=value fields. The integrity
// fields (8/9/10) are NOT stored in fields — they are derived at Serialize time
// and verified at Parse time, so a Message describes the SEMANTIC content (msg
// type + body) independent of framing. Order of body fields is preserved.
type Message struct {
	fields []field
	index  map[int]string // last value per tag, for O(1) Get
}

// NewMessage builds an empty message of the given MsgType (tag 35). MsgType is
// always the FIRST body field after the standard header in FIX, which Serialize
// enforces.
func NewMessage(msgType string) *Message {
	m := &Message{index: make(map[int]string)}
	m.Set(TagMsgType, msgType)
	return m
}

// Set appends (or, for the integrity/header-derived tags, would replace) a field.
// It preserves insertion order for body fields and updates the lookup index. The
// caller must not Set tags 8/9/10 — they are framing, computed by Serialize.
func (m *Message) Set(tag int, val string) *Message {
	if tag == TagBeginString || tag == TagBodyLength || tag == TagCheckSum {
		// Framing tags are derived, never set on the semantic message.
		return m
	}
	// MsgType is unique and must stay first; replace in place if already present.
	if tag == TagMsgType {
		if _, ok := m.index[tag]; ok {
			for i := range m.fields {
				if m.fields[i].tag == tag {
					m.fields[i].val = val
				}
			}
			m.index[tag] = val
			return m
		}
	}
	m.fields = append(m.fields, field{tag: tag, val: val})
	m.index[tag] = val
	return m
}

// SetInt is Set for an integer value.
func (m *Message) SetInt(tag int, v int) *Message { return m.Set(tag, strconv.Itoa(v)) }

// Get returns the last value for a tag and whether it was present.
func (m *Message) Get(tag int) (string, bool) {
	v, ok := m.index[tag]
	return v, ok
}

// GetInt returns a tag's value parsed as int. Missing or non-numeric => error.
func (m *Message) GetInt(tag int) (int, error) {
	v, ok := m.index[tag]
	if !ok {
		return 0, fmt.Errorf("fix: missing tag %d", tag)
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return 0, fmt.Errorf("fix: tag %d not an int: %q", tag, v)
	}
	return n, nil
}

// GetFloat returns a tag's value parsed as float64. Missing or non-numeric =>
// error.
func (m *Message) GetFloat(tag int) (float64, error) {
	v, ok := m.index[tag]
	if !ok {
		return 0, fmt.Errorf("fix: missing tag %d", tag)
	}
	f, err := strconv.ParseFloat(v, 64)
	if err != nil {
		return 0, fmt.Errorf("fix: tag %d not a float: %q", tag, v)
	}
	return f, nil
}

// MsgType returns the message's tag-35 value (empty if absent).
func (m *Message) MsgType() string {
	v, _ := m.index[TagMsgType]
	return v
}

// Serialize renders the message to canonical FIX wire bytes with computed
// BodyLength (tag 9) and CheckSum (tag 10). Layout:
//
//	8=FIXT.1.1 <SOH> 9=<bodyLen> <SOH> <body...> 10=<checksum> <SOH>
//
// where <body...> is 35=<type> followed by every Set field IN ORDER, and
// bodyLen counts the bytes of <body...> exactly (from the first byte after the
// 9=...<SOH> up to and including the <SOH> preceding 10=).
func (m *Message) Serialize() []byte {
	var body strings.Builder
	// MsgType (35) must lead the body; Set keeps it first, but enforce defensively.
	if mt, ok := m.index[TagMsgType]; ok {
		body.WriteString(strconv.Itoa(TagMsgType))
		body.WriteByte('=')
		body.WriteString(mt)
		body.WriteByte(SOH)
	}
	for _, f := range m.fields {
		if f.tag == TagMsgType {
			continue // already written first
		}
		body.WriteString(strconv.Itoa(f.tag))
		body.WriteByte('=')
		body.WriteString(f.val)
		body.WriteByte(SOH)
	}
	bodyBytes := body.String()

	var head strings.Builder
	head.WriteString(strconv.Itoa(TagBeginString))
	head.WriteByte('=')
	head.WriteString(BeginString)
	head.WriteByte(SOH)
	head.WriteString(strconv.Itoa(TagBodyLength))
	head.WriteByte('=')
	head.WriteString(strconv.Itoa(len(bodyBytes)))
	head.WriteByte(SOH)

	prefix := head.String() + bodyBytes
	cs := checksum([]byte(prefix))
	return []byte(fmt.Sprintf("%s%d=%03d%c", prefix, TagCheckSum, cs, SOH))
}

// checksum is the FIX integrity checksum: sum of all bytes mod 256.
func checksum(b []byte) int {
	var sum int
	for _, c := range b {
		sum += int(c)
	}
	return sum % 256
}

// Parse decodes one complete FIX frame, VALIDATING BodyLength (tag 9) and
// CheckSum (tag 10). It returns the semantic Message (framing tags stripped). A
// frame MUST begin with 8=, carry 9= second, end with a well-formed 10= field,
// and its integrity fields must match the content exactly — otherwise an error
// (ErrMalformed / ErrBodyLength / ErrChecksum) is returned. This is the strict
// acceptor-side check: a corrupt or truncated frame is rejected, never matched.
func Parse(raw []byte) (*Message, error) {
	if len(raw) == 0 {
		return nil, fmt.Errorf("%w: empty", ErrMalformed)
	}
	// Split into tag=value tokens on SOH. A valid frame ends with SOH, so the
	// final split element is empty and dropped.
	s := string(raw)
	if s[len(s)-1] != SOH {
		return nil, fmt.Errorf("%w: no trailing SOH", ErrMalformed)
	}
	tokens := strings.Split(s[:len(s)-1], string(SOH))
	if len(tokens) < 3 {
		return nil, fmt.Errorf("%w: too few fields", ErrMalformed)
	}

	type kv struct {
		tag int
		val string
	}
	parsed := make([]kv, 0, len(tokens))
	for _, t := range tokens {
		eq := strings.IndexByte(t, '=')
		if eq <= 0 {
			return nil, fmt.Errorf("%w: bad field %q", ErrMalformed, t)
		}
		tag, err := strconv.Atoi(t[:eq])
		if err != nil {
			return nil, fmt.Errorf("%w: bad tag %q", ErrMalformed, t[:eq])
		}
		parsed = append(parsed, kv{tag: tag, val: t[eq+1:]})
	}

	// Mandatory positional fields: 8 first, 9 second, 10 last.
	if parsed[0].tag != TagBeginString {
		return nil, fmt.Errorf("%w: first field is tag %d, want 8", ErrMalformed, parsed[0].tag)
	}
	if parsed[0].val != BeginString {
		return nil, fmt.Errorf("%w: BeginString=%q, want %q", ErrMalformed, parsed[0].val, BeginString)
	}
	if parsed[1].tag != TagBodyLength {
		return nil, fmt.Errorf("%w: second field is tag %d, want 9", ErrMalformed, parsed[1].tag)
	}
	last := parsed[len(parsed)-1]
	if last.tag != TagCheckSum {
		return nil, fmt.Errorf("%w: last field is tag %d, want 10", ErrMalformed, last.tag)
	}

	// --- CheckSum: recompute over everything up to and including the SOH before
	// the 10= field, compare to the carried 3-digit value. ---
	csFieldStr := fmt.Sprintf("%d=%s%c", TagCheckSum, last.val, SOH)
	prefix := raw[:len(raw)-len(csFieldStr)]
	wantCS, err := strconv.Atoi(last.val)
	if err != nil || len(last.val) != 3 {
		return nil, fmt.Errorf("%w: bad checksum field %q", ErrMalformed, last.val)
	}
	if got := checksum(prefix); got != wantCS {
		return nil, fmt.Errorf("%w: got %03d, frame carried %03d", ErrChecksum, got, wantCS)
	}

	// --- BodyLength: the carried 9= value must equal the byte length of the body
	// (everything after the 9=...SOH up to and including the SOH before 10=). ---
	wantBL, err := strconv.Atoi(parsed[1].val)
	if err != nil {
		return nil, fmt.Errorf("%w: bad body length %q", ErrMalformed, parsed[1].val)
	}
	headStr := fmt.Sprintf("%d=%s%c%d=%s%c",
		TagBeginString, parsed[0].val, SOH, TagBodyLength, parsed[1].val, SOH)
	bodyLen := len(prefix) - len(headStr)
	if bodyLen != wantBL {
		return nil, fmt.Errorf("%w: header says %d, body is %d bytes", ErrBodyLength, wantBL, bodyLen)
	}

	// Build the semantic message: every field except the framing tags 8/9/10.
	m := &Message{index: make(map[int]string)}
	for _, p := range parsed {
		if p.tag == TagBeginString || p.tag == TagBodyLength || p.tag == TagCheckSum {
			continue
		}
		m.fields = append(m.fields, field{tag: p.tag, val: p.val})
		m.index[p.tag] = p.val
	}
	if m.MsgType() == "" {
		return nil, fmt.Errorf("%w: missing MsgType (tag 35)", ErrMalformed)
	}
	return m, nil
}
