// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package fix

import (
	"strconv"
	"time"
)

// send.go is the session's outbound side: it stamps the standard FIX header
// (SenderCompID/TargetCompID/MsgSeqNum/SendingTime), serializes via the codec
// (which appends BodyLength + CheckSum), and writes the frame to the conn under
// the session mutex so outbound sequence numbers stay monotonic and writes never
// interleave across goroutines.

// sendingTimeFormat is the FIX UTCTimestamp format (millisecond precision).
const sendingTimeFormat = "20060102-15:04:05.000"

// send stamps the standard header onto msg, serializes it, and writes it to the
// peer. It assigns the next outbound MsgSeqNum and SendingTime. Safe for
// concurrent callers (the heartbeat goroutine and the read loop both send).
func (s *session) send(msg *Message) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Standard header: identity + sequence + time. These precede the body fields
	// the caller already Set; FIX requires 49/56/34/52 in the header, but since
	// BodyLength is computed over the whole body the exact ordering among header
	// and body app-fields is not integrity-relevant — we prepend them logically by
	// setting before serialize. (MsgType stays first via Serialize.)
	msg.Set(TagSenderCompID, s.senderID).
		Set(TagTargetCompID, s.targetID).
		SetInt(TagMsgSeqNum, s.outSeq).
		Set(TagSendingTime, time.Now().UTC().Format(sendingTimeFormat))
	s.outSeq++

	_ = s.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
	if _, err := s.conn.Write(msg.Serialize()); err != nil {
		s.log.Debug("fix write failed", "err", err)
	}
}

// sendLogout emits a Logout (35=5) with an optional reason in Text (58).
func (s *session) sendLogout(reason string) {
	m := NewMessage(MsgTypeLogout)
	if reason != "" {
		m.Set(TagText, reason)
	}
	// A pre-logon Logout has no established identity; send still stamps empty
	// 49/56, which is acceptable for a rejection handshake.
	s.send(m)
}

// sendSessionReject emits a session-level Reject (35=3) referencing the offending
// inbound sequence number with a reason in Text. Used for malformed frames and
// sequencing errors.
func (s *session) sendSessionReject(refSeq int, reason string) {
	m := NewMessage(MsgTypeReject)
	if refSeq > 0 {
		m.SetInt(45, refSeq) // RefSeqNum
	}
	m.Set(TagText, reason)
	s.send(m)
}

// sendCancelReject emits an OrderCancelReject (35=9) for a failed cancel.
func (s *session) sendCancelReject(clOrdID, origClOrdID, reason string) {
	m := NewMessage(MsgTypeOrderCancelReject)
	if clOrdID != "" {
		m.Set(TagClOrdID, clOrdID)
	}
	if origClOrdID != "" {
		m.Set(TagOrigClOrdID, origClOrdID)
	}
	m.Set(TagOrdStatus, OrdStatusRejected).
		Set(434, "1"). // CxlRejResponseTo = OrderCancelRequest
		Set(TagText, reason)
	s.send(m)
}

// sendOrderReject emits an ExecutionReport(Rejected) (35=8, 150=8, 39=8) for an
// order the gateway or venue refused.
func (s *session) sendOrderReject(clOrdID, symbol, reason string) {
	s.sendExecReport(execReport{
		clOrdID:   clOrdID,
		symbol:    symbol,
		execType:  ExecTypeRejected,
		ordStatus: OrdStatusRejected,
		text:      reason,
	})
}

// execReport is the data for one ExecutionReport (35=8). Zero-valued numeric
// fields are still emitted where FIX expects them (e.g. LeavesQty on an ack).
type execReport struct {
	clOrdID   string
	orderID   string
	symbol    string
	side      string // tag 54 value ("1"/"2"); empty omits it (e.g. a bare reject)
	execType  string // tag 150
	ordStatus string // tag 39
	orderQty  float64
	lastQty   float64
	lastPx    float64
	leavesQty float64
	cumQty    float64
	avgPx     float64
	text      string
}

// sendExecReport renders and sends an ExecutionReport. It always carries an
// ExecID (a fresh per-report id), the OrderID/ClOrdID linkage, ExecType (150),
// OrdStatus (39), and the quantity/price fields. This is the spec-correct shape a
// FIX client decodes to learn an ack, a fill, a cancel, or a reject.
func (s *session) sendExecReport(r execReport) {
	m := NewMessage(MsgTypeExecutionReport)
	if r.orderID != "" {
		m.Set(TagOrderID, r.orderID)
	}
	if r.clOrdID != "" {
		m.Set(TagClOrdID, r.clOrdID)
	}
	m.Set(TagExecID, strconv.FormatInt(time.Now().UnixNano(), 10)).
		Set(TagExecType, r.execType).
		Set(TagOrdStatus, r.ordStatus)
	if r.symbol != "" {
		m.Set(TagSymbol, r.symbol)
	}
	if r.side != "" {
		m.Set(TagSide, r.side)
	}
	if r.orderQty > 0 {
		m.Set(TagOrderQty, formatQty(r.orderQty))
	}
	if r.lastQty > 0 {
		m.Set(TagLastQty, formatQty(r.lastQty))
		m.Set(TagLastPx, formatPx(r.lastPx))
	}
	m.Set(TagLeavesQty, formatQty(r.leavesQty)).
		Set(TagCumQty, formatQty(r.cumQty))
	if r.avgPx > 0 {
		m.Set(TagAvgPx, formatPx(r.avgPx))
	}
	if r.text != "" {
		m.Set(TagText, r.text)
	}
	s.send(m)
}

// formatQty / formatPx render numeric quantities/prices with full float64
// precision (strconv 'g' with -1 precision is round-trippable). FIX carries these
// as decimal strings; the venue's authoritative units are integer (the custody
// ledger), and the float fill price/size are exact for the test grid.
func formatQty(v float64) string { return strconv.FormatFloat(v, 'f', -1, 64) }
func formatPx(v float64) string  { return strconv.FormatFloat(v, 'f', -1, 64) }
