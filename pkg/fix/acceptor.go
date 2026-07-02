// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package fix

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/luxfi/dex/pkg/venue"
	"github.com/luxfi/dex/pkg/zapwire"
	"github.com/luxfi/log"
)

// acceptor.go is the FIX 4.4 acceptor: the TCP server side of the FIX trading
// path. It owns FIX SESSION semantics (Logon/Heartbeat/TestRequest/Logout,
// monotonic sequence numbers) and translates an application NewOrderSingle into a
// Venue call against THE matcher — it does not match. One Acceptor serves many
// sessions; each session is a goroutine reading SOH-framed messages off a TCP
// conn, validated by the message.go codec (BodyLength + CheckSum), and answered
// with spec-correct ExecutionReports.
//
// THE DECOMPLECT: this file is protocol + session only. Order routing is one call
// to Venue.Submit / Venue.Place / Venue.Cancel — the SAME operations ZAP, WS, and
// the 0x9010 precompile route. The matcher, the book, the custody ledger, and
// consensus all live behind Venue; FIX adds a wire dialect, nothing more.

// Config configures an Acceptor.
type Config struct {
	// Addr is the TCP listen address (":0" for an ephemeral test port).
	Addr string
	// Venue is the matcher seam every accepted order routes to.
	Venue venue.Venue
	// TargetCompID is this acceptor's identity (the client's TargetCompID, our
	// SenderCompID on outbound). A logon whose TargetCompID does not match is
	// rejected. Empty disables the check (single-tenant test/dev).
	TargetCompID string
	// HeartBtInt is the heartbeat interval advertised on Logon (seconds). 0 => 30.
	HeartBtInt int
	// Logger is the structured logger; nil => no-op.
	Logger log.Logger
}

// Acceptor is a FIX 4.4 acceptor listening on a TCP port. Start serving with
// Serve(ctx); Addr() reports the bound address (useful with ":0").
type Acceptor struct {
	cfg      Config
	ln       net.Listener
	log      log.Logger
	wg       sync.WaitGroup
	heartBt  time.Duration
	closing  chan struct{}
	closeOne sync.Once
}

// NewAcceptor binds the listener and returns an Acceptor. It does not start
// serving until Serve is called. Binding here (not in Serve) lets a caller read
// Addr() — the ephemeral port for ":0" — before accepting, which the e2e needs.
func NewAcceptor(cfg Config) (*Acceptor, error) {
	if cfg.Venue == nil {
		return nil, errors.New("fix: acceptor needs a Venue")
	}
	logger := cfg.Logger
	if logger == nil {
		logger = log.NewNoOpLogger()
	}
	hb := time.Duration(cfg.HeartBtInt) * time.Second
	if hb <= 0 {
		hb = 30 * time.Second
	}
	addr := cfg.Addr
	if addr == "" {
		addr = "127.0.0.1:0"
	}
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return nil, fmt.Errorf("fix: listen %s: %w", addr, err)
	}
	return &Acceptor{
		cfg:     cfg,
		ln:      ln,
		log:     logger,
		heartBt: hb,
		closing: make(chan struct{}),
	}, nil
}

// Addr returns the actual bound TCP address.
func (a *Acceptor) Addr() string { return a.ln.Addr().String() }

// Serve accepts connections until ctx is done or Close is called, running each
// session in its own goroutine. It blocks; run it in a goroutine. On ctx
// cancellation it stops accepting and waits for in-flight sessions to drain.
func (a *Acceptor) Serve(ctx context.Context) error {
	// Unblock Accept when ctx is cancelled by closing the listener.
	go func() {
		select {
		case <-ctx.Done():
			a.Close()
		case <-a.closing:
		}
	}()

	for {
		conn, err := a.ln.Accept()
		if err != nil {
			select {
			case <-a.closing:
				a.wg.Wait()
				return nil
			default:
				return fmt.Errorf("fix: accept: %w", err)
			}
		}
		a.wg.Add(1)
		go func() {
			defer a.wg.Done()
			s := newSession(conn, a.cfg, a.log, a.heartBt)
			s.serve(ctx)
		}()
	}
}

// Close stops accepting new connections and unblocks Serve. In-flight sessions
// finish their current message and exit. Idempotent.
func (a *Acceptor) Close() error {
	a.closeOne.Do(func() {
		close(a.closing)
		_ = a.ln.Close()
	})
	return nil
}

// session is one FIX session over a TCP conn. It tracks the inbound/outbound
// sequence numbers and the peer identity established at Logon.
type session struct {
	conn    net.Conn
	r       *bufio.Reader
	cfg     Config
	log     log.Logger
	heartBt time.Duration

	mu       sync.Mutex // guards writes to conn + outSeq
	outSeq   int        // next outbound MsgSeqNum
	inSeq    int        // next expected inbound MsgSeqNum
	loggedOn bool
	senderID string // our identity (their TargetCompID)
	targetID string // peer identity (their SenderCompID)
}

func newSession(conn net.Conn, cfg Config, logger log.Logger, hb time.Duration) *session {
	return &session{
		conn:    conn,
		r:       bufio.NewReader(conn),
		cfg:     cfg,
		log:     logger,
		heartBt: hb,
		outSeq:  1,
		inSeq:   1,
	}
}

// serve runs the read loop for one session until the peer disconnects, sends
// Logout, or ctx is cancelled. Every inbound frame is read by readMessage (which
// enforces the codec's BodyLength + CheckSum), then dispatched by MsgType.
func (s *session) serve(ctx context.Context) {
	defer s.conn.Close()

	for {
		select {
		case <-ctx.Done():
			s.sendLogout("server shutting down")
			return
		default:
		}

		// A generous read deadline: a healthy peer sends a heartbeat well within
		// it; silence past it ends the session. Reset on every successful read.
		_ = s.conn.SetReadDeadline(time.Now().Add(2*s.heartBt + 5*time.Second))

		msg, err := s.readMessage()
		if err != nil {
			if err == io.EOF {
				return // peer closed
			}
			if ne, ok := err.(net.Error); ok && ne.Timeout() {
				s.sendLogout("heartbeat timeout")
				return
			}
			// A malformed/integrity-failed frame: FIX Reject, then keep the session
			// (a single bad frame is recoverable). Reject carries the reason.
			s.sendSessionReject(0, err.Error())
			continue
		}

		if done := s.dispatch(ctx, msg); done {
			return
		}
	}
}

// readMessage reads one complete FIX frame off the wire. A FIX frame's length is
// self-describing: read 8=...SOH and 9=<n>SOH, then read exactly n body bytes,
// then the 10=xxxSOH trailer (always 7 bytes: "10=" + 3 digits + SOH). The whole
// frame is then handed to the strict codec, which re-validates BodyLength and
// CheckSum. This two-step (length-prefixed read + full re-validate) is the correct
// FIX framing — you cannot split on SOH alone because a body value could in
// principle contain data; BodyLength is authoritative for where the body ends.
func (s *session) readMessage() (*Message, error) {
	// 8=FIX.4.4<SOH>
	begin, err := s.readField()
	if err != nil {
		return nil, err
	}
	// 9=<bodyLen><SOH>
	bodyLenField, err := s.readField()
	if err != nil {
		return nil, err
	}
	// Parse the declared body length.
	eq := strings.IndexByte(bodyLenField, '=')
	if eq < 0 || bodyLenField[:eq] != strconv.Itoa(TagBodyLength) {
		return nil, fmt.Errorf("%w: expected BodyLength, got %q", ErrMalformed, bodyLenField)
	}
	n, err := strconv.Atoi(bodyLenField[eq+1 : len(bodyLenField)-1]) // strip trailing SOH
	if err != nil || n < 0 || n > maxBodyLen {
		return nil, fmt.Errorf("%w: bad BodyLength %q", ErrMalformed, bodyLenField)
	}
	// Read exactly n body bytes.
	body := make([]byte, n)
	if _, err := io.ReadFull(s.r, body); err != nil {
		return nil, err
	}
	// Read the 10=xxx<SOH> trailer: "10=" + 3 digits + SOH = 7 bytes.
	trailer := make([]byte, 7)
	if _, err := io.ReadFull(s.r, trailer); err != nil {
		return nil, err
	}
	// Reassemble the COMPLETE frame — 8=..SOH, 9=..SOH, body, 10=..SOH — and hand
	// it to the strict codec, which re-validates BodyLength and CheckSum end to
	// end. The 9= field must be included or Parse sees the body lead (35=) as the
	// second field.
	frame := make([]byte, 0, len(begin)+len(bodyLenField)+len(body)+len(trailer))
	frame = append(frame, begin...)
	frame = append(frame, bodyLenField...)
	frame = append(frame, body...)
	frame = append(frame, trailer...)
	return Parse(frame)
}

// readField reads bytes up to and including the next SOH and returns them as a
// string (including the trailing SOH). Used for the two length-prefix header
// fields (8 and 9) before the body length is known.
func (s *session) readField() (string, error) {
	b, err := s.r.ReadBytes(SOH)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// maxBodyLen caps a single FIX message body to a sane size, so a malicious or
// corrupt BodyLength cannot make us allocate unbounded memory.
const maxBodyLen = 1 << 20 // 1 MiB

// dispatch routes one validated message by MsgType. It returns true when the
// session should end (Logout received). Sequence-number gap handling: a message
// whose MsgSeqNum is below expected is a possible duplicate; above expected is a
// gap. For this acceptor (a fresh session per connection, no message store) we
// require strictly monotonic sequencing from Logon and Reject out-of-order app
// messages rather than silently accepting them.
func (s *session) dispatch(ctx context.Context, msg *Message) bool {
	mt := msg.MsgType()

	// Logon must be first and establishes identity + the inbound seq baseline.
	if !s.loggedOn {
		if mt != MsgTypeLogon {
			s.sendLogout("first message must be Logon (35=A)")
			return true
		}
		return s.handleLogon(msg)
	}

	// Sequence check for every post-Logon message.
	if seq, err := msg.GetInt(TagMsgSeqNum); err == nil {
		if seq < s.inSeq {
			// Below expected without PossDup => session-level error per FIX; we
			// reject the frame but keep the session.
			s.sendSessionReject(seq, fmt.Sprintf("MsgSeqNum too low: got %d, expected %d", seq, s.inSeq))
			return false
		}
		if seq > s.inSeq {
			s.sendSessionReject(seq, fmt.Sprintf("MsgSeqNum gap: got %d, expected %d", seq, s.inSeq))
			return false
		}
		s.inSeq = seq + 1
	}

	switch mt {
	case MsgTypeHeartbeat:
		return false // peer is alive; nothing to do
	case MsgTypeTestRequest:
		s.handleTestRequest(msg)
		return false
	case MsgTypeLogout:
		s.sendLogout("logout acknowledged")
		return true
	case MsgTypeNewOrderSingle:
		s.handleNewOrderSingle(ctx, msg)
		return false
	case MsgTypeOrderCancelReq:
		s.handleOrderCancelRequest(ctx, msg)
		return false
	default:
		s.sendSessionReject(s.inSeq-1, fmt.Sprintf("unsupported MsgType %q", mt))
		return false
	}
}

// handleLogon validates the Logon, establishes identity, and replies with a
// Logon (35=A) echoing the heartbeat interval. Returns true (end session) only on
// a fatal logon error already answered with Logout.
func (s *session) handleLogon(msg *Message) bool {
	sender, _ := msg.Get(TagSenderCompID) // peer's SenderCompID = our target
	target, _ := msg.Get(TagTargetCompID) // peer's TargetCompID = our identity

	if s.cfg.TargetCompID != "" && target != s.cfg.TargetCompID {
		s.sendLogout(fmt.Sprintf("unknown TargetCompID %q", target))
		return true
	}
	seq, err := msg.GetInt(TagMsgSeqNum)
	if err != nil {
		s.sendLogout("Logon missing MsgSeqNum")
		return true
	}

	s.targetID = sender
	s.senderID = target
	s.inSeq = seq + 1
	s.loggedOn = true

	hbInt := s.cfg.HeartBtInt
	if hbInt <= 0 {
		hbInt = int(s.heartBt / time.Second)
	}
	resp := NewMessage(MsgTypeLogon).
		SetInt(TagEncryptMethod, 0).
		SetInt(TagHeartBtInt, hbInt)
	s.send(resp)
	s.log.Debug("fix logon", "sender", s.targetID, "target", s.senderID, "hbInt", hbInt)
	return false
}

// handleTestRequest answers a TestRequest (35=1) with a Heartbeat (35=0) echoing
// the TestReqID (tag 112), the FIX liveness challenge-response.
func (s *session) handleTestRequest(msg *Message) {
	hb := NewMessage(MsgTypeHeartbeat)
	if id, ok := msg.Get(TagTestReqID); ok {
		hb.Set(TagTestReqID, id)
	}
	s.send(hb)
}

// handleNewOrderSingle translates a NewOrderSingle (35=D) into a Venue submit and
// emits ExecutionReport(s): an ack (ExecType=New) if nothing crossed and the
// order is fully open, then one report per fill (PartialFill/Fill), or a single
// Rejected report on a venue error. The order is submitted as a MARKETABLE order
// bounded by its limit price (IOC) — the faithful DEX taker primitive that
// crosses the resting book and returns fills, identical to the 0x9010 Swap path.
func (s *session) handleNewOrderSingle(ctx context.Context, msg *Message) {
	clOrdID, _ := msg.Get(TagClOrdID)
	symbol, ok := msg.Get(TagSymbol)
	if !ok || symbol == "" {
		s.sendOrderReject(clOrdID, "", "missing Symbol (tag 55)")
		return
	}
	sideStr, ok := msg.Get(TagSide)
	if !ok {
		s.sendOrderReject(clOrdID, symbol, "missing Side (tag 54)")
		return
	}
	var side uint8
	switch sideStr {
	case SideBuy:
		side = zapwire.SideBuy
	case SideSell:
		side = zapwire.SideSell
	default:
		s.sendOrderReject(clOrdID, symbol, fmt.Sprintf("bad Side %q", sideStr))
		return
	}
	qty, err := msg.GetFloat(TagOrderQty)
	if err != nil || qty <= 0 {
		s.sendOrderReject(clOrdID, symbol, "missing or non-positive OrderQty (tag 38)")
		return
	}
	ordType, _ := msg.Get(TagOrdType)
	isMarket := ordType == OrdTypeMarket

	var limit float64
	if !isMarket {
		limit, err = msg.GetFloat(TagPrice)
		if err != nil || limit <= 0 {
			s.sendOrderReject(clOrdID, symbol, "limit order missing or non-positive Price (tag 44)")
			return
		}
	}

	// The account whose ledger the order locks/settles: the session's peer
	// identity (SenderCompID), truncated to the venue's 16-byte user handle. An
	// explicit Account (tag 1) could override this; the session identity is the
	// authenticated default.
	user := venue.UserHandle(s.targetID)
	poolID := venue.SymbolToPoolID(symbol)

	fills, err := s.cfg.Venue.Submit(ctx, poolID, side, isMarket, limit, qty, user)
	if err != nil {
		s.sendOrderReject(clOrdID, symbol, fmt.Sprintf("venue submit: %v", err))
		return
	}

	orderID := clOrdID // exchange order id surfaced back; clOrdID is a stable handle
	if orderID == "" {
		orderID = strconv.FormatInt(time.Now().UnixNano(), 10)
	}

	// Aggregate fill stats for cumulative fields. Wire fills carry EXACT integers
	// (price = PriceInt ×1e8, size = base units); render human floats for FIX.
	fillPx := func(f zapwire.Fill) float64 { return float64(f.Price) / float64(zapwire.PriceScale) }
	fillSz := func(f zapwire.Fill) float64 { return float64(f.Size) }
	var cumQty, notional float64
	for _, f := range fills {
		cumQty += fillSz(f)
		notional += fillPx(f) * fillSz(f)
	}
	leaves := qty - cumQty
	if leaves < 0 {
		leaves = 0
	}

	if len(fills) == 0 {
		// Nothing crossed. A marketable order that does not cross is IOC-done with
		// zero fills; report it as Rejected? No — it is a valid order that simply
		// found no liquidity. Emit New then a Canceled (IOC expiry of the unfilled
		// remainder) so the client sees a terminal state without a phantom fill.
		s.sendExecReport(execReport{
			clOrdID: clOrdID, orderID: orderID, symbol: symbol, side: sideStr,
			execType: ExecTypeNew, ordStatus: OrdStatusNew,
			orderQty: qty, lastQty: 0, lastPx: 0, leavesQty: qty, cumQty: 0, avgPx: 0,
		})
		s.sendExecReport(execReport{
			clOrdID: clOrdID, orderID: orderID, symbol: symbol, side: sideStr,
			execType: ExecTypeCanceled, ordStatus: OrdStatusCanceled,
			orderQty: qty, lastQty: 0, lastPx: 0, leavesQty: 0, cumQty: 0, avgPx: 0,
			text: "IOC: no liquidity crossed",
		})
		return
	}

	// One ExecutionReport per fill. The last fill's report carries the terminal
	// status: Filled if fully filled, otherwise PartialFill (IOC remainder dropped,
	// which we still render as the final partial — leavesQty already reflects the
	// dropped remainder as 0 on a marketable IOC submit once the sweep is done).
	runningCum := 0.0
	runningNotional := 0.0
	for i, f := range fills {
		runningCum += fillSz(f)
		runningNotional += fillPx(f) * fillSz(f)
		avg := runningNotional / runningCum
		last := i == len(fills)-1
		execType := ExecTypePartialFill
		ordStatus := OrdStatusPartialFill
		thisLeaves := qty - runningCum
		if thisLeaves < 0 {
			thisLeaves = 0
		}
		if last {
			// On a marketable IOC submit the sweep is terminal: any remainder is
			// dropped, so the order is done. Fully filled => Filled; else the
			// remainder was IOC-cancelled but we mark the final fill Filled iff
			// cumQty==orderQty, otherwise PartialFill with leaves shown as dropped.
			if thisLeaves == 0 {
				execType = ExecTypeFill
				ordStatus = OrdStatusFilled
			}
		}
		s.sendExecReport(execReport{
			clOrdID: clOrdID, orderID: orderID, symbol: symbol, side: sideStr,
			execType: execType, ordStatus: ordStatus,
			orderQty: qty, lastQty: fillSz(f), lastPx: fillPx(f),
			leavesQty: thisLeaves, cumQty: runningCum, avgPx: avg,
		})
	}
}

// handleOrderCancelRequest translates an OrderCancelRequest (35=F) into a Venue
// cancel and replies with an ExecutionReport(Canceled) on success or an
// OrderCancelReject (35=9) on failure. The order is addressed by OrigClOrdID,
// which the client mapped to the venue order id it received at place time; the
// client passes the numeric venue order id as OrderID (tag 37).
func (s *session) handleOrderCancelRequest(ctx context.Context, msg *Message) {
	clOrdID, _ := msg.Get(TagClOrdID)
	origClOrdID, _ := msg.Get(TagOrigClOrdID)
	symbol, ok := msg.Get(TagSymbol)
	if !ok || symbol == "" {
		s.sendCancelReject(clOrdID, origClOrdID, "missing Symbol (tag 55)")
		return
	}
	orderIDStr, ok := msg.Get(TagOrderID)
	if !ok {
		s.sendCancelReject(clOrdID, origClOrdID, "missing OrderID (tag 37)")
		return
	}
	orderID, err := strconv.ParseUint(orderIDStr, 10, 64)
	if err != nil {
		s.sendCancelReject(clOrdID, origClOrdID, fmt.Sprintf("bad OrderID %q", orderIDStr))
		return
	}
	poolID := venue.SymbolToPoolID(symbol)
	if err := s.cfg.Venue.Cancel(ctx, poolID, orderID); err != nil {
		s.sendCancelReject(clOrdID, origClOrdID, fmt.Sprintf("venue cancel: %v", err))
		return
	}
	s.sendExecReport(execReport{
		clOrdID: clOrdID, orderID: orderIDStr, symbol: symbol,
		execType: ExecTypeCanceled, ordStatus: OrdStatusCanceled,
	})
}
