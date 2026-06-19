// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package fix

import (
	"bufio"
	"context"
	"io"
	"net"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/luxfi/dex/pkg/venue"
	"github.com/luxfi/dex/pkg/zapwire"
)

// fakeVenue is a deterministic in-memory Venue for unit-testing the acceptor in
// isolation from the matcher: it records calls and returns canned fills. The
// acceptor's job is protocol translation, so a fake here proves the FIX wire ↔
// Venue mapping without consensus. (The unified e2e proves the REAL matcher.)
type fakeVenue struct {
	mu          sync.Mutex
	ensured     [][32]byte
	placed      []placeCall
	submitted   []submitCall
	canceled    []cancelCall
	nextOrderID uint64
	fills       []zapwire.Fill // returned by Submit
	submitErr   error
	cancelErr   error
}

type placeCall struct {
	pool        [32]byte
	side        uint8
	price, size float64
	user        string
}
type submitCall struct {
	pool        [32]byte
	side        uint8
	isMarket    bool
	limit, size float64
	user        string
}
type cancelCall struct {
	pool    [32]byte
	orderID uint64
}

func (f *fakeVenue) EnsureMarket(_ context.Context, pool [32]byte) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.ensured = append(f.ensured, pool)
	return nil
}
func (f *fakeVenue) Place(_ context.Context, pool [32]byte, side uint8, price, size float64, user string) (uint64, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.placed = append(f.placed, placeCall{pool, side, price, size, user})
	f.nextOrderID++
	return f.nextOrderID, nil
}
func (f *fakeVenue) Cancel(_ context.Context, pool [32]byte, orderID uint64) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.canceled = append(f.canceled, cancelCall{pool, orderID})
	return f.cancelErr
}
func (f *fakeVenue) Submit(_ context.Context, pool [32]byte, side uint8, isMarket bool, limit, size float64, user string) ([]zapwire.Fill, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.submitted = append(f.submitted, submitCall{pool, side, isMarket, limit, size, user})
	if f.submitErr != nil {
		return nil, f.submitErr
	}
	return f.fills, nil
}

// testClient is a minimal FIX client for tests: it writes serialized messages
// with a monotonic outbound seq and reads framed ExecutionReports back. It uses
// the SAME codec the acceptor uses, so this also proves client↔server interop.
type testClient struct {
	conn   net.Conn
	r      *bufio.Reader
	sender string
	target string
	seq    int
}

func dialClient(t *testing.T, addr, sender, target string) *testClient {
	t.Helper()
	conn, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatalf("dial acceptor: %v", err)
	}
	return &testClient{conn: conn, r: bufio.NewReader(conn), sender: sender, target: target, seq: 1}
}

func (c *testClient) send(t *testing.T, m *Message) {
	t.Helper()
	m.Set(TagSenderCompID, c.sender).
		Set(TagTargetCompID, c.target).
		SetInt(TagMsgSeqNum, c.seq)
	c.seq++
	if _, err := c.conn.Write(m.Serialize()); err != nil {
		t.Fatalf("client write: %v", err)
	}
}

// recv reads one framed FIX message using the same length-prefixed framing the
// acceptor uses (8=..SOH, 9=<n>SOH, n body bytes, 7-byte 10= trailer).
func (c *testClient) recv(t *testing.T) *Message {
	t.Helper()
	_ = c.conn.SetReadDeadline(time.Now().Add(5 * time.Second))
	begin, err := c.r.ReadBytes(SOH)
	if err != nil {
		t.Fatalf("recv begin: %v", err)
	}
	blField, err := c.r.ReadBytes(SOH)
	if err != nil {
		t.Fatalf("recv bodylen: %v", err)
	}
	n, err := strconv.Atoi(string(blField[2 : len(blField)-1]))
	if err != nil {
		t.Fatalf("recv bad bodylen %q: %v", blField, err)
	}
	body := make([]byte, n)
	if _, err := io.ReadFull(c.r, body); err != nil {
		t.Fatalf("recv body: %v", err)
	}
	trailer := make([]byte, 7)
	if _, err := io.ReadFull(c.r, trailer); err != nil {
		t.Fatalf("recv trailer: %v", err)
	}
	frame := make([]byte, 0, len(begin)+len(blField)+len(body)+len(trailer))
	frame = append(frame, begin...)
	frame = append(frame, blField...)
	frame = append(frame, body...)
	frame = append(frame, trailer...)
	m, err := Parse(frame)
	if err != nil {
		t.Fatalf("recv parse: %v (frame=%q)", err, frame)
	}
	return m
}

func (c *testClient) close() { _ = c.conn.Close() }

// startAcceptor boots an acceptor on an ephemeral port with the given venue and
// returns it plus a teardown.
func startAcceptor(t *testing.T, v venue.Venue) *Acceptor {
	t.Helper()
	a, err := NewAcceptor(Config{Addr: "127.0.0.1:0", Venue: v, TargetCompID: "LXDEX", HeartBtInt: 1})
	if err != nil {
		t.Fatalf("NewAcceptor: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	go func() { _ = a.Serve(ctx) }()
	t.Cleanup(func() { cancel(); a.Close() })
	return a
}

// logon performs the Logon handshake and asserts a Logon reply.
func (c *testClient) logon(t *testing.T) {
	t.Helper()
	c.send(t, NewMessage(MsgTypeLogon).SetInt(TagEncryptMethod, 0).SetInt(TagHeartBtInt, 1))
	reply := c.recv(t)
	if reply.MsgType() != MsgTypeLogon {
		t.Fatalf("expected Logon reply, got %q", reply.MsgType())
	}
}

// TestAcceptorLogon proves a Logon establishes the session and a wrong
// TargetCompID is rejected with a Logout.
func TestAcceptorLogon(t *testing.T) {
	a := startAcceptor(t, &fakeVenue{})

	// Happy path.
	c := dialClient(t, a.Addr(), "CLIENT", "LXDEX")
	defer c.close()
	c.logon(t)

	// Wrong target => Logout.
	bad := dialClient(t, a.Addr(), "CLIENT", "WRONG")
	defer bad.close()
	bad.send(t, NewMessage(MsgTypeLogon).SetInt(TagHeartBtInt, 1))
	reply := bad.recv(t)
	if reply.MsgType() != MsgTypeLogout {
		t.Fatalf("wrong target must Logout, got %q", reply.MsgType())
	}
}

// TestAcceptorNewOrderFills proves a NewOrderSingle routes to Venue.Submit and the
// canned fills come back as ExecutionReports with correct ExecType/OrdStatus and
// quantity/price fields — the FIX fill path.
func TestAcceptorNewOrderFills(t *testing.T) {
	v := &fakeVenue{fills: []zapwire.Fill{
		{Price: 100, Size: 10, TakerSide: zapwire.SideBuy},
		{Price: 101, Size: 2, TakerSide: zapwire.SideBuy},
	}}
	a := startAcceptor(t, v)
	c := dialClient(t, a.Addr(), "CLIENT", "LXDEX")
	defer c.close()
	c.logon(t)

	c.send(t, NewMessage(MsgTypeNewOrderSingle).
		Set(TagClOrdID, "ORD1").
		Set(TagSymbol, "LUX/LUSD").
		Set(TagSide, SideBuy).
		Set(TagOrderQty, "12").
		Set(TagOrdType, OrdTypeLimit).
		Set(TagPrice, "101"))

	// Two fills => two ExecutionReports; the last is a full Fill (12 of 12).
	r1 := c.recv(t)
	assertExec(t, r1, ExecTypePartialFill, OrdStatusPartialFill, "10", "100")
	r2 := c.recv(t)
	assertExec(t, r2, ExecTypeFill, OrdStatusFilled, "2", "101")

	// And the venue saw exactly one marketable submit for the right market/side.
	v.mu.Lock()
	defer v.mu.Unlock()
	if len(v.submitted) != 1 {
		t.Fatalf("venue submits = %d, want 1", len(v.submitted))
	}
	sc := v.submitted[0]
	if sc.side != zapwire.SideBuy || sc.size != 12 || sc.limit != 101 || sc.isMarket {
		t.Fatalf("submit call wrong: %+v", sc)
	}
	if sc.pool != venue.SymbolToPoolID("LUX/LUSD") {
		t.Fatalf("submit pool != venue.SymbolToPoolID(LUX/LUSD)")
	}
	if sc.user != "CLIENT" {
		t.Fatalf("submit user = %q, want CLIENT (session identity)", sc.user)
	}
}

// TestAcceptorNewOrderNoLiquidity proves a marketable order that crosses nothing
// gets New then Canceled (IOC), never a phantom fill.
func TestAcceptorNewOrderNoLiquidity(t *testing.T) {
	a := startAcceptor(t, &fakeVenue{fills: nil})
	c := dialClient(t, a.Addr(), "CLIENT", "LXDEX")
	defer c.close()
	c.logon(t)

	c.send(t, NewMessage(MsgTypeNewOrderSingle).
		Set(TagClOrdID, "ORDX").Set(TagSymbol, "LUX/LUSD").
		Set(TagSide, SideSell).Set(TagOrderQty, "5").
		Set(TagOrdType, OrdTypeLimit).Set(TagPrice, "200"))

	r1 := c.recv(t)
	if r1.MsgType() != MsgTypeExecutionReport {
		t.Fatalf("want ExecutionReport, got %q", r1.MsgType())
	}
	if et, _ := r1.Get(TagExecType); et != ExecTypeNew {
		t.Fatalf("first report ExecType=%q want New", et)
	}
	r2 := c.recv(t)
	if et, _ := r2.Get(TagExecType); et != ExecTypeCanceled {
		t.Fatalf("second report ExecType=%q want Canceled (IOC)", et)
	}
}

// TestAcceptorCancel proves an OrderCancelRequest routes to Venue.Cancel and gets
// an ExecutionReport(Canceled); a venue cancel failure gets OrderCancelReject.
func TestAcceptorCancel(t *testing.T) {
	v := &fakeVenue{}
	a := startAcceptor(t, v)
	c := dialClient(t, a.Addr(), "CLIENT", "LXDEX")
	defer c.close()
	c.logon(t)

	c.send(t, NewMessage(MsgTypeOrderCancelReq).
		Set(TagClOrdID, "CXL1").Set(TagOrigClOrdID, "ORD1").
		Set(TagSymbol, "LUX/LUSD").Set(TagOrderID, "7"))
	r := c.recv(t)
	if r.MsgType() != MsgTypeExecutionReport {
		t.Fatalf("want ExecutionReport, got %q", r.MsgType())
	}
	if st, _ := r.Get(TagOrdStatus); st != OrdStatusCanceled {
		t.Fatalf("OrdStatus=%q want Canceled", st)
	}
	v.mu.Lock()
	if len(v.canceled) != 1 || v.canceled[0].orderID != 7 {
		t.Fatalf("venue cancel calls = %+v, want one for order 7", v.canceled)
	}
	v.mu.Unlock()

	// Now a failing cancel => OrderCancelReject.
	v.cancelErr = venue.ErrRejected
	c.send(t, NewMessage(MsgTypeOrderCancelReq).
		Set(TagClOrdID, "CXL2").Set(TagOrigClOrdID, "ORD9").
		Set(TagSymbol, "LUX/LUSD").Set(TagOrderID, "9"))
	rr := c.recv(t)
	if rr.MsgType() != MsgTypeOrderCancelReject {
		t.Fatalf("failing cancel must be OrderCancelReject, got %q", rr.MsgType())
	}
}

// TestAcceptorTestRequest proves a TestRequest gets a Heartbeat echoing TestReqID.
func TestAcceptorTestRequest(t *testing.T) {
	a := startAcceptor(t, &fakeVenue{})
	c := dialClient(t, a.Addr(), "CLIENT", "LXDEX")
	defer c.close()
	c.logon(t)

	c.send(t, NewMessage(MsgTypeTestRequest).Set(TagTestReqID, "PING-1"))
	r := c.recv(t)
	if r.MsgType() != MsgTypeHeartbeat {
		t.Fatalf("want Heartbeat, got %q", r.MsgType())
	}
	if id, _ := r.Get(TagTestReqID); id != "PING-1" {
		t.Fatalf("Heartbeat TestReqID=%q want PING-1", id)
	}
}

// TestAcceptorRejectsOrderBeforeLogon proves the session refuses application
// traffic until Logon.
func TestAcceptorRejectsOrderBeforeLogon(t *testing.T) {
	a := startAcceptor(t, &fakeVenue{})
	c := dialClient(t, a.Addr(), "CLIENT", "LXDEX")
	defer c.close()

	c.send(t, NewMessage(MsgTypeNewOrderSingle).
		Set(TagClOrdID, "EARLY").Set(TagSymbol, "LUX/LUSD").
		Set(TagSide, SideBuy).Set(TagOrderQty, "1").
		Set(TagOrdType, OrdTypeLimit).Set(TagPrice, "1"))
	r := c.recv(t)
	if r.MsgType() != MsgTypeLogout {
		t.Fatalf("pre-logon order must be rejected with Logout, got %q", r.MsgType())
	}
}

// TestAcceptorLogout proves a Logout is acknowledged and ends the session.
func TestAcceptorLogout(t *testing.T) {
	a := startAcceptor(t, &fakeVenue{})
	c := dialClient(t, a.Addr(), "CLIENT", "LXDEX")
	defer c.close()
	c.logon(t)

	c.send(t, NewMessage(MsgTypeLogout))
	r := c.recv(t)
	if r.MsgType() != MsgTypeLogout {
		t.Fatalf("want Logout ack, got %q", r.MsgType())
	}
}

// assertExec asserts an ExecutionReport's ExecType/OrdStatus and (when given) the
// LastQty/LastPx of a fill.
func assertExec(t *testing.T, m *Message, execType, ordStatus, lastQty, lastPx string) {
	t.Helper()
	if m.MsgType() != MsgTypeExecutionReport {
		t.Fatalf("MsgType=%q want ExecutionReport(8)", m.MsgType())
	}
	if et, _ := m.Get(TagExecType); et != execType {
		t.Errorf("ExecType=%q want %q", et, execType)
	}
	if st, _ := m.Get(TagOrdStatus); st != ordStatus {
		t.Errorf("OrdStatus=%q want %q", st, ordStatus)
	}
	if lastQty != "" {
		if q, _ := m.Get(TagLastQty); q != lastQty {
			t.Errorf("LastQty=%q want %q", q, lastQty)
		}
		if p, _ := m.Get(TagLastPx); p != lastPx {
			t.Errorf("LastPx=%q want %q", p, lastPx)
		}
	}
}
