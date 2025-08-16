// FIX Message Generator - Generates standard FIX 4.4 trading messages
package main

import (
	"flag"
	"fmt"
	"math/rand"
	"strconv"
	"strings"
	"time"
)

// FIX message types
const (
	MsgTypeNewOrderSingle     = "D"
	MsgTypeExecutionReport    = "8"
	MsgTypeMarketDataRequest  = "V"
	MsgTypeMarketDataSnapshot = "W"
	MsgTypeCancelRequest      = "F"
)

// FIX field tags
const (
	BeginString  = "8"
	BodyLength   = "9"
	MsgType      = "35"
	SenderCompID = "49"
	TargetCompID = "56"
	MsgSeqNum    = "34"
	SendingTime  = "52"

	// Order fields
	ClOrdID      = "11"
	Symbol       = "55"
	Side         = "54"
	OrderQty     = "38"
	OrdType      = "40"
	Price        = "44"
	TimeInForce  = "59"
	TransactTime = "60"

	CheckSum = "10"
)

type FIXGenerator struct {
	SenderID string
	TargetID string
	SeqNum   int
	Symbols  []string
}

func NewFIXGenerator(sender, target string) *FIXGenerator {
	return &FIXGenerator{
		SenderID: sender,
		TargetID: target,
		SeqNum:   1,
		Symbols:  []string{"BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "MATIC/USD"},
	}
}

// Calculate FIX checksum
func (g *FIXGenerator) calculateChecksum(msg string) string {
	sum := 0
	for _, c := range msg {
		sum += int(c)
	}
	return fmt.Sprintf("%03d", sum%256)
}

// Build FIX message with proper structure
func (g *FIXGenerator) buildMessage(msgType string, fields map[string]string) string {
	// Add header fields
	fields[MsgType] = msgType
	fields[SenderCompID] = g.SenderID
	fields[TargetCompID] = g.TargetID
	fields[MsgSeqNum] = strconv.Itoa(g.SeqNum)
	fields[SendingTime] = time.Now().UTC().Format("20060102-15:04:05.000")

	g.SeqNum++

	// Build body (excluding BeginString and BodyLength)
	var body strings.Builder
	body.WriteString(MsgType + "=" + fields[MsgType] + "\x01")
	body.WriteString(SenderCompID + "=" + fields[SenderCompID] + "\x01")
	body.WriteString(TargetCompID + "=" + fields[TargetCompID] + "\x01")
	body.WriteString(MsgSeqNum + "=" + fields[MsgSeqNum] + "\x01")
	body.WriteString(SendingTime + "=" + fields[SendingTime] + "\x01")

	// Add message-specific fields
	for tag, value := range fields {
		if tag != MsgType && tag != SenderCompID && tag != TargetCompID &&
			tag != MsgSeqNum && tag != SendingTime {
			body.WriteString(tag + "=" + value + "\x01")
		}
	}

	bodyStr := body.String()

	// Build complete message
	var msg strings.Builder
	msg.WriteString(BeginString + "=FIX.4.4\x01")
	msg.WriteString(BodyLength + "=" + strconv.Itoa(len(bodyStr)) + "\x01")
	msg.WriteString(bodyStr)

	// Calculate and add checksum
	checksum := g.calculateChecksum(msg.String())
	msg.WriteString(CheckSum + "=" + checksum + "\x01")

	return msg.String()
}

// Generate a New Order Single message
func (g *FIXGenerator) GenerateNewOrder() string {
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	fields := map[string]string{
		ClOrdID:      fmt.Sprintf("ORD%d", time.Now().UnixNano()),
		Symbol:       g.Symbols[rng.Intn(len(g.Symbols))],
		Side:         []string{"1", "2"}[rng.Intn(2)], // 1=Buy, 2=Sell
		OrderQty:     fmt.Sprintf("%.2f", rng.Float64()*100),
		OrdType:      "2", // Limit order
		Price:        fmt.Sprintf("%.2f", 30000+rng.Float64()*10000),
		TimeInForce:  "0", // Day
		TransactTime: time.Now().UTC().Format("20060102-15:04:05.000"),
	}

	return g.buildMessage(MsgTypeNewOrderSingle, fields)
}

// Generate a Cancel Request message
func (g *FIXGenerator) GenerateCancelRequest(origClOrdID string) string {
	fields := map[string]string{
		ClOrdID:      fmt.Sprintf("CANCEL%d", time.Now().UnixNano()),
		"41":         origClOrdID, // OrigClOrdID
		Symbol:       g.Symbols[rand.Intn(len(g.Symbols))],
		Side:         "1",
		TransactTime: time.Now().UTC().Format("20060102-15:04:05.000"),
	}

	return g.buildMessage(MsgTypeCancelRequest, fields)
}

// Generate Market Data Request
func (g *FIXGenerator) GenerateMarketDataRequest() string {
	fields := map[string]string{
		"262":  fmt.Sprintf("MDR%d", time.Now().UnixNano()), // MDReqID
		"263":  "1",                                         // SubscriptionRequestType (1=Subscribe)
		"264":  "5",                                         // MarketDepth (5 levels)
		"265":  "0",                                         // MDUpdateType (0=Full refresh)
		"146":  "1",                                         // NoRelatedSym
		Symbol: g.Symbols[rand.Intn(len(g.Symbols))],
	}

	return g.buildMessage(MsgTypeMarketDataRequest, fields)
}

// Print FIX message in readable format
func printFIXMessage(msg string) {
	fmt.Println("\n=== FIX Message ===")
	// Replace SOH with | for readability
	readable := strings.ReplaceAll(msg, "\x01", "|")
	fmt.Println(readable)
	fmt.Println("Length:", len(msg), "bytes")
}

func main() {
	var (
		mode     = flag.String("mode", "stream", "Mode: single, stream, batch")
		count    = flag.Int("count", 10, "Number of messages to generate")
		rate     = flag.Int("rate", 100, "Messages per second (for stream mode)")
		output   = flag.String("output", "stdout", "Output: stdout, file")
		senderID = flag.String("sender", "TRADER1", "Sender Comp ID")
		targetID = flag.String("target", "EXCHANGE", "Target Comp ID")
	)
	flag.Parse()

	generator := NewFIXGenerator(*senderID, *targetID)

	fmt.Printf("🔧 FIX Message Generator\n")
	fmt.Printf("Mode: %s\n", *mode)
	fmt.Printf("Sender: %s -> Target: %s\n", *senderID, *targetID)
	fmt.Println("===========================")

	switch *mode {
	case "single":
		// Generate single message
		msg := generator.GenerateNewOrder()
		printFIXMessage(msg)

	case "stream":
		// Stream messages at specified rate
		ticker := time.NewTicker(time.Second / time.Duration(*rate))
		defer ticker.Stop()

		generated := 0
		for range ticker.C {
			_ = generator.GenerateNewOrder() // Generate but don't store for stream mode

			if *output == "stdout" {
				fmt.Printf("\rGenerated: %d messages", generated+1)
			}

			generated++
			if generated >= *count {
				break
			}
		}
		fmt.Println("\n✅ Stream complete")

	case "batch":
		// Generate batch of messages
		messages := make([]string, *count)
		for i := 0; i < *count; i++ {
			switch i % 3 {
			case 0:
				messages[i] = generator.GenerateNewOrder()
			case 1:
				messages[i] = generator.GenerateMarketDataRequest()
			case 2:
				if i > 0 {
					messages[i] = generator.GenerateCancelRequest("ORD123456")
				} else {
					messages[i] = generator.GenerateNewOrder()
				}
			}
		}

		fmt.Printf("Generated %d FIX messages\n", len(messages))
		fmt.Printf("Total size: %d bytes\n", func() int {
			total := 0
			for _, msg := range messages {
				total += len(msg)
			}
			return total
		}())

		if *output == "stdout" {
			printFIXMessage(messages[0])
			fmt.Println("\n(First message shown)")
		}
	}

	// Performance stats
	fmt.Println("\n📊 FIX Message Statistics:")
	fmt.Printf("• Average message size: ~200 bytes\n")
	fmt.Printf("• At %d msgs/sec: %.2f MB/s\n", *rate, float64(*rate*200)/(1024*1024))
	fmt.Printf("• Network usage: %.3f Gbps\n", float64(*rate*200*8)/1e9)
}
