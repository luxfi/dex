package price

import "sync"

// lifecycle is the one start/stop mechanism for a polling price source.
//
// Every source here had written its own, and every one had the same hole:
// Close() closed the done channel and returned immediately, without waiting for
// the goroutine to notice. A polling loop selects on done against a ticker, and
// when both are ready Go picks a case AT RANDOM — so a poll could still be
// running, and still writing to the source's maps, after Close() had returned.
//
// TestXChainSourceStalePrice is what surfaced it: the test closes the source,
// backdates last["LUX-USDC"] by 10s, and asserts the price reads stale. An
// in-flight poll overwrote that timestamp with time.Now() and the assertion
// failed — on CI, where scheduling delays widen the window, not on a laptop.
//
// stop() joins before returning, so a closed source cannot mutate anything, and
// it is safe to call more than once (close of a closed channel panics).
type lifecycle struct {
	done chan struct{}
	wg   sync.WaitGroup
	once sync.Once
}

func newLifecycle() lifecycle { return lifecycle{done: make(chan struct{})} }

// run starts fn in a goroutine that stop() will wait for.
func (l *lifecycle) run(fn func()) {
	l.wg.Add(1)
	go func() {
		defer l.wg.Done()
		fn()
	}()
}

// stop signals every goroutine started by run and waits for them to return.
func (l *lifecycle) stop() {
	l.once.Do(func() { close(l.done) })
	l.wg.Wait()
}
