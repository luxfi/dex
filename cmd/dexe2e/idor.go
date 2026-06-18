package main

import (
	"fmt"
	"math/big"

	"github.com/luxfi/geth/core/types"
)

// runIDOR proves, on the LIVE devnet via the EVM 0x9010 path, that the durable
// cancel-authorization binding (precompile RED H4 fix) does NOT open an IDOR: a
// caller can cancel ONLY its own resting order. A maker rests an ask; a DIFFERENT
// account (attacker) attempts to cancel it with the SAME tick band + salt; the
// attacker's cancel MUST revert and the maker's order MUST stay locked. Then the
// real maker cancels its own order successfully (unlocking the full remainder).
//
// This is the live counterpart to the precompile unit test
// TestZAPCancelAuthRejectsOthersAcrossMultiExec.
func runIDOR(bank *runner) {
	fmt.Println("================ LIVE IDOR CHECK (EVM 0x9010 cancel-auth) ================")

	makerKey, makerAddr := freshKey()
	attackerKey, attackerAddr := freshKey()
	maker := bank.childRunner(makerKey, makerAddr)
	attacker := bank.childRunner(attackerKey, attackerAddr)
	fmt.Printf("maker (fresh)    = %s\n", makerAddr.Hex())
	fmt.Printf("attacker (fresh) = %s\n", attackerAddr.Hex())

	// Fund both fresh accounts (gas + deposit headroom).
	fmt.Println("\n---- fund maker + attacker from treasury ----")
	bank.fundAccount(makerAddr, new(big.Int).Mul(big.NewInt(2), big.NewInt(1e18)))
	bank.fundAccount(attackerAddr, new(big.Int).Mul(big.NewInt(2), big.NewInt(1e18)))

	depositAmt := big.NewInt(10_000_000_000)

	// ensure market exists (idempotent).
	fmt.Println("\n---- ensure market ----")
	initData := append(sel(selInitialize), poolKeyABI()...)
	q96 := new(big.Int).Lsh(big.NewInt(1), 96)
	initData = append(initData, uint256ABI(q96)...)
	maker.send("initialize", initData, big.NewInt(0))

	// maker + attacker both deposit so neither is blocked by an empty ledger
	// (the attacker has funds — the only thing stopping it is authorization).
	fmt.Println("\n---- deposit (maker + attacker) ----")
	maker.deposit(depositAmt)
	attacker.deposit(depositAmt)

	// maker rests an ask.
	fmt.Println("\n---- maker rests an ask (available -> locked) ----")
	askTick := int64(6960)
	restSize := int64(1000)
	recRest, availDeltaRest := maker.modifyLiquidity("maker place", askTick, restSize)
	locked := new(big.Int).Neg(availDeltaRest)
	makerLockedAfterPlace := new(big.Int).Sub(depositAmt, maker.availOf(maker.from, nativeLUX))
	fmt.Printf("  maker locked = %s (availΔ=%s)\n", locked, availDeltaRest)
	placeOK := recRest.Status == types.ReceiptStatusSuccessful && locked.Sign() > 0

	// attacker tries to cancel the maker's order with the SAME tick band + salt.
	// The 0x9010 handle is keyed by the CALLER's address, so the attacker has no
	// durable binding -> the cancel MUST revert and the maker's lock MUST persist.
	fmt.Println("\n---- attacker attempts to cancel the maker's order (MUST revert) ----")
	recAtk, _ := attacker.modifyLiquidity("attacker cancel", askTick, -restSize)
	attackerReverted := recAtk.Status != types.ReceiptStatusSuccessful
	makerLockedAfterAttack := new(big.Int).Sub(depositAmt, maker.availOf(maker.from, nativeLUX))
	lockSurvived := makerLockedAfterAttack.Cmp(makerLockedAfterPlace) == 0 && makerLockedAfterAttack.Sign() > 0
	fmt.Printf("  attacker cancel reverted = %v ; maker still-locked = %s (was %s)\n",
		attackerReverted, makerLockedAfterAttack, makerLockedAfterPlace)

	// the real maker cancels its own order -> unlocks the full remainder.
	fmt.Println("\n---- maker cancels its OWN order (MUST succeed) ----")
	recOwn, unlocked := maker.modifyLiquidity("maker self-cancel", askTick, -restSize)
	ownOK := recOwn.Status == types.ReceiptStatusSuccessful && unlocked.Sign() > 0
	fmt.Printf("  maker self-cancel ok = %v ; unlocked = +%s\n", ownOK, unlocked)

	// drain back out so the run leaves no locked residual.
	fmt.Println("\n---- withdraw (cleanup) ----")
	maker.withdraw()
	attacker.withdraw()

	fmt.Println("\n================ IDOR RESULT ================")
	fmt.Printf("%-40s %v\n", "maker place locked funds", placeOK)
	fmt.Printf("%-40s %v\n", "attacker cancel REVERTED (IDOR closed)", attackerReverted)
	fmt.Printf("%-40s %v\n", "maker's lock survived the attack", lockSurvived)
	fmt.Printf("%-40s %v\n", "maker self-cancel succeeded + unlocked", ownOK)
	allOK := placeOK && attackerReverted && lockSurvived && ownOK
	if allOK {
		fmt.Println("\nPASS: a caller can cancel ONLY its own order (durable cancel-auth, RED H4).")
	} else {
		fmt.Println("\nFAIL: cancel-authorization invariant violated on the live path.")
	}
}
