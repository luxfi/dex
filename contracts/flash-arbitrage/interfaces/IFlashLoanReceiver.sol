// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/**
 * @title IFlashLoanReceiver
 * @notice Interface for contracts that receive flash loans
 * @dev Implement this to create arbitrage strategies
 */
interface IFlashLoanReceiver {
    /**
     * @notice Called by FlashLoanPool after transferring funds
     * @param asset The token borrowed
     * @param amount The amount borrowed
     * @param fee The fee to pay (amount * 0.09%)
     * @param initiator The address that initiated the flash loan
     * @param params Arbitrary data passed from the initiator
     * @return success Must return true if execution succeeded
     *
     * @dev Your contract must:
     * 1. Receive `amount` of `asset`
     * 2. Execute your arbitrage logic
     * 3. Ensure you have `amount + fee` by end of function
     * 4. Approve FlashLoanPool to pull `amount + fee`
     * 5. Return true
     *
     * If anything fails, revert and the entire transaction rolls back (zero risk)
     */
    function executeOperation(
        address asset,
        uint256 amount,
        uint256 fee,
        address initiator,
        bytes calldata params
    ) external returns (bool success);

    /**
     * @notice Called by FlashLoanPool for multi-asset flash loans
     * @param assets Array of tokens borrowed
     * @param amounts Array of amounts borrowed
     * @param fees Array of fees to pay
     * @param initiator The address that initiated the flash loan
     * @param params Arbitrary data passed from the initiator
     */
    function executeOperationMulti(
        address[] calldata assets,
        uint256[] calldata amounts,
        uint256[] calldata fees,
        address initiator,
        bytes calldata params
    ) external returns (bool success);
}
