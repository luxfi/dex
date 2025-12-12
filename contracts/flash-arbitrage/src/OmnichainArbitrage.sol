// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "../interfaces/IFlashLoanReceiver.sol";
import "../interfaces/IERC20.sol";
import "../interfaces/IWarpMessenger.sol";

/**
 * @title OmnichainArbitrage
 * @notice Execute arbitrage across multiple chains using Lux as settlement layer
 * @dev Integrates with flash loans for capital-efficient arbitrage
 *
 * ARBITRAGE FLOW:
 * 1. Detect price discrepancy (off-chain scanner)
 * 2. Borrow via flash loan (zero collateral)
 * 3. Execute trades across chains via Warp
 * 4. Settle profits back to Lux
 * 5. Repay flash loan + fee
 * 6. Keep profit
 *
 * SUPPORTED STRATEGIES:
 * - Simple: Buy low on Chain A, sell high on Chain B
 * - Triangular: A->B->C->A with net profit
 * - Multi-hop: Complex routes through multiple DEXs/chains
 */
contract OmnichainArbitrage is IFlashLoanReceiver {
    // ============ Constants ============
    uint256 public constant MIN_PROFIT_BPS = 10; // 0.1% minimum profit after fees
    uint256 public constant MAX_SLIPPAGE_BPS = 50; // 0.5% max slippage

    // ============ State ============
    address public immutable flashLoanPool;
    IWarpMessenger public immutable warpMessenger;
    bytes32 public immutable luxChainID;

    // DEX routers on various chains
    mapping(bytes32 => address) public dexRouters; // chainID => router
    mapping(bytes32 => address) public bridgeContracts; // chainID => bridge

    // Supported trading pairs
    mapping(bytes32 => bool) public supportedPairs; // keccak256(tokenA, tokenB) => supported

    // Execution tracking
    mapping(bytes32 => ArbExecution) public executions;
    uint256 public totalProfitGenerated;
    uint256 public totalArbitrages;

    // Access control
    address public owner;
    mapping(address => bool) public authorizedExecutors;

    // ============ Structs ============
    struct ArbExecution {
        bytes32 executionId;
        address initiator;
        uint256 borrowedAmount;
        uint256 profit;
        uint256 timestamp;
        ArbStatus status;
    }

    struct ArbParams {
        ArbType arbType;
        Route[] routes;
        uint256 minProfitBps;
        uint256 maxSlippageBps;
        uint256 deadline;
    }

    struct Route {
        bytes32 chainID;
        address dex;
        address tokenIn;
        address tokenOut;
        uint256 amountIn;
        uint256 minAmountOut;
        bytes swapData;
    }

    enum ArbType {
        SIMPLE,           // A->B single swap
        TRIANGULAR,       // A->B->C->A
        MULTI_HOP,        // Complex multi-chain route
        FLASH_SWAP        // DEX flash swap based
    }

    enum ArbStatus {
        PENDING,
        EXECUTING,
        COMPLETED,
        FAILED
    }

    // ============ Events ============
    event ArbitrageExecuted(
        bytes32 indexed executionId,
        address indexed executor,
        uint256 borrowed,
        uint256 profit,
        ArbType arbType
    );

    event CrossChainSwapInitiated(
        bytes32 indexed executionId,
        bytes32 indexed targetChain,
        address tokenIn,
        uint256 amountIn
    );

    event ProfitSettled(
        bytes32 indexed executionId,
        uint256 grossProfit,
        uint256 netProfit
    );

    // ============ Errors ============
    error UnauthorizedExecutor();
    error InsufficientProfit();
    error SlippageExceeded();
    error DeadlineExpired();
    error InvalidRoute();
    error ExecutionFailed();
    error UnsupportedChain();

    // ============ Modifiers ============
    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    modifier onlyAuthorized() {
        if (!authorizedExecutors[msg.sender] && msg.sender != owner) {
            revert UnauthorizedExecutor();
        }
        _;
    }

    modifier onlyFlashLoanPool() {
        require(msg.sender == flashLoanPool, "Only flash loan pool");
        _;
    }

    // ============ Constructor ============
    constructor(address _flashLoanPool, address _warpMessenger) {
        flashLoanPool = _flashLoanPool;
        warpMessenger = IWarpMessenger(_warpMessenger);
        luxChainID = warpMessenger.getBlockchainID();
        owner = msg.sender;
        authorizedExecutors[msg.sender] = true;
    }

    // ============ Flash Loan Callbacks ============

    /**
     * @notice Called by flash loan pool - execute the arbitrage
     */
    function executeOperation(
        address asset,
        uint256 amount,
        uint256 fee,
        address initiator,
        bytes calldata params
    ) external override onlyFlashLoanPool returns (bool) {
        ArbParams memory arbParams = abi.decode(params, (ArbParams));

        // Validate deadline
        if (block.timestamp > arbParams.deadline) revert DeadlineExpired();

        // Generate execution ID
        bytes32 executionId = keccak256(
            abi.encodePacked(block.timestamp, initiator, asset, amount)
        );

        executions[executionId] = ArbExecution({
            executionId: executionId,
            initiator: initiator,
            borrowedAmount: amount,
            profit: 0,
            timestamp: block.timestamp,
            status: ArbStatus.EXECUTING
        });

        // Execute based on arbitrage type
        uint256 profit;
        if (arbParams.arbType == ArbType.SIMPLE) {
            profit = _executeSimpleArb(asset, amount, arbParams);
        } else if (arbParams.arbType == ArbType.TRIANGULAR) {
            profit = _executeTriangularArb(asset, amount, arbParams);
        } else if (arbParams.arbType == ArbType.MULTI_HOP) {
            profit = _executeMultiHopArb(asset, amount, arbParams);
        } else {
            revert InvalidRoute();
        }

        // Verify minimum profit
        uint256 requiredProfit = (amount * arbParams.minProfitBps) / 10000;
        if (profit < requiredProfit + fee) revert InsufficientProfit();

        // Approve repayment
        uint256 amountToRepay = amount + fee;
        IERC20(asset).approve(flashLoanPool, amountToRepay);

        // Update execution record
        executions[executionId].profit = profit - fee;
        executions[executionId].status = ArbStatus.COMPLETED;
        totalProfitGenerated += profit - fee;
        totalArbitrages++;

        emit ArbitrageExecuted(executionId, initiator, amount, profit - fee, arbParams.arbType);

        return true;
    }

    /**
     * @notice Multi-asset flash loan callback
     */
    function executeOperationMulti(
        address[] calldata assets,
        uint256[] calldata amounts,
        uint256[] calldata fees,
        address initiator,
        bytes calldata params
    ) external override onlyFlashLoanPool returns (bool) {
        ArbParams memory arbParams = abi.decode(params, (ArbParams));

        if (block.timestamp > arbParams.deadline) revert DeadlineExpired();

        // Execute multi-asset arbitrage
        uint256 totalProfit = _executeMultiAssetArb(assets, amounts, arbParams);

        // Calculate total fees
        uint256 totalFees;
        for (uint256 i = 0; i < fees.length; i++) {
            totalFees += fees[i];
        }

        if (totalProfit < totalFees) revert InsufficientProfit();

        // Approve all repayments
        for (uint256 i = 0; i < assets.length; i++) {
            IERC20(assets[i]).approve(flashLoanPool, amounts[i] + fees[i]);
        }

        totalProfitGenerated += totalProfit - totalFees;
        totalArbitrages++;

        return true;
    }

    // ============ Arbitrage Execution ============

    /**
     * @notice Simple arbitrage: buy on one chain, sell on another
     */
    function _executeSimpleArb(
        address asset,
        uint256 amount,
        ArbParams memory params
    ) internal returns (uint256 profit) {
        require(params.routes.length == 2, "Simple arb needs 2 routes");

        Route memory buyRoute = params.routes[0];
        Route memory sellRoute = params.routes[1];

        // Execute buy
        uint256 bought = _executeSwap(buyRoute, amount);

        // Bridge to sell chain if different
        if (buyRoute.chainID != sellRoute.chainID) {
            bought = _bridgeAsset(
                buyRoute.chainID,
                sellRoute.chainID,
                buyRoute.tokenOut,
                bought
            );
        }

        // Execute sell
        uint256 received = _executeSwap(sellRoute, bought);

        // Calculate profit
        profit = received > amount ? received - amount : 0;

        return profit;
    }

    /**
     * @notice Triangular arbitrage: A -> B -> C -> A
     */
    function _executeTriangularArb(
        address asset,
        uint256 amount,
        ArbParams memory params
    ) internal returns (uint256 profit) {
        require(params.routes.length == 3, "Triangular arb needs 3 routes");

        uint256 current = amount;

        // Execute each leg
        for (uint256 i = 0; i < 3; i++) {
            Route memory route = params.routes[i];

            // Bridge if needed
            if (i > 0 && params.routes[i-1].chainID != route.chainID) {
                current = _bridgeAsset(
                    params.routes[i-1].chainID,
                    route.chainID,
                    params.routes[i-1].tokenOut,
                    current
                );
            }

            current = _executeSwap(route, current);
        }

        // Calculate profit (should end up with more of starting asset)
        profit = current > amount ? current - amount : 0;

        return profit;
    }

    /**
     * @notice Multi-hop arbitrage across many chains/DEXs
     */
    function _executeMultiHopArb(
        address asset,
        uint256 amount,
        ArbParams memory params
    ) internal returns (uint256 profit) {
        uint256 current = amount;
        bytes32 currentChain = luxChainID;

        for (uint256 i = 0; i < params.routes.length; i++) {
            Route memory route = params.routes[i];

            // Bridge if changing chains
            if (currentChain != route.chainID) {
                current = _bridgeAsset(
                    currentChain,
                    route.chainID,
                    i == 0 ? asset : params.routes[i-1].tokenOut,
                    current
                );
                currentChain = route.chainID;
            }

            current = _executeSwap(route, current);
        }

        // Bridge back to Lux if needed
        if (currentChain != luxChainID) {
            current = _bridgeAsset(
                currentChain,
                luxChainID,
                params.routes[params.routes.length - 1].tokenOut,
                current
            );
        }

        profit = current > amount ? current - amount : 0;

        return profit;
    }

    /**
     * @notice Multi-asset arbitrage for complex strategies
     */
    function _executeMultiAssetArb(
        address[] calldata assets,
        uint256[] calldata amounts,
        ArbParams memory params
    ) internal returns (uint256 totalProfit) {
        // Complex multi-asset arbitrage logic
        // This could involve:
        // - Parallel swaps across chains
        // - Liquidity provision/removal
        // - Yield optimization

        for (uint256 i = 0; i < params.routes.length; i++) {
            Route memory route = params.routes[i];
            uint256 amountIn = route.amountIn;

            uint256 received = _executeSwap(route, amountIn);

            if (received > route.minAmountOut) {
                totalProfit += received - route.minAmountOut;
            }
        }

        return totalProfit;
    }

    // ============ Swap Execution ============

    /**
     * @notice Execute a swap on a DEX
     */
    function _executeSwap(Route memory route, uint256 amountIn) internal returns (uint256 amountOut) {
        if (route.chainID == luxChainID) {
            // Local Lux swap
            amountOut = _executeLocalSwap(route, amountIn);
        } else {
            // Cross-chain swap via Warp
            amountOut = _executeCrossChainSwap(route, amountIn);
        }

        // Verify slippage
        if (amountOut < route.minAmountOut) revert SlippageExceeded();

        return amountOut;
    }

    /**
     * @notice Execute swap on Lux DEX
     */
    function _executeLocalSwap(Route memory route, uint256 amountIn) internal returns (uint256) {
        // Approve DEX router
        IERC20(route.tokenIn).approve(route.dex, amountIn);

        // Execute swap via router
        // This would call the actual DEX (LX DEX, AMM, etc.)
        (bool success, bytes memory result) = route.dex.call(route.swapData);
        require(success, "Swap failed");

        return abi.decode(result, (uint256));
    }

    /**
     * @notice Execute swap on remote chain via Warp
     */
    function _executeCrossChainSwap(Route memory route, uint256 amountIn) internal returns (uint256) {
        // Encode cross-chain swap message
        bytes memory message = abi.encode(
            CrossChainSwapMessage({
                targetDex: route.dex,
                tokenIn: route.tokenIn,
                tokenOut: route.tokenOut,
                amountIn: amountIn,
                minAmountOut: route.minAmountOut,
                swapData: route.swapData,
                returnChain: luxChainID,
                returnAddress: address(this)
            })
        );

        // Send via Warp
        bytes32 messageId = warpMessenger.sendWarpMessage(message);

        emit CrossChainSwapInitiated(
            messageId,
            route.chainID,
            route.tokenIn,
            amountIn
        );

        // In practice, this would wait for Warp confirmation
        // For atomic execution, we use Warp's synchronous mode
        return route.minAmountOut; // Placeholder - actual impl uses Warp callbacks
    }

    /**
     * @notice Bridge asset between chains
     */
    function _bridgeAsset(
        bytes32 sourceChain,
        bytes32 destChain,
        address token,
        uint256 amount
    ) internal returns (uint256) {
        address bridge = bridgeContracts[destChain];
        if (bridge == address(0)) revert UnsupportedChain();

        // Approve bridge
        IERC20(token).approve(bridge, amount);

        // Execute bridge (Lux native bridge or Warp)
        // This is simplified - actual impl handles bridge fees, etc.
        return amount; // Assume 1:1 for now
    }

    // ============ Entry Points ============

    /**
     * @notice Initiate an arbitrage opportunity
     * @param asset Token to borrow
     * @param amount Amount to borrow
     * @param params Arbitrage parameters
     */
    function executeArbitrage(
        address asset,
        uint256 amount,
        ArbParams calldata params
    ) external onlyAuthorized {
        // Encode params for flash loan callback
        bytes memory encodedParams = abi.encode(params);

        // Request flash loan - this will call executeOperation
        (bool success, ) = flashLoanPool.call(
            abi.encodeWithSignature(
                "flashLoan(address,address,uint256,bytes)",
                address(this),
                asset,
                amount,
                encodedParams
            )
        );

        if (!success) revert ExecutionFailed();
    }

    /**
     * @notice Batch execute multiple arbitrage opportunities
     */
    function executeArbitrageBatch(
        address[] calldata assets,
        uint256[] calldata amounts,
        ArbParams[] calldata params
    ) external onlyAuthorized {
        require(assets.length == amounts.length && amounts.length == params.length, "Length mismatch");

        for (uint256 i = 0; i < assets.length; i++) {
            bytes memory encodedParams = abi.encode(params[i]);

            flashLoanPool.call(
                abi.encodeWithSignature(
                    "flashLoan(address,address,uint256,bytes)",
                    address(this),
                    assets[i],
                    amounts[i],
                    encodedParams
                )
            );
        }
    }

    // ============ Admin ============

    function setDexRouter(bytes32 chainID, address router) external onlyOwner {
        dexRouters[chainID] = router;
    }

    function setBridgeContract(bytes32 chainID, address bridge) external onlyOwner {
        bridgeContracts[chainID] = bridge;
    }

    function setAuthorizedExecutor(address executor, bool status) external onlyOwner {
        authorizedExecutors[executor] = status;
    }

    function withdrawProfit(address token, uint256 amount, address recipient) external onlyOwner {
        IERC20(token).transfer(recipient, amount);
    }

    // ============ View Functions ============

    function getExecution(bytes32 executionId) external view returns (ArbExecution memory) {
        return executions[executionId];
    }

    function getStats() external view returns (uint256 totalProfit, uint256 totalExecs) {
        return (totalProfitGenerated, totalArbitrages);
    }
}

// ============ Structs ============

struct CrossChainSwapMessage {
    address targetDex;
    address tokenIn;
    address tokenOut;
    uint256 amountIn;
    uint256 minAmountOut;
    bytes swapData;
    bytes32 returnChain;
    address returnAddress;
}
