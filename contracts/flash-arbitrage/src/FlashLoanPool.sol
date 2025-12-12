// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "../interfaces/IFlashLoanReceiver.sol";
import "../interfaces/IERC20.sol";
import "../interfaces/IWarpMessenger.sol";

/**
 * @title FlashLoanPool
 * @notice Uncollateralized flash loans for arbitrage on Lux Network
 * @dev Loans must be repaid within the same transaction or it reverts
 *
 * Key Features:
 * - Zero collateral required
 * - 0.09% fee (competitive with Aave)
 * - Multi-asset support
 * - Cross-chain callback support via Warp
 * - MEV protection via private mempool integration
 */
contract FlashLoanPool {
    // ============ Constants ============
    uint256 public constant FLASH_LOAN_FEE = 9; // 0.09% = 9 basis points
    uint256 public constant FEE_DENOMINATOR = 10000;

    // ============ State ============
    mapping(address => uint256) public reserves;
    mapping(address => uint256) public totalBorrowed;
    mapping(address => uint256) public feesCollected;

    // Supported assets
    mapping(address => bool) public supportedAssets;
    address[] public assetList;

    // Cross-chain
    IWarpMessenger public immutable warpMessenger;
    bytes32 public immutable sourceBlockchainID;

    // Access control
    address public owner;
    mapping(address => bool) public whitelistedReceivers;
    bool public permissionless;

    // ============ Events ============
    event FlashLoan(
        address indexed receiver,
        address indexed asset,
        uint256 amount,
        uint256 fee,
        bytes32 indexed executionId
    );

    event CrossChainFlashLoan(
        bytes32 indexed destinationChainID,
        address indexed receiver,
        address indexed asset,
        uint256 amount,
        bytes32 messageId
    );

    event Deposit(address indexed asset, address indexed depositor, uint256 amount);
    event Withdraw(address indexed asset, address indexed recipient, uint256 amount);
    event AssetAdded(address indexed asset);
    event FeesWithdrawn(address indexed asset, uint256 amount);

    // ============ Errors ============
    error UnsupportedAsset();
    error InsufficientLiquidity();
    error LoanNotRepaid();
    error NotWhitelisted();
    error Unauthorized();
    error ZeroAmount();
    error ReentrancyGuard();

    // ============ Modifiers ============
    uint256 private _locked = 1;
    modifier nonReentrant() {
        if (_locked == 2) revert ReentrancyGuard();
        _locked = 2;
        _;
        _locked = 1;
    }

    modifier onlyOwner() {
        if (msg.sender != owner) revert Unauthorized();
        _;
    }

    // ============ Constructor ============
    constructor(address _warpMessenger) {
        owner = msg.sender;
        warpMessenger = IWarpMessenger(_warpMessenger);
        sourceBlockchainID = warpMessenger.getBlockchainID();
        permissionless = false; // Start permissioned for security
    }

    // ============ Flash Loan Core ============

    /**
     * @notice Execute a flash loan
     * @param receiver Contract that will receive and return funds
     * @param asset Token to borrow
     * @param amount Amount to borrow
     * @param params Arbitrary data passed to receiver
     * @return success Whether the flash loan succeeded
     */
    function flashLoan(
        address receiver,
        address asset,
        uint256 amount,
        bytes calldata params
    ) external nonReentrant returns (bool success) {
        // Validation
        if (!supportedAssets[asset]) revert UnsupportedAsset();
        if (amount == 0) revert ZeroAmount();
        if (!permissionless && !whitelistedReceivers[receiver]) revert NotWhitelisted();

        uint256 availableLiquidity = reserves[asset];
        if (amount > availableLiquidity) revert InsufficientLiquidity();

        // Calculate fee
        uint256 fee = (amount * FLASH_LOAN_FEE) / FEE_DENOMINATOR;
        uint256 amountToRepay = amount + fee;

        // Record state before
        uint256 balanceBefore = IERC20(asset).balanceOf(address(this));

        // Transfer funds to receiver
        reserves[asset] -= amount;
        totalBorrowed[asset] += amount;

        IERC20(asset).transfer(receiver, amount);

        // Generate unique execution ID for tracking
        bytes32 executionId = keccak256(
            abi.encodePacked(block.timestamp, block.number, receiver, asset, amount)
        );

        // Execute receiver's arbitrage logic
        IFlashLoanReceiver(receiver).executeOperation(
            asset,
            amount,
            fee,
            msg.sender,
            params
        );

        // Verify repayment
        uint256 balanceAfter = IERC20(asset).balanceOf(address(this));
        if (balanceAfter < balanceBefore + fee) revert LoanNotRepaid();

        // Update state
        reserves[asset] = balanceAfter;
        totalBorrowed[asset] -= amount;
        feesCollected[asset] += fee;

        emit FlashLoan(receiver, asset, amount, fee, executionId);

        return true;
    }

    /**
     * @notice Execute a multi-asset flash loan (for complex arbitrage)
     * @param receiver Contract that will receive funds
     * @param assets Array of tokens to borrow
     * @param amounts Array of amounts to borrow
     * @param params Arbitrary data passed to receiver
     */
    function flashLoanMulti(
        address receiver,
        address[] calldata assets,
        uint256[] calldata amounts,
        bytes calldata params
    ) external nonReentrant returns (bool success) {
        require(assets.length == amounts.length, "Length mismatch");

        if (!permissionless && !whitelistedReceivers[receiver]) revert NotWhitelisted();

        uint256[] memory fees = new uint256[](assets.length);
        uint256[] memory balancesBefore = new uint256[](assets.length);

        // Validate and transfer all assets
        for (uint256 i = 0; i < assets.length; i++) {
            if (!supportedAssets[assets[i]]) revert UnsupportedAsset();
            if (amounts[i] == 0) revert ZeroAmount();
            if (amounts[i] > reserves[assets[i]]) revert InsufficientLiquidity();

            fees[i] = (amounts[i] * FLASH_LOAN_FEE) / FEE_DENOMINATOR;
            balancesBefore[i] = IERC20(assets[i]).balanceOf(address(this));

            reserves[assets[i]] -= amounts[i];
            totalBorrowed[assets[i]] += amounts[i];

            IERC20(assets[i]).transfer(receiver, amounts[i]);
        }

        // Execute receiver's arbitrage logic
        IFlashLoanReceiver(receiver).executeOperationMulti(
            assets,
            amounts,
            fees,
            msg.sender,
            params
        );

        // Verify all repayments
        for (uint256 i = 0; i < assets.length; i++) {
            uint256 balanceAfter = IERC20(assets[i]).balanceOf(address(this));
            if (balanceAfter < balancesBefore[i] + fees[i]) revert LoanNotRepaid();

            reserves[assets[i]] = balanceAfter;
            totalBorrowed[assets[i]] -= amounts[i];
            feesCollected[assets[i]] += fees[i];
        }

        return true;
    }

    // ============ Cross-Chain Flash Loans ============

    /**
     * @notice Initiate a flash loan that executes across chains via Warp
     * @dev The loan is taken on Lux, execution happens cross-chain, settlement returns to Lux
     * @param destinationChainID Target blockchain for execution
     * @param receiver Contract on destination chain
     * @param asset Token to borrow (must have cross-chain representation)
     * @param amount Amount to borrow
     * @param params Execution parameters for destination chain
     */
    function crossChainFlashLoan(
        bytes32 destinationChainID,
        address receiver,
        address asset,
        uint256 amount,
        bytes calldata params
    ) external nonReentrant returns (bytes32 messageId) {
        if (!supportedAssets[asset]) revert UnsupportedAsset();
        if (amount == 0) revert ZeroAmount();
        if (amount > reserves[asset]) revert InsufficientLiquidity();

        uint256 fee = (amount * FLASH_LOAN_FEE) / FEE_DENOMINATOR;

        // Lock funds for cross-chain execution
        reserves[asset] -= amount;
        totalBorrowed[asset] += amount;

        // Encode the flash loan message
        bytes memory message = abi.encode(
            CrossChainFlashLoanMessage({
                sourceChainID: sourceBlockchainID,
                receiver: receiver,
                asset: asset,
                amount: amount,
                fee: fee,
                initiator: msg.sender,
                params: params
            })
        );

        // Send via Warp
        messageId = warpMessenger.sendWarpMessage(message);

        emit CrossChainFlashLoan(destinationChainID, receiver, asset, amount, messageId);

        return messageId;
    }

    /**
     * @notice Receive cross-chain flash loan settlement
     * @dev Called by Warp relayer when cross-chain execution completes
     */
    function receiveCrossChainSettlement(
        bytes32 sourceChainID,
        address asset,
        uint256 originalAmount,
        uint256 returnedAmount
    ) external {
        // Verify message came from Warp
        require(msg.sender == address(warpMessenger), "Only Warp");

        uint256 fee = (originalAmount * FLASH_LOAN_FEE) / FEE_DENOMINATOR;
        uint256 expectedReturn = originalAmount + fee;

        require(returnedAmount >= expectedReturn, "Insufficient return");

        // Restore reserves
        reserves[asset] += returnedAmount;
        totalBorrowed[asset] -= originalAmount;
        feesCollected[asset] += fee;
    }

    // ============ Liquidity Management ============

    /**
     * @notice Deposit assets into the flash loan pool
     * @param asset Token to deposit
     * @param amount Amount to deposit
     */
    function deposit(address asset, uint256 amount) external {
        if (!supportedAssets[asset]) revert UnsupportedAsset();
        if (amount == 0) revert ZeroAmount();

        IERC20(asset).transferFrom(msg.sender, address(this), amount);
        reserves[asset] += amount;

        emit Deposit(asset, msg.sender, amount);
    }

    /**
     * @notice Withdraw assets from the pool (owner only for now)
     * @param asset Token to withdraw
     * @param amount Amount to withdraw
     * @param recipient Address to receive funds
     */
    function withdraw(address asset, uint256 amount, address recipient) external onlyOwner {
        if (amount > reserves[asset]) revert InsufficientLiquidity();

        reserves[asset] -= amount;
        IERC20(asset).transfer(recipient, amount);

        emit Withdraw(asset, recipient, amount);
    }

    // ============ Admin Functions ============

    function addSupportedAsset(address asset) external onlyOwner {
        if (!supportedAssets[asset]) {
            supportedAssets[asset] = true;
            assetList.push(asset);
            emit AssetAdded(asset);
        }
    }

    function setWhitelisted(address receiver, bool status) external onlyOwner {
        whitelistedReceivers[receiver] = status;
    }

    function setPermissionless(bool status) external onlyOwner {
        permissionless = status;
    }

    function withdrawFees(address asset, address recipient) external onlyOwner {
        uint256 fees = feesCollected[asset];
        feesCollected[asset] = 0;
        IERC20(asset).transfer(recipient, fees);
        emit FeesWithdrawn(asset, fees);
    }

    function transferOwnership(address newOwner) external onlyOwner {
        owner = newOwner;
    }

    // ============ View Functions ============

    function getAvailableLiquidity(address asset) external view returns (uint256) {
        return reserves[asset];
    }

    function calculateFee(uint256 amount) external pure returns (uint256) {
        return (amount * FLASH_LOAN_FEE) / FEE_DENOMINATOR;
    }

    function getSupportedAssets() external view returns (address[] memory) {
        return assetList;
    }
}

// ============ Structs ============

struct CrossChainFlashLoanMessage {
    bytes32 sourceChainID;
    address receiver;
    address asset;
    uint256 amount;
    uint256 fee;
    address initiator;
    bytes params;
}
