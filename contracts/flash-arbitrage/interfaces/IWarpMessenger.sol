// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/**
 * @title IWarpMessenger
 * @notice Interface for Lux Warp cross-chain messaging
 * @dev Warp enables sub-second cross-chain communication
 */
interface IWarpMessenger {
    /**
     * @notice Get the blockchain ID of this chain
     */
    function getBlockchainID() external view returns (bytes32);

    /**
     * @notice Send a Warp message to be relayed cross-chain
     * @param message The message payload
     * @return messageID Unique identifier for tracking
     */
    function sendWarpMessage(bytes calldata message) external returns (bytes32 messageID);

    /**
     * @notice Get a verified Warp message
     * @param index The message index
     * @return message The verified message
     * @return valid Whether the message is valid
     */
    function getVerifiedWarpMessage(uint32 index) external view returns (
        WarpMessage memory message,
        bool valid
    );

    /**
     * @notice Get block hash for verification
     */
    function getVerifiedWarpBlockHash(uint32 index) external view returns (
        WarpBlockHash memory blockHash,
        bool valid
    );
}

struct WarpMessage {
    bytes32 sourceChainID;
    address originSenderAddress;
    bytes payload;
}

struct WarpBlockHash {
    bytes32 sourceChainID;
    bytes32 blockHash;
}
