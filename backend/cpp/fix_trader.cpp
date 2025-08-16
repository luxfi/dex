// Pure C++ FIX Trading Client - High-performance FIX message sender
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <random>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <sstream>
#include <iomanip>

class FIXTrader {
private:
    std::string senderID;
    std::string targetID;
    std::atomic<int> seqNum{1};
    std::atomic<uint64_t> messagesSent{0};
    std::atomic<uint64_t> bytesSent{0};
    std::atomic<uint64_t> errors{0};
    std::vector<std::string> symbols = {"BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "MATIC/USD"};
    
    // Calculate FIX checksum
    std::string calculateChecksum(const std::string& msg) {
        int sum = 0;
        for (char c : msg) {
            sum += static_cast<unsigned char>(c);
        }
        char buf[4];
        snprintf(buf, sizeof(buf), "%03d", sum % 256);
        return std::string(buf);
    }
    
    // Get current timestamp in FIX format
    std::string getTimestamp() {
        auto now = std::chrono::system_clock::now();
        auto tt = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        
        std::stringstream ss;
        ss << std::put_time(std::gmtime(&tt), "%Y%m%d-%H:%M:%S");
        ss << "." << std::setfill('0') << std::setw(3) << ms.count();
        return ss.str();
    }
    
public:
    FIXTrader(const std::string& sender, const std::string& target) 
        : senderID(sender), targetID(target) {}
    
    // Build a FIX New Order Single message
    std::string buildNewOrder(const std::string& symbol, double price, double qty, bool isBuy) {
        std::stringstream body;
        
        // Message type and header
        body << "35=D\x01";  // MsgType = NewOrderSingle
        body << "49=" << senderID << "\x01";  // SenderCompID
        body << "56=" << targetID << "\x01";  // TargetCompID
        body << "34=" << seqNum.fetch_add(1) << "\x01";  // MsgSeqNum
        body << "52=" << getTimestamp() << "\x01";  // SendingTime
        
        // Order fields
        body << "11=ORD" << std::chrono::steady_clock::now().time_since_epoch().count() << "\x01";  // ClOrdID
        body << "55=" << symbol << "\x01";  // Symbol
        body << "54=" << (isBuy ? "1" : "2") << "\x01";  // Side
        body << "38=" << std::fixed << std::setprecision(2) << qty << "\x01";  // OrderQty
        body << "40=2\x01";  // OrdType = Limit
        body << "44=" << std::fixed << std::setprecision(2) << price << "\x01";  // Price
        body << "59=0\x01";  // TimeInForce = Day
        body << "60=" << getTimestamp() << "\x01";  // TransactTime
        
        std::string bodyStr = body.str();
        
        // Build complete message
        std::stringstream msg;
        msg << "8=FIX.4.4\x01";  // BeginString
        msg << "9=" << bodyStr.length() << "\x01";  // BodyLength
        msg << bodyStr;
        
        // Add checksum
        std::string msgStr = msg.str();
        msg << "10=" << calculateChecksum(msgStr) << "\x01";  // CheckSum
        
        return msg.str();
    }
    
    // Send message via TCP
    bool sendMessage(int socket, const std::string& msg) {
        int sent = send(socket, msg.c_str(), msg.length(), 0);
        if (sent > 0) {
            messagesSent.fetch_add(1);
            bytesSent.fetch_add(sent);
            return true;
        } else {
            errors.fetch_add(1);
            return false;
        }
    }
    
    // Get statistics
    void getStats(uint64_t& msgs, uint64_t& bytes, uint64_t& errs) {
        msgs = messagesSent.load();
        bytes = bytesSent.load();
        errs = errors.load();
    }
    
    void resetStats() {
        messagesSent = 0;
        bytesSent = 0;
        errors = 0;
    }
};

// Worker thread for sending orders
void traderWorker(int id, const std::string& serverAddr, int port, int ordersPerSec, 
                  int duration, FIXTrader* trader) {
    // Create socket
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        std::cerr << "Trader " << id << ": Failed to create socket\n";
        return;
    }
    
    // Connect to server
    struct sockaddr_in server;
    server.sin_family = AF_INET;
    server.sin_port = htons(port);
    inet_pton(AF_INET, serverAddr.c_str(), &server.sin_addr);
    
    if (connect(sock, (struct sockaddr*)&server, sizeof(server)) < 0) {
        std::cerr << "Trader " << id << ": Failed to connect to " << serverAddr << ":" << port << "\n";
        close(sock);
        return;
    }
    
    // Random number generator
    std::mt19937 rng(id);
    std::uniform_real_distribution<> priceDist(30000, 40000);
    std::uniform_real_distribution<> qtyDist(0.1, 10.0);
    std::uniform_int_distribution<> sideDist(0, 1);
    std::uniform_int_distribution<> symbolDist(0, 4);
    
    std::vector<std::string> symbols = {"BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "MATIC/USD"};
    
    // Send orders at specified rate
    auto interval = std::chrono::microseconds(1000000 / ordersPerSec);
    auto startTime = std::chrono::steady_clock::now();
    auto endTime = startTime + std::chrono::seconds(duration);
    auto nextSend = startTime;
    
    while (std::chrono::steady_clock::now() < endTime) {
        // Build and send order
        std::string msg = trader->buildNewOrder(
            symbols[symbolDist(rng)],
            priceDist(rng),
            qtyDist(rng),
            sideDist(rng) == 0
        );
        
        trader->sendMessage(sock, msg);
        
        // Rate limiting
        nextSend += interval;
        std::this_thread::sleep_until(nextSend);
    }
    
    close(sock);
}

int main(int argc, char* argv[]) {
    // Parse arguments
    std::string serverAddr = "127.0.0.1";
    int port = 5555;
    int numTraders = 100;
    int ordersPerSec = 100;
    int duration = 30;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-server" && i + 1 < argc) {
            serverAddr = argv[++i];
        } else if (arg == "-port" && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        } else if (arg == "-traders" && i + 1 < argc) {
            numTraders = std::stoi(argv[++i]);
        } else if (arg == "-rate" && i + 1 < argc) {
            ordersPerSec = std::stoi(argv[++i]);
        } else if (arg == "-duration" && i + 1 < argc) {
            duration = std::stoi(argv[++i]);
        }
    }
    
    std::cout << "⚡ C++ FIX Trading Client\n";
    std::cout << "========================\n";
    std::cout << "Server: " << serverAddr << ":" << port << "\n";
    std::cout << "Traders: " << numTraders << "\n";
    std::cout << "Rate: " << ordersPerSec << " orders/sec per trader\n";
    std::cout << "Total Rate: " << (numTraders * ordersPerSec) << " orders/sec\n";
    std::cout << "Duration: " << duration << " seconds\n\n";
    
    // Create FIX trader
    FIXTrader trader("CPP_TRADER", "EXCHANGE");
    
    // Start trader threads
    std::vector<std::thread> threads;
    auto startTime = std::chrono::steady_clock::now();
    
    for (int i = 0; i < numTraders; i++) {
        threads.emplace_back(traderWorker, i, serverAddr, port, ordersPerSec, duration, &trader);
    }
    
    // Print stats while running
    while (std::chrono::steady_clock::now() < startTime + std::chrono::seconds(duration)) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        
        uint64_t msgs, bytes, errs;
        trader.getStats(msgs, bytes, errs);
        
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - startTime).count();
        
        std::cout << "\r📊 Sent: " << msgs 
                  << " | Rate: " << (msgs / elapsed) << "/sec"
                  << " | Data: " << (bytes / 1024.0 / 1024.0) << " MB"
                  << " | Errors: " << errs << std::flush;
    }
    
    // Wait for all threads
    for (auto& t : threads) {
        t.join();
    }
    
    // Final stats
    uint64_t finalMsgs, finalBytes, finalErrs;
    trader.getStats(finalMsgs, finalBytes, finalErrs);
    double totalTime = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - startTime).count();
    
    std::cout << "\n\n=== Final Results ===\n";
    std::cout << "Duration: " << totalTime << " seconds\n";
    std::cout << "Messages Sent: " << finalMsgs << "\n";
    std::cout << "Errors: " << finalErrs << "\n";
    std::cout << "Average Rate: " << (finalMsgs / totalTime) << " msgs/sec\n";
    std::cout << "Total Data: " << (finalBytes / 1024.0 / 1024.0) << " MB\n";
    std::cout << "Network: " << (finalBytes * 8.0 / totalTime / 1e9) << " Gbps\n";
    std::cout << "Efficiency: " << (100.0 * finalMsgs / totalTime / (numTraders * ordersPerSec)) << "%\n";
    
    return 0;
}
