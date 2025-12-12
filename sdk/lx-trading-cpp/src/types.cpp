// LX Trading SDK - Types Implementation

#include <lx/trading/types.hpp>
#include <algorithm>
#include <charconv>
#include <sstream>

namespace lx::trading {

// Decimal implementation
Decimal Decimal::from_string(std::string_view s) {
    // Find decimal point
    auto dot = s.find('.');
    if (dot == std::string_view::npos) {
        // Integer
        int64_t val = 0;
        std::from_chars(s.data(), s.data() + s.size(), val);
        return Decimal(val * SCALE);
    }

    // Parse integer and fractional parts
    std::string_view int_part = s.substr(0, dot);
    std::string_view frac_part = s.substr(dot + 1);

    int64_t int_val = 0;
    if (!int_part.empty()) {
        std::from_chars(int_part.data(), int_part.data() + int_part.size(), int_val);
    }

    int64_t frac_val = 0;
    if (!frac_part.empty()) {
        // Pad or truncate to PRECISION digits
        std::string frac_str(frac_part);
        if (frac_str.size() < PRECISION) {
            frac_str.append(PRECISION - frac_str.size(), '0');
        } else if (frac_str.size() > PRECISION) {
            frac_str = frac_str.substr(0, PRECISION);
        }
        std::from_chars(frac_str.data(), frac_str.data() + frac_str.size(), frac_val);
    }

    bool negative = (s[0] == '-');
    int64_t result = int_val * SCALE + (negative ? -frac_val : frac_val);
    return Decimal(result);
}

std::string Decimal::to_string() const {
    int64_t abs_val = value_ < 0 ? -value_ : value_;
    int64_t int_part = abs_val / SCALE;
    int64_t frac_part = abs_val % SCALE;

    std::ostringstream oss;
    if (value_ < 0) oss << '-';
    oss << int_part << '.';

    // Format fractional part with leading zeros
    std::string frac_str = std::to_string(frac_part);
    oss << std::string(PRECISION - frac_str.size(), '0') << frac_str;

    std::string result = oss.str();

    // Trim trailing zeros after decimal point
    size_t last_non_zero = result.find_last_not_of('0');
    if (last_non_zero != std::string::npos && result[last_non_zero] == '.') {
        last_non_zero--;
    }
    result = result.substr(0, last_non_zero + 1);

    return result;
}

// TradingPair implementation
std::optional<TradingPair> TradingPair::from_symbol(std::string_view symbol) {
    const char separators[] = {'-', '/', '_'};

    for (char sep : separators) {
        auto pos = symbol.find(sep);
        if (pos != std::string_view::npos) {
            TradingPair pair;
            auto base = symbol.substr(0, pos);
            auto quote = symbol.substr(pos + 1);

            if (base.size() > 15 || quote.size() > 15) {
                return std::nullopt;
            }

            std::copy(base.begin(), base.end(), pair.base.begin());
            std::copy(quote.begin(), quote.end(), pair.quote.begin());

            return pair;
        }
    }

    return std::nullopt;
}

std::string TradingPair::to_hummingbot() const {
    std::string result;
    result.reserve(33);
    for (char c : base) {
        if (c == 0) break;
        result += c;
    }
    result += '-';
    for (char c : quote) {
        if (c == 0) break;
        result += c;
    }
    return result;
}

std::string TradingPair::to_ccxt() const {
    std::string result;
    result.reserve(33);
    for (char c : base) {
        if (c == 0) break;
        result += c;
    }
    result += '/';
    for (char c : quote) {
        if (c == 0) break;
        result += c;
    }
    return result;
}

// OrderRequest factory methods
OrderRequest OrderRequest::market(std::string_view symbol, Side side, Decimal quantity) {
    OrderRequest req;
    req.symbol = std::string(symbol);
    req.side = side;
    req.order_type = OrderType::Market;
    req.quantity = quantity;
    req.time_in_force = TimeInForce::IOC;
    return req;
}

OrderRequest OrderRequest::limit(std::string_view symbol, Side side, Decimal quantity, Decimal price) {
    OrderRequest req;
    req.symbol = std::string(symbol);
    req.side = side;
    req.order_type = OrderType::Limit;
    req.quantity = quantity;
    req.price = price;
    req.time_in_force = TimeInForce::GTC;
    return req;
}

}  // namespace lx::trading
