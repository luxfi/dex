#!/bin/bash
# Generate ML-DSA certificates for QZMQ
# This script creates a dev CA and node certificates using ML-DSA (Dilithium)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CERT_DIR="$PROJECT_DIR/certs"
CONFIG_DIR="$PROJECT_DIR/configs/qzmq"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   QZMQ ML-DSA Certificate Generator    ${NC}"
echo -e "${GREEN}========================================${NC}"

# Create certificate directory
mkdir -p "$CERT_DIR"/{ca,nodes,clients}

# Function to generate ML-DSA key pair (stub - would use actual PQ crypto library)
generate_mldsa_keypair() {
    local name=$1
    local type=$2
    local strength=$3
    
    echo -e "${YELLOW}Generating ML-DSA-${strength} keypair for ${name}...${NC}"
    
    # In production, this would call actual ML-DSA implementation
    # For now, create placeholder files
    
    # Private key (would be ML-DSA private key)
    cat > "$CERT_DIR/${type}/${name}.key" << EOF
-----BEGIN ML-DSA PRIVATE KEY-----
Algorithm: ML-DSA-${strength}
Name: ${name}
Type: ${type}
Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
# This is a placeholder for actual ML-DSA private key
# In production, use liboqs or similar for real PQ crypto
$(openssl rand -base64 128)
-----END ML-DSA PRIVATE KEY-----
EOF
    
    # Public key (would be ML-DSA public key)
    cat > "$CERT_DIR/${type}/${name}.pub" << EOF
-----BEGIN ML-DSA PUBLIC KEY-----
Algorithm: ML-DSA-${strength}
Name: ${name}
Type: ${type}
Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
# This is a placeholder for actual ML-DSA public key
$(openssl rand -base64 64)
-----END ML-DSA PUBLIC KEY-----
EOF
}

# Function to create certificate
create_certificate() {
    local name=$1
    local type=$2
    local issuer=$3
    local subject=$4
    local strength=$5
    
    echo -e "${YELLOW}Creating certificate for ${name}...${NC}"
    
    cat > "$CERT_DIR/${type}/${name}.cert" << EOF
-----BEGIN CERTIFICATE-----
Version: 3
Serial Number: $(openssl rand -hex 16)
Signature Algorithm: ML-DSA-${strength}
Issuer: ${issuer}
Validity:
    Not Before: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
    Not After: $(date -u -d "+30 days" +"%Y-%m-%d %H:%M:%S UTC")
Subject: ${subject}
Public Key Algorithm: ML-DSA-${strength}
Public Key:
$(cat "$CERT_DIR/${type}/${name}.pub" | grep -v "BEGIN\|END")
Extensions:
    Key Usage: Digital Signature, Key Agreement
    Extended Key Usage: TLS Server Auth, TLS Client Auth
    Subject Alternative Name: DNS:${name}.lx.dex
Signature:
    # This would be the ML-DSA signature of the certificate
    $(openssl rand -base64 128)
-----END CERTIFICATE-----
EOF
}

# Generate CA certificate
echo -e "${GREEN}Step 1: Generating Certificate Authority${NC}"
generate_mldsa_keypair "lx-ca" "ca" "3"
create_certificate "lx-ca" "ca" \
    "CN=LX DEX Root CA, O=Lux Network, C=US" \
    "CN=LX DEX Root CA, O=Lux Network, C=US" \
    "3"

# Generate node certificates
echo -e "${GREEN}Step 2: Generating Node Certificates${NC}"
for i in {0..2}; do
    generate_mldsa_keypair "node-${i}" "nodes" "2"
    create_certificate "node-${i}" "nodes" \
        "CN=LX DEX Root CA, O=Lux Network, C=US" \
        "CN=Node-${i}, OU=DEX Nodes, O=Lux Network, C=US" \
        "2"
done

# Generate client certificates
echo -e "${GREEN}Step 3: Generating Client Certificates${NC}"
for client in "trader" "market-maker" "admin"; do
    generate_mldsa_keypair "${client}" "clients" "2"
    create_certificate "${client}" "clients" \
        "CN=LX DEX Root CA, O=Lux Network, C=US" \
        "CN=${client}, OU=DEX Clients, O=Lux Network, C=US" \
        "2"
done

# Create certificate configuration
echo -e "${GREEN}Step 4: Creating Certificate Configuration${NC}"
cat > "$CONFIG_DIR/certs.yaml" << EOF
# ML-DSA Certificate Configuration
certificates:
  ca:
    path: ${CERT_DIR}/ca/lx-ca.cert
    key: ${CERT_DIR}/ca/lx-ca.key
    algorithm: ML-DSA-3
    
  nodes:
    - id: 0
      cert: ${CERT_DIR}/nodes/node-0.cert
      key: ${CERT_DIR}/nodes/node-0.key
      algorithm: ML-DSA-2
    - id: 1
      cert: ${CERT_DIR}/nodes/node-1.cert
      key: ${CERT_DIR}/nodes/node-1.key
      algorithm: ML-DSA-2
    - id: 2
      cert: ${CERT_DIR}/nodes/node-2.cert
      key: ${CERT_DIR}/nodes/node-2.key
      algorithm: ML-DSA-2
      
  clients:
    trader:
      cert: ${CERT_DIR}/clients/trader.cert
      key: ${CERT_DIR}/clients/trader.key
      algorithm: ML-DSA-2
      permissions: ["read", "trade"]
    market_maker:
      cert: ${CERT_DIR}/clients/market-maker.cert
      key: ${CERT_DIR}/clients/market-maker.key
      algorithm: ML-DSA-2
      permissions: ["read", "trade", "provide_liquidity"]
    admin:
      cert: ${CERT_DIR}/clients/admin.cert
      key: ${CERT_DIR}/clients/admin.key
      algorithm: ML-DSA-2
      permissions: ["all"]

  validation:
    verify_chain: true
    check_revocation: true
    pin_public_keys: true
    max_chain_depth: 3
    
  renewal:
    auto_renew: true
    days_before_expiry: 7
    rotation_strategy: "gradual"
EOF

# Create certificate pinning configuration
echo -e "${GREEN}Step 5: Creating Public Key Pins${NC}"
cat > "$CONFIG_DIR/pins.yaml" << EOF
# Public Key Pinning Configuration
pins:
  ca:
    algorithm: ML-DSA-3
    hash: sha384
    value: $(openssl rand -base64 48)
    
  nodes:
    - id: 0
      algorithm: ML-DSA-2
      hash: sha256
      value: $(openssl rand -base64 32)
    - id: 1
      algorithm: ML-DSA-2
      hash: sha256
      value: $(openssl rand -base64 32)
    - id: 2
      algorithm: ML-DSA-2
      hash: sha256
      value: $(openssl rand -base64 32)
      
  backup_pins:
    - algorithm: ML-DSA-3
      hash: sha384
      value: $(openssl rand -base64 48)
      
  max_age: 5184000  # 60 days
  include_subdomains: true
  report_uri: "https://lx.dex/pkp-report"
EOF

# Create test script
echo -e "${GREEN}Step 6: Creating Certificate Test Script${NC}"
cat > "$CERT_DIR/test-certs.sh" << 'EOF'
#!/bin/bash
# Test ML-DSA certificates

CERT_DIR="$(dirname "$0")"

echo "Testing ML-DSA Certificates..."

# Verify CA certificate
echo "1. Verifying CA certificate..."
if [ -f "$CERT_DIR/ca/lx-ca.cert" ]; then
    echo "   ✓ CA certificate exists"
else
    echo "   ✗ CA certificate missing"
fi

# Verify node certificates
echo "2. Verifying node certificates..."
for i in {0..2}; do
    if [ -f "$CERT_DIR/nodes/node-${i}.cert" ]; then
        echo "   ✓ Node ${i} certificate exists"
    else
        echo "   ✗ Node ${i} certificate missing"
    fi
done

# Verify client certificates
echo "3. Verifying client certificates..."
for client in trader market-maker admin; do
    if [ -f "$CERT_DIR/clients/${client}.cert" ]; then
        echo "   ✓ Client ${client} certificate exists"
    else
        echo "   ✗ Client ${client} certificate missing"
    fi
done

echo "Certificate test complete!"
EOF

chmod +x "$CERT_DIR/test-certs.sh"

# Summary
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}    Certificate Generation Complete!    ${NC}"
echo -e "${GREEN}========================================${NC}"
echo
echo "Generated certificates in: $CERT_DIR"
echo
echo "Certificate types:"
echo "  • CA: ML-DSA-3 (highest security)"
echo "  • Nodes: ML-DSA-2 (balanced)"
echo "  • Clients: ML-DSA-2 (balanced)"
echo
echo "Files created:"
find "$CERT_DIR" -type f -name "*.cert" -o -name "*.key" -o -name "*.pub" | sort
echo
echo "Configuration files:"
echo "  • $CONFIG_DIR/certs.yaml"
echo "  • $CONFIG_DIR/pins.yaml"
echo
echo -e "${YELLOW}Note: These are placeholder certificates for development.${NC}"
echo -e "${YELLOW}In production, use actual ML-DSA implementation (e.g., liboqs).${NC}"
echo
echo "To test certificates, run:"
echo "  $CERT_DIR/test-certs.sh"