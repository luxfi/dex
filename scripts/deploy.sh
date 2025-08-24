#!/bin/bash
set -euo pipefail

# LX DEX Deployment Script
# Usage: ./deploy.sh [staging|production] [action]

ENVIRONMENT=${1:-staging}
ACTION=${2:-deploy}
NAMESPACE="lxdex-${ENVIRONMENT}"
HELM_RELEASE="lxdex-${ENVIRONMENT}"
VERSION=$(git describe --tags --always --dirty)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    command -v kubectl >/dev/null 2>&1 || error "kubectl is not installed"
    command -v helm >/dev/null 2>&1 || error "helm is not installed"
    command -v docker >/dev/null 2>&1 || error "docker is not installed"
    
    # Check kubectl connection
    kubectl cluster-info >/dev/null 2>&1 || error "kubectl is not configured or cluster is unreachable"
    
    log "Prerequisites check passed"
}

# Build and push Docker images
build_images() {
    log "Building Docker images..."
    
    # Build backend image
    docker build -f docker/backend/Dockerfile -t registry.lux.network/lxdex:${VERSION} .
    docker build -f docker/backend/Dockerfile -t registry.lux.network/lxdex:${ENVIRONMENT} .
    
    # Build UI image
    docker build -f docker/ui/Dockerfile -t registry.lux.network/lxdex-ui:${VERSION} ./ui
    docker build -f docker/ui/Dockerfile -t registry.lux.network/lxdex-ui:${ENVIRONMENT} ./ui
    
    log "Pushing images to registry..."
    docker push registry.lux.network/lxdex:${VERSION}
    docker push registry.lux.network/lxdex:${ENVIRONMENT}
    docker push registry.lux.network/lxdex-ui:${VERSION}
    docker push registry.lux.network/lxdex-ui:${ENVIRONMENT}
    
    log "Images built and pushed successfully"
}

# Deploy using kubectl
deploy_kubectl() {
    log "Deploying to ${ENVIRONMENT} using kubectl..."
    
    # Apply configurations
    kubectl apply -f k8s/${ENVIRONMENT}/
    
    # Wait for rollout
    kubectl rollout status statefulset/lxdex-node -n ${NAMESPACE} --timeout=600s
    
    log "Deployment completed"
}

# Deploy using Helm
deploy_helm() {
    log "Deploying to ${ENVIRONMENT} using Helm..."
    
    # Add required repositories
    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm repo add nats https://nats-io.github.io/k8s/helm/charts/
    helm repo update
    
    # Create namespace if it doesn't exist
    kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy using Helm
    helm upgrade --install ${HELM_RELEASE} ./helm/lxdex \
        --namespace ${NAMESPACE} \
        --values helm/lxdex/values.yaml \
        --values helm/lxdex/values.${ENVIRONMENT}.yaml \
        --set image.tag=${VERSION} \
        --wait \
        --timeout 10m
    
    log "Helm deployment completed"
}

# Run tests
run_tests() {
    log "Running deployment tests..."
    
    # Wait for pods to be ready
    kubectl wait --for=condition=ready pod \
        -l app=lxdex \
        -n ${NAMESPACE} \
        --timeout=300s
    
    # Get service endpoint
    SERVICE_IP=$(kubectl get svc lxdex-http -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    
    if [ -z "$SERVICE_IP" ]; then
        SERVICE_IP="localhost"
        kubectl port-forward -n ${NAMESPACE} svc/lxdex-http 8080:80 &
        PF_PID=$!
        sleep 5
    fi
    
    # Test health endpoint
    curl -f http://${SERVICE_IP}:8080/health || error "Health check failed"
    
    # Run integration tests
    go test -tags integration ./test/integration/... -v
    
    # Kill port-forward if started
    [ ! -z "${PF_PID:-}" ] && kill $PF_PID
    
    log "Tests passed"
}

# Rollback deployment
rollback() {
    log "Rolling back ${ENVIRONMENT} deployment..."
    
    if [ "$ENVIRONMENT" == "production" ]; then
        read -p "Are you sure you want to rollback production? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            error "Rollback cancelled"
        fi
    fi
    
    # Rollback using kubectl
    kubectl rollout undo statefulset/lxdex-node -n ${NAMESPACE}
    kubectl rollout status statefulset/lxdex-node -n ${NAMESPACE} --timeout=600s
    
    # Or rollback using Helm
    # helm rollback ${HELM_RELEASE} -n ${NAMESPACE}
    
    log "Rollback completed"
}

# Show status
show_status() {
    log "Showing status for ${ENVIRONMENT}..."
    
    echo -e "\n${GREEN}Pods:${NC}"
    kubectl get pods -n ${NAMESPACE} -l app=lxdex
    
    echo -e "\n${GREEN}Services:${NC}"
    kubectl get svc -n ${NAMESPACE}
    
    echo -e "\n${GREEN}Ingress:${NC}"
    kubectl get ingress -n ${NAMESPACE}
    
    echo -e "\n${GREEN}Recent Events:${NC}"
    kubectl get events -n ${NAMESPACE} --sort-by='.lastTimestamp' | tail -10
}

# Show logs
show_logs() {
    log "Showing logs for ${ENVIRONMENT}..."
    
    POD=${3:-}
    if [ -z "$POD" ]; then
        # Get first pod
        POD=$(kubectl get pods -n ${NAMESPACE} -l app=lxdex -o jsonpath='{.items[0].metadata.name}')
    fi
    
    kubectl logs -n ${NAMESPACE} $POD -f
}

# Scale deployment
scale() {
    REPLICAS=${3:-3}
    log "Scaling ${ENVIRONMENT} to ${REPLICAS} replicas..."
    
    kubectl scale statefulset/lxdex-node -n ${NAMESPACE} --replicas=${REPLICAS}
    kubectl rollout status statefulset/lxdex-node -n ${NAMESPACE} --timeout=600s
    
    log "Scaling completed"
}

# Main execution
main() {
    check_prerequisites
    
    case $ACTION in
        deploy)
            if [ "$ENVIRONMENT" == "production" ]; then
                warning "Deploying to PRODUCTION environment"
                read -p "Are you sure? (y/N) " -n 1 -r
                echo
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    error "Deployment cancelled"
                fi
            fi
            
            build_images
            deploy_helm  # or deploy_kubectl
            run_tests
            show_status
            ;;
        
        rollback)
            rollback
            ;;
        
        status)
            show_status
            ;;
        
        logs)
            show_logs
            ;;
        
        scale)
            scale
            ;;
        
        test)
            run_tests
            ;;
        
        *)
            echo "Usage: $0 [staging|production] [deploy|rollback|status|logs|scale|test]"
            exit 1
            ;;
    esac
    
    log "Operation completed successfully"
}

# Run main function
main