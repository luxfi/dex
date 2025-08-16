# LX DEX Deployment Guide

## Overview

This guide covers deployment strategies for the LX DEX platform across different environments, from local development to production-scale deployments supporting millions of orders per second.

## Deployment Architectures

### 1. Development (Single Node)
```
┌──────────────────────────────────┐
│         Developer Machine         │
│  ┌────────────────────────────┐  │
│  │    Docker Compose Stack    │  │
│  │  - LX Engine (Hybrid)      │  │
│  │  - Prometheus              │  │
│  │  - Grafana                 │  │
│  └────────────────────────────┘  │
└──────────────────────────────────┘
```

### 2. Staging (Multi-Engine)
```
┌─────────────────────────────────────────────┐
│              Load Balancer                  │
└────────┬────────────┬────────────┬──────────┘
         │            │            │
    ┌────▼───┐   ┌───▼────┐  ┌───▼────┐
    │Go Engine│   │C++ Engine│ │TS Engine│
    └────┬───┘   └────┬────┘  └───┬────┘
         │            │            │
    ┌────▼────────────▼────────────▼────┐
    │          Shared Services           │
    │  - NATS    - Redis   - PostgreSQL  │
    └────────────────────────────────────┘
```

### 3. Production (Geo-Distributed)
```
┌──────────────────────────────────────────────┐
│                 Global CDN                   │
└────────────────────┬─────────────────────────┘
                     │
        ┌────────────▼────────────┐
        │    Global Load Balancer  │
        └────┬──────────────┬─────┘
             │              │
    ┌────────▼───┐    ┌────▼──────┐
    │  US Region │    │ EU Region  │
    │            │    │            │
    │ ┌────────┐ │    │ ┌────────┐ │
    │ │C++ CEX │ │    │ │C++ CEX │ │
    │ └────────┘ │    │ └────────┘ │
    │            │    │            │
    │ ┌────────┐ │    │ ┌────────┐ │
    │ │Hybrid  │ │    │ │Hybrid  │ │
    │ │DEX     │ │    │ │DEX     │ │
    │ └────────┘ │    │ └────────┘ │
    └────────────┘    └────────────┘
```

## Environment-Specific Deployments

### Local Development

#### Quick Start
```bash
# Clone repository
git clone https://github.com/luxfi/dex.git
cd dex

# Run setup script
./scripts/dev-setup.sh

# Start services
docker-compose -f docker-compose.dev.yml up

# Or run directly
make server  # Terminal 1
make trader  # Terminal 2
```

#### Development Configuration
```yaml
# config/dev.yaml
engine:
  type: hybrid
  debug: true
  port: 50051

performance:
  max_orders_per_second: 10000
  batch_size: 10

monitoring:
  enabled: true
  prometheus_port: 9090
  
logging:
  level: debug
  format: text
```

### Staging Environment

#### Docker Compose Deployment
```yaml
# docker-compose.staging.yml
version: '3.8'

services:
  go-engine:
    image: lx-dex:go-staging
    environment:
      - ENGINE_TYPE=go
      - LOG_LEVEL=info
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '2'
          memory: 2G
          
  cpp-engine:
    image: lx-dex:cpp-staging
    environment:
      - ENGINE_TYPE=cpp
      - PERFORMANCE_MODE=balanced
    deploy:
      replicas: 1
      resources:
        limits:
          cpus: '4'
          memory: 4G
          
  hybrid-engine:
    image: lx-dex:hybrid-staging
    environment:
      - ENGINE_TYPE=hybrid
      - CGO_ENABLED=1
    deploy:
      replicas: 2
      
  load-balancer:
    image: nginx:alpine
    volumes:
      - ./nginx/staging.conf:/etc/nginx/nginx.conf
    ports:
      - "80:80"
      - "443:443"
```

#### Kubernetes Staging
```yaml
# k8s/staging/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: lx-engine-staging
  namespace: lx-staging
spec:
  replicas: 3
  selector:
    matchLabels:
      app: lx-engine
      env: staging
  template:
    metadata:
      labels:
        app: lx-engine
        env: staging
    spec:
      containers:
      - name: engine
        image: lx-dex:hybrid-staging
        env:
        - name: ENGINE_TYPE
          value: "hybrid"
        - name: CONFIG_PATH
          value: "/config/staging.yaml"
        resources:
          requests:
            memory: "1Gi"
            cpu: "1"
          limits:
            memory: "2Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          grpc:
            port: 50051
          initialDelaySeconds: 10
          periodSeconds: 5
```

### Production Deployment

#### High-Performance Configuration
```yaml
# config/production.yaml
engine:
  type: cpp  # Maximum performance
  port: 50051
  
performance:
  max_orders_per_second: 1000000
  batch_size: 100
  thread_pool_size: 32
  
  # C++ specific
  use_hugepages: true
  numa_node: 0
  cpu_affinity: [0,1,2,3,4,5,6,7]
  
networking:
  tcp_nodelay: true
  socket_buffer_size: 8388608  # 8MB
  max_connections: 10000
  
security:
  tls:
    enabled: true
    cert_file: /secrets/tls.crt
    key_file: /secrets/tls.key
    ca_file: /secrets/ca.crt
  
  rate_limiting:
    enabled: true
    orders_per_second_per_user: 1000
    
monitoring:
  prometheus:
    enabled: true
    port: 9090
  
  tracing:
    enabled: true
    sampling_rate: 0.001  # 0.1% in production
    
logging:
  level: warn
  format: json
  output: stdout
  
  # Ship to centralized logging
  syslog:
    enabled: true
    endpoint: syslog.lx.internal:514
```

#### Kubernetes Production

##### Main Deployment
```yaml
# k8s/production/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: lx-cpp-engine
  namespace: lx-production
spec:
  replicas: 10
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 2
      maxUnavailable: 0
  selector:
    matchLabels:
      app: lx-engine
      engine: cpp
      tier: production
  template:
    metadata:
      labels:
        app: lx-engine
        engine: cpp
        tier: production
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
    spec:
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - key: app
                operator: In
                values:
                - lx-engine
            topologyKey: kubernetes.io/hostname
            
      nodeSelector:
        node.kubernetes.io/instance-type: c5n.24xlarge  # Network optimized
        
      tolerations:
      - key: "dedicated"
        operator: "Equal"
        value: "trading"
        effect: "NoSchedule"
        
      containers:
      - name: cpp-engine
        image: lx-dex:cpp-production-v1.0.0
        imagePullPolicy: IfNotPresent
        
        env:
        - name: ENGINE_TYPE
          value: "cpp"
        - name: PERFORMANCE_MODE
          value: "ultra"
        - name: CONFIG_PATH
          value: "/config/production.yaml"
          
        resources:
          requests:
            memory: "8Gi"
            cpu: "8"
            ephemeral-storage: "10Gi"
          limits:
            memory: "16Gi"
            cpu: "16"
            ephemeral-storage: "20Gi"
            
        volumeMounts:
        - name: config
          mountPath: /config
        - name: tls-certs
          mountPath: /secrets
          readOnly: true
        - name: hugepages
          mountPath: /dev/hugepages
          
        ports:
        - name: grpc
          containerPort: 50051
          protocol: TCP
        - name: metrics
          containerPort: 9090
          protocol: TCP
          
        livenessProbe:
          grpc:
            port: 50051
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
          
        readinessProbe:
          grpc:
            port: 50051
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          successThreshold: 1
          failureThreshold: 3
          
        startupProbe:
          grpc:
            port: 50051
          initialDelaySeconds: 0
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 30
          
      volumes:
      - name: config
        configMap:
          name: lx-config-production
      - name: tls-certs
        secret:
          secretName: lx-tls-certs
      - name: hugepages
        emptyDir:
          medium: HugePages
```

##### Service Configuration
```yaml
# k8s/production/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: lx-engine
  namespace: lx-production
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
    service.beta.kubernetes.io/aws-load-balancer-cross-zone-load-balancing-enabled: "true"
spec:
  type: LoadBalancer
  sessionAffinity: ClientIP
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 10800
  selector:
    app: lx-engine
    tier: production
  ports:
  - name: grpc
    port: 50051
    targetPort: 50051
    protocol: TCP
  - name: metrics
    port: 9090
    targetPort: 9090
    protocol: TCP
```

##### Horizontal Pod Autoscaler
```yaml
# k8s/production/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: lx-engine-hpa
  namespace: lx-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: lx-cpp-engine
  minReplicas: 10
  maxReplicas: 100
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: lx_orders_per_second
      target:
        type: AverageValue
        averageValue: "100000"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
```

## Cloud Provider Deployments

### AWS Deployment

#### Infrastructure as Code (Terraform)
```hcl
# terraform/aws/main.tf
provider "aws" {
  region = var.region
}

# EKS Cluster for container orchestration
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  
  cluster_name    = "lx-dex-production"
  cluster_version = "1.28"
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  node_groups = {
    trading = {
      desired_capacity = 20
      max_capacity     = 100
      min_capacity     = 10
      
      instance_types = ["c5n.24xlarge"]
      
      k8s_labels = {
        Environment = "production"
        Engine      = "cpp"
      }
      
      additional_tags = {
        "k8s.io/cluster-autoscaler/enabled" = "true"
      }
    }
  }
}

# RDS for persistent storage
resource "aws_db_instance" "postgres" {
  identifier = "lx-dex-production"
  
  engine         = "postgres"
  engine_version = "15.3"
  instance_class = "db.r6g.8xlarge"
  
  allocated_storage     = 1000
  max_allocated_storage = 10000
  storage_encrypted     = true
  
  multi_az               = true
  backup_retention_period = 30
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  performance_insights_enabled = true
  monitoring_interval         = 60
}

# ElastiCache for Redis
resource "aws_elasticache_replication_group" "redis" {
  replication_group_id       = "lx-dex-cache"
  replication_group_description = "Redis cluster for LX DEX"
  
  engine               = "redis"
  node_type           = "cache.r6g.4xlarge"
  number_cache_clusters = 3
  
  automatic_failover_enabled = true
  multi_az_enabled          = true
  
  snapshot_retention_limit = 7
  snapshot_window         = "03:00-05:00"
}

# Application Load Balancer
resource "aws_lb" "main" {
  name               = "lx-dex-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets           = module.vpc.public_subnets
  
  enable_deletion_protection = true
  enable_http2              = true
  enable_cross_zone_load_balancing = true
}
```

### GCP Deployment

#### GKE Configuration
```yaml
# gcp/gke-cluster.yaml
apiVersion: container.gke.io/v1beta1
kind: Cluster
metadata:
  name: lx-dex-production
spec:
  location: us-central1
  
  network: lx-vpc
  subnetwork: lx-subnet
  
  initialNodeCount: 10
  
  nodeConfig:
    machineType: c2-standard-60
    diskSizeGb: 500
    diskType: pd-ssd
    
    oauthScopes:
    - https://www.googleapis.com/auth/cloud-platform
    
    labels:
      environment: production
      engine: cpp
      
    taints:
    - key: dedicated
      value: trading
      effect: NO_SCHEDULE
      
  autoscaling:
    enabled: true
    minNodeCount: 10
    maxNodeCount: 100
    
  maintenancePolicy:
    window:
      dailyMaintenanceWindow:
        startTime: 03:00
```

### Azure Deployment

#### AKS Configuration
```json
{
  "name": "lx-dex-production",
  "location": "eastus",
  "properties": {
    "kubernetesVersion": "1.28",
    "dnsPrefix": "lx-dex",
    "agentPoolProfiles": [
      {
        "name": "trading",
        "count": 10,
        "vmSize": "Standard_F64s_v2",
        "osDiskSizeGB": 500,
        "vnetSubnetID": "/subscriptions/.../subnets/trading",
        "maxCount": 100,
        "minCount": 10,
        "enableAutoScaling": true,
        "type": "VirtualMachineScaleSets",
        "mode": "User",
        "nodeLabels": {
          "environment": "production",
          "engine": "cpp"
        },
        "nodeTaints": [
          "dedicated=trading:NoSchedule"
        ]
      }
    ],
    "networkProfile": {
      "networkPlugin": "azure",
      "networkPolicy": "calico",
      "loadBalancerSku": "standard"
    }
  }
}
```

## Deployment Automation

### CI/CD Pipeline

#### GitHub Actions
```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build Docker images
        run: |
          docker build -f Dockerfile.cpp -t lx-dex:cpp-${{ github.ref_name }} .
          docker build -f Dockerfile.hybrid -t lx-dex:hybrid-${{ github.ref_name }} .
          
      - name: Push to registry
        run: |
          docker tag lx-dex:cpp-${{ github.ref_name }} gcr.io/lx-project/lx-dex:cpp-${{ github.ref_name }}
          docker push gcr.io/lx-project/lx-dex:cpp-${{ github.ref_name }}
          
  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/lx-cpp-engine \
            cpp-engine=gcr.io/lx-project/lx-dex:cpp-${{ github.ref_name }} \
            -n lx-production
            
      - name: Wait for rollout
        run: |
          kubectl rollout status deployment/lx-cpp-engine -n lx-production
          
      - name: Run smoke tests
        run: |
          ./scripts/smoke-test.sh production
```

### Blue-Green Deployment

```bash
#!/bin/bash
# scripts/blue-green-deploy.sh

NEW_VERSION=$1
OLD_VERSION=$(kubectl get deployment lx-engine-blue -o jsonpath='{.spec.template.spec.containers[0].image}' | cut -d: -f2)

echo "Deploying $NEW_VERSION (current: $OLD_VERSION)"

# Deploy to green environment
kubectl set image deployment/lx-engine-green \
  engine=lx-dex:cpp-$NEW_VERSION \
  -n lx-production

# Wait for green to be ready
kubectl rollout status deployment/lx-engine-green -n lx-production

# Run tests against green
./scripts/test-deployment.sh green

if [ $? -eq 0 ]; then
  echo "Tests passed, switching traffic to green"
  
  # Switch traffic to green
  kubectl patch service lx-engine \
    -p '{"spec":{"selector":{"version":"green"}}}' \
    -n lx-production
    
  # Update blue with new version for next deployment
  kubectl set image deployment/lx-engine-blue \
    engine=lx-dex:cpp-$NEW_VERSION \
    -n lx-production
else
  echo "Tests failed, keeping traffic on blue"
  exit 1
fi
```

## Performance Optimization

### Linux Kernel Tuning
```bash
# /etc/sysctl.d/99-lx-performance.conf

# Network performance
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.ipv4.tcp_rmem = 4096 87380 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
net.core.netdev_max_backlog = 30000
net.ipv4.tcp_congestion_control = bbr
net.ipv4.tcp_notsent_lowat = 16384

# Connection handling
net.ipv4.tcp_max_syn_backlog = 8192
net.core.somaxconn = 65535
net.ipv4.tcp_fin_timeout = 10
net.ipv4.tcp_tw_reuse = 1

# Memory
vm.swappiness = 10
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5

# Huge pages for C++ engine
vm.nr_hugepages = 1024
```

### CPU Affinity
```bash
# scripts/set-affinity.sh
#!/bin/bash

# Pin C++ engine to specific CPUs
ENGINE_PID=$(pgrep lx-cpp-engine)

# Pin to CPUs 0-7 (first NUMA node)
taskset -cp 0-7 $ENGINE_PID

# Pin interrupt handlers to CPUs 8-15
for IRQ in $(grep eth0 /proc/interrupts | cut -d: -f1); do
  echo 8-15 > /proc/irq/$IRQ/smp_affinity_list
done
```

## Monitoring & Rollback

### Deployment Monitoring
```bash
# scripts/monitor-deployment.sh
#!/bin/bash

DEPLOYMENT=$1
NAMESPACE=$2

# Monitor key metrics during deployment
while true; do
  # Check pod status
  kubectl get pods -l app=lx-engine -n $NAMESPACE
  
  # Check metrics
  kubectl top pods -l app=lx-engine -n $NAMESPACE
  
  # Check errors
  kubectl logs -l app=lx-engine -n $NAMESPACE --tail=100 | grep ERROR
  
  sleep 5
done
```

### Automatic Rollback
```yaml
# k8s/production/rollback-policy.yaml
apiVersion: flagger.app/v1beta1
kind: Canary
metadata:
  name: lx-engine
  namespace: lx-production
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: lx-cpp-engine
  progressDeadlineSeconds: 600
  service:
    port: 50051
    targetPort: 50051
  analysis:
    interval: 1m
    threshold: 5
    maxWeight: 50
    stepWeight: 10
    metrics:
    - name: request-success-rate
      thresholdRange:
        min: 99
      interval: 1m
    - name: request-duration
      thresholdRange:
        max: 1
      interval: 30s
    webhooks:
    - name: load-test
      url: http://flagger-loadtester.lx-production/
      timeout: 5s
      metadata:
        cmd: "hey -z 1m -q 10 -c 2 http://lx-engine.lx-production:50051/"
```

## Disaster Recovery

### Backup Strategy
```yaml
# k8s/backup/backup-cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: lx-backup
  namespace: lx-production
spec:
  schedule: "0 */6 * * *"  # Every 6 hours
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: lx-backup:latest
            command:
            - /bin/sh
            - -c
            - |
              # Backup order book state
              kubectl exec -n lx-production lx-engine-0 -- \
                /app/lx-backup --output s3://lx-backups/orderbook-$(date +%Y%m%d-%H%M%S).dat
              
              # Backup database
              pg_dump $DATABASE_URL | gzip > /tmp/db-backup.sql.gz
              aws s3 cp /tmp/db-backup.sql.gz s3://lx-backups/db-$(date +%Y%m%d-%H%M%S).sql.gz
              
              # Backup configuration
              kubectl get configmap -n lx-production -o yaml > /tmp/config-backup.yaml
              aws s3 cp /tmp/config-backup.yaml s3://lx-backups/config-$(date +%Y%m%d-%H%M%S).yaml
```

### Recovery Procedures
```bash
#!/bin/bash
# scripts/disaster-recovery.sh

BACKUP_DATE=$1

echo "Starting disaster recovery from $BACKUP_DATE"

# Restore database
aws s3 cp s3://lx-backups/db-$BACKUP_DATE.sql.gz /tmp/
gunzip /tmp/db-$BACKUP_DATE.sql.gz
psql $DATABASE_URL < /tmp/db-$BACKUP_DATE.sql

# Restore configuration
aws s3 cp s3://lx-backups/config-$BACKUP_DATE.yaml /tmp/
kubectl apply -f /tmp/config-$BACKUP_DATE.yaml

# Restore order book state
aws s3 cp s3://lx-backups/orderbook-$BACKUP_DATE.dat /tmp/
kubectl cp /tmp/orderbook-$BACKUP_DATE.dat lx-engine-0:/tmp/
kubectl exec lx-engine-0 -- /app/lx-restore --input /tmp/orderbook-$BACKUP_DATE.dat

echo "Recovery complete"
```

## Security Hardening

### Network Policies
```yaml
# k8s/production/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: lx-engine-netpol
  namespace: lx-production
spec:
  podSelector:
    matchLabels:
      app: lx-engine
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: lx-production
    - podSelector:
        matchLabels:
          app: lx-gateway
    ports:
    - protocol: TCP
      port: 50051
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: lx-production
    ports:
    - protocol: TCP
      port: 5432  # PostgreSQL
    - protocol: TCP
      port: 6379  # Redis
```

### Pod Security Policy
```yaml
# k8s/production/pod-security-policy.yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: lx-engine-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'secret'
    - 'emptyDir'
    - 'persistentVolumeClaim'
  hostNetwork: false
  hostIPC: false
  hostPID: false
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
  readOnlyRootFilesystem: true
```

## Deployment Checklist

### Pre-Deployment
- [ ] All tests passing
- [ ] Security scan completed
- [ ] Performance benchmarks met
- [ ] Configuration reviewed
- [ ] Rollback plan prepared
- [ ] Monitoring alerts configured
- [ ] Team notified

### During Deployment
- [ ] Monitor error rates
- [ ] Check latency metrics
- [ ] Verify health checks
- [ ] Watch resource usage
- [ ] Monitor customer impact

### Post-Deployment
- [ ] Verify all services healthy
- [ ] Run smoke tests
- [ ] Check monitoring dashboards
- [ ] Review logs for errors
- [ ] Update documentation
- [ ] Send deployment report

## Conclusion

This deployment guide provides comprehensive strategies for deploying the LX DEX platform across various environments. Key considerations:

1. **Start simple**: Use docker-compose for development
2. **Scale gradually**: Test performance at each tier
3. **Monitor everything**: Comprehensive observability is crucial
4. **Automate deployment**: CI/CD reduces human error
5. **Plan for failure**: Always have rollback procedures
6. **Security first**: Implement defense in depth

For additional support, contact the DevOps team at devops@lx.com.