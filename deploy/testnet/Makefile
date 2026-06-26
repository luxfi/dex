# dexd multi-node testnet — one command per box.
#
#   make build              # build the dexd-testnet image from this repo (native arch)
#   make up                 # generate compose for N venues and start them
#   make up N=11            # ... with 11 venues
#   make ps / make logs     # status / tail logs
#   make bench              # drive load across ALL boxes (needs EVO_IP/SPARK_IP in .env)
#   make verify             # cross-arch determinism proof across two boxes
#   make down               # stop venues (keep data)   |  make clean (stop + wipe)
#
# Config comes from .env (copy from .env.example). N can be overridden on the CLI.

SHELL := /bin/bash
-include .env
export

N          ?= 3
DEX_IMAGE  ?= dexd-testnet:local
BASE_PORT  ?= 9099
BIND_ADDR  ?= 0.0.0.0
COMPOSE    ?= docker compose
GEN        := compose.gen.yml

BENCH_CONNS    ?= 32
BENCH_MARKETS  ?= 4
BENCH_DURATION ?= 30s
BENCH_WARMUP   ?= 3s
BENCH_OP       ?= place

.PHONY: help build gen up down ps logs restart bench verify pull clean addrs

help:
	@sed -n '2,12p' Makefile

# Build the testnet image (dexd + dexbench) from the repo root, native arch.
# Pass a private-module token by exporting GH_TOKEN (only needed for a cold fetch).
build:
	cd ../.. && DOCKER_BUILDKIT=1 docker build -f deploy/testnet/Dockerfile -t $(DEX_IMAGE) \
	  $${GH_TOKEN:+--secret id=gh_token,env=GH_TOKEN} .

# Generate compose.gen.yml for N venues.
gen:
	N=$(N) ./gen-compose.sh

# Start N venues. Builds the image first if it is missing.
up: gen
	@docker image inspect $(DEX_IMAGE) >/dev/null 2>&1 || $(MAKE) build
	$(COMPOSE) -f $(GEN) up -d
	@echo "up: $(N) venues on $(BIND_ADDR):$(BASE_PORT)..$$(($(BASE_PORT)+$(N)-1))"

down:
	$(COMPOSE) -f $(GEN) down

clean:
	$(COMPOSE) -f $(GEN) down -v
	rm -f $(GEN)

ps:
	$(COMPOSE) -f $(GEN) ps

logs:
	$(COMPOSE) -f $(GEN) logs -f --tail=50

restart:
	$(COMPOSE) -f $(GEN) restart

pull:
	docker pull $(DEX_IMAGE)

# Print the endpoint list the bench will drive (this box's N nodes on each box IP).
addrs:
	@$(call build_addrs)

# Drive load across every venue on every configured box.
# Uses --network host so the bench container reaches both boxes over the LAN.
bench:
	@addrs=$$( $(call build_addrs) ); \
	echo "bench -> $$addrs"; \
	docker run --rm --network host --entrypoint dexbench $(DEX_IMAGE) \
	  -addrs "$$addrs" -conns $(BENCH_CONNS) -markets $(BENCH_MARKETS) \
	  -warmup $(BENCH_WARMUP) -duration $(BENCH_DURATION) -op $(BENCH_OP)

# Cross-arch determinism: identical order sequence into one venue on each of two
# boxes; assert byte-identical consensus output. Needs EVO_IP and SPARK_IP.
verify:
	@if [ -z "$(EVO_IP)" ] || [ -z "$(SPARK_IP)" ]; then \
	  echo "set EVO_IP and SPARK_IP in .env first"; exit 2; fi
	docker run --rm --network host --entrypoint dexbench $(DEX_IMAGE) \
	  -verify -addrs "$(EVO_IP):$(BASE_PORT),$(SPARK_IP):$(BASE_PORT)"

# build_addrs: emit "IP:port,..." for nodes 0..N-1 on each non-empty box IP.
define build_addrs
bash -c 'a=""; for ip in $(EVO_IP) $(SPARK_IP) $(RA_IP); do \
  [ -z "$$ip" ] && continue; \
  for i in $$(seq 0 $$(($(N)-1))); do a="$$a$${a:+,}$$ip:$$(($(BASE_PORT)+i))"; done; \
done; \
[ -z "$$a" ] && a="127.0.0.1:$(BASE_PORT)"; \
echo "$$a"'
endef
