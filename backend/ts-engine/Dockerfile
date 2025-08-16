# TypeScript Engine Dockerfile
FROM node:20-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci

COPY tsconfig.json ./
COPY src ./src
COPY proto ../proto

RUN npm run build

FROM node:20-alpine

RUN apk add --no-cache ca-certificates curl

WORKDIR /app
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/package*.json ./

EXPOSE 50055
CMD ["node", "dist/server.js"]