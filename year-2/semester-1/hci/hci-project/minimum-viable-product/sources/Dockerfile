FROM node:23-slim

WORKDIR /app
COPY ./ .
RUN corepack enable \
    && corepack install \
    && pnpm install --frozen-lockfile \
    && DATABASE_PATH=local.db pnpm run build

ENV PORT=5000
ENV NODE_ENV=production
EXPOSE 5000
CMD pnpm run db:push && echo -n "$DATABASE_PATH" > env_DATABASE_PATH && node build
