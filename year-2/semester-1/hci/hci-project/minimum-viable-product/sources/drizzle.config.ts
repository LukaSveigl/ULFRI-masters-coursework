import { defineConfig } from "drizzle-kit";

if (!process.env.DATABASE_PATH) throw new Error("DATABASE_PATH is not set (drizzle config)");

export default defineConfig({
    schema: "./src/lib/server/db/schema.ts",

    dbCredentials: {
        url: `file:${process.env.DATABASE_PATH}`
    },

    verbose: true,
    strict: false,
    dialect: "sqlite"
});
