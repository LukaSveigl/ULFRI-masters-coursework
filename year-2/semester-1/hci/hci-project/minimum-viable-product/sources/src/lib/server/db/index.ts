import { env } from "$env/dynamic/private";
import * as t from "$lib/server/db/schema";
import Database from "better-sqlite3";
import { drizzle } from "drizzle-orm/better-sqlite3";
import fs from "node:fs";

const databasePath = (() => {
    if (env.DATABASE_PATH) return env.DATABASE_PATH;
    try {
        return fs.readFileSync("env_DATABASE_PATH", "utf8");
    } catch {
        throw new Error("DATABASE_PATH is not set");
    }
})();

const client = new Database(databasePath);

export const db = drizzle(client, { schema: t });
export * as t from "$lib/server/db/schema";

export async function dbRetry<T>(fn: () => Promise<T>) {
    while (true) {
        try {
            return await fn();
        } catch (e) {
            if (!(e instanceof Database.SqliteError)) throw e;
            if (e.code != "SQLITE_BUSY") throw e;
        }
    }
}

export async function dbRetryTx<T>(
    fn: (tx: Parameters<Parameters<typeof db.transaction>[0]>[0]) => Promise<T>
) {
    return dbRetry(() => db.transaction(fn));
}
