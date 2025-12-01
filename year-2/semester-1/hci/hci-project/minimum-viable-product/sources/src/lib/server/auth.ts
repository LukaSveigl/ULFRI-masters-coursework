import { db, t } from "$lib/server/db";
import { selectUser } from "$lib/server/db_utils";
import { sha256 } from "@oslojs/crypto/sha2";
import { encodeBase64urlNoPadding } from "@oslojs/encoding";
import type { RequestEvent } from "@sveltejs/kit";
import { eq } from "drizzle-orm";

const DAY_IN_MS = 1000 * 60 * 60 * 24;

export const sessionCookieName = "banana-auth-session";

export function generateSessionToken() {
    return Buffer.from(crypto.getRandomValues(new Uint8Array(18)));
}

export async function createSession(token: Buffer, userId: Buffer) {
    const sessionId = Buffer.from(sha256(token));
    const session = {
        id: sessionId,
        userId,
        expiresAt: new Date(Date.now() + DAY_IN_MS * 30)
    } satisfies typeof t.session.$inferInsert;
    await db.insert(t.session).values(session);
    return session;
}

export async function validateSessionToken(token: Buffer) {
    const sessionId = Buffer.from(sha256(token));
    const [result] = await db
        .select({
            user: selectUser(t.user),
            session: t.session
        })
        .from(t.session)
        .innerJoin(t.user, eq(t.session.userId, t.user.id))
        .where(eq(t.session.id, sessionId));

    if (!result) {
        return { session: null, user: null };
    }
    const { session, user } = result;

    const sessionExpired = Date.now() >= session.expiresAt.getTime();
    if (sessionExpired) {
        await db.delete(t.session).where(eq(t.session.id, session.id));
        return { session: null, user: null };
    }

    const renewSession = Date.now() >= session.expiresAt.getTime() - DAY_IN_MS * 15;
    if (renewSession) {
        session.expiresAt = new Date(Date.now() + DAY_IN_MS * 30);
        await db
            .update(t.session)
            .set({ expiresAt: session.expiresAt })
            .where(eq(t.session.id, session.id));
    }

    return { session, user };
}

export type SessionValidationResult = Awaited<ReturnType<typeof validateSessionToken>>;

export async function invalidateSession(sessionId: Buffer) {
    await db.delete(t.session).where(eq(t.session.id, sessionId));
}

export function setSessionTokenCookie(event: RequestEvent, token: Buffer, expiresAt: Date) {
    event.cookies.set(sessionCookieName, encodeBase64urlNoPadding(token), {
        expires: expiresAt,
        path: "/"
    });
    event.cookies.set(sessionCookieName + "-exists", "", {
        expires: expiresAt,
        path: "/",
        httpOnly: false
    });
}

export function deleteSessionTokenCookie(event: RequestEvent) {
    event.cookies.delete(sessionCookieName, { path: "/" });
    event.cookies.delete(sessionCookieName + "-exists", { path: "/" });
}

export function validateUsername(username: unknown): username is string {
    return (
        typeof username === "string" &&
        username.length >= 3 &&
        username.length <= 31 &&
        /^[a-z0-9_-]+$/.test(username)
    );
}

export function validatePassword(password: unknown): password is string {
    return typeof password === "string" && password.length >= 6 && password.length <= 255;
}
