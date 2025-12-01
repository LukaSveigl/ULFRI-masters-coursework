import { env } from "$env/dynamic/private";
import * as auth from "$lib/server/auth";
import { db, t } from "$lib/server/db";
import { hash, verify } from "@node-rs/argon2";
import { fail, redirect } from "@sveltejs/kit";
import { eq } from "drizzle-orm";
import type { Actions, PageServerLoad } from "./$types";

export const load: PageServerLoad = async (event) => {
    if (event.locals.user) {
        return redirect(302, "/dashboard");
    }
    return {};
};

const argon2settings = {
    memoryCost: 19456,
    timeCost: 2,
    outputLen: 32,
    parallelism: 1
};

export const actions: Actions = {
    async login(event) {
        const formData = await event.request.formData();
        const username = formData.get("username");
        const password = formData.get("password");
        const registrationToken = formData.get("registration_token");

        if (!auth.validateUsername(username)) {
            return fail(400, {
                message: "Invalid username (min 3, max 31 characters, alphanumeric only)"
            });
        }
        if (!auth.validatePassword(password)) {
            return fail(400, { message: "Invalid password (min 6, max 255 characters)" });
        }

        if (registrationToken && !validateRegistrationToken(registrationToken)) {
            return fail(400, { message: "Invalid registration token" });
        }

        const results = await db.select().from(t.user).where(eq(t.user.username, username));
        const existingUser = results.at(0);

        if (registrationToken && !existingUser) {
            const userId = generateUserId();
            const passwordHash = await hash(password, argon2settings);

            try {
                await db.insert(t.user).values({
                    id: userId,
                    displayName: username,
                    avatarModified: new Date(),
                    username,
                    passwordHash
                });

                const sessionToken = auth.generateSessionToken();
                const session = await auth.createSession(sessionToken, userId);
                auth.setSessionTokenCookie(event, sessionToken, session.expiresAt);
            } catch (e) {
                return fail(500, { message: "An error has occurred" });
            }
            return redirect(302, "/dashboard");
        } else if (registrationToken && existingUser) {
            return fail(400, { message: "This username is taken" });
        } else if (!registrationToken && !existingUser) {
            return fail(400, { message: "Incorrect username or password" });
        } else if (!registrationToken && existingUser) {
            const validPassword = await verify(existingUser.passwordHash, password, argon2settings);
            if (!validPassword) {
                return fail(400, { message: "Incorrect username or password" });
            }

            const sessionToken = auth.generateSessionToken();
            const session = await auth.createSession(sessionToken, existingUser.id);
            auth.setSessionTokenCookie(event, sessionToken, session.expiresAt);

            return redirect(302, "/dashboard");
        } else {
            throw new Error("unreachable");
        }
    },
    async logout(event) {
        if (!event.locals.session) {
            return fail(401);
        }
        await auth.invalidateSession(event.locals.session.id);
        auth.deleteSessionTokenCookie(event);

        return redirect(302, "/login");
    }
};

function generateUserId() {
    return Buffer.from(crypto.getRandomValues(new Uint8Array(16)));
}

function validateRegistrationToken(token: unknown): boolean {
    if (!env.REGISTRATION_TOKEN) throw new Error("REGISTRATION_TOKEN is not set");
    return typeof token == "string" && token == env.REGISTRATION_TOKEN;
}
