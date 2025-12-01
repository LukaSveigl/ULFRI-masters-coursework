import * as auth from "$lib/server/auth";
import { decodeBase64urlIgnorePadding } from "@oslojs/encoding";
import type { Handle, ServerInit } from "@sveltejs/kit";
import { config } from "telefunc";

export const handle: Handle = async ({ event, resolve }) => {
    // Extremely janky demo code, can't expose it to the public
    const isRestricted = event.url.pathname.startsWith("/api");

    const sessionToken = event.cookies.get(auth.sessionCookieName);
    if (!sessionToken) {
        if (isRestricted) {
            return new Response(null, { status: 403 });
        }

        event.locals.user = null;
        event.locals.session = null;
        return resolve(event);
    }
    const sessionTokenBufffer = Buffer.from(decodeBase64urlIgnorePadding(sessionToken));

    const { session, user } = await auth.validateSessionToken(sessionTokenBufffer);
    if (session) {
        auth.setSessionTokenCookie(event, sessionTokenBufffer, session.expiresAt);
    } else {
        auth.deleteSessionTokenCookie(event);
    }

    event.locals.user = user;
    event.locals.session = session;

    if (isRestricted && !user) {
        return new Response(null, { status: 403 });
    }

    return resolve(event);
};

export const init: ServerInit = async () => {
    config.telefuncUrl = "/api/_telefunc";
};
