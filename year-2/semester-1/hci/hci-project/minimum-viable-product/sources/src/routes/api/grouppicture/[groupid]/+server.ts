import { db, dbRetry, t } from "$lib/server/db";
import { decodeBase64urlIgnorePadding } from "@oslojs/encoding";
import { eq } from "drizzle-orm";
import type { RequestHandler } from "./$types";

export const GET: RequestHandler = async (event) => {
    const [result] = await dbRetry(() =>
        db
            .select({ avatar: t.group.avatar })
            .from(t.group)
            .where(eq(t.group.id, Buffer.from(decodeBase64urlIgnorePadding(event.params.groupid))))
    );
    if (!result) return new Response(null, { status: 404 });
    return new Response(result.avatar, { headers: { ["content-type"]: "image/png" } });
};
