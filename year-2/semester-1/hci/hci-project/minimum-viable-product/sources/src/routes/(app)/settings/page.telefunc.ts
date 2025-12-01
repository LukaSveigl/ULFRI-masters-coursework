import { db, dbRetry, t } from "$lib/server/db";
import { getUser } from "$lib/server/telefunc_util";
import { decodeBase64urlIgnorePadding } from "@oslojs/encoding";
import { eq } from "drizzle-orm";
import { Abort } from "telefunc";
import { displayNameValid } from "./+page.svelte";

export async function onChangeProfilePic(base64url: string) {
    const user = getUser();
    await dbRetry(async () => {
        await db
            .update(t.user)
            .set({
                avatar: Buffer.from(decodeBase64urlIgnorePadding(base64url)),
                avatarModified: new Date()
            })
            .where(eq(t.user.id, user.id));
    });
}

export async function onChangeDisplayName(displayName: string) {
    const user = getUser();
    if (!displayNameValid(displayName)) throw Abort({ invalidDisplayName: true });
    await dbRetry(async () => {
        await db.update(t.user).set({ displayName: displayName }).where(eq(t.user.id, user.id));
    });
}
