import { validateUsername } from "$lib/server/auth";
import { dbRetryTx, t } from "$lib/server/db";
import { generateId } from "$lib/server/db_utils";
import { getUser } from "$lib/server/telefunc_util";
import { and, eq, or, sql } from "drizzle-orm";

export async function onAddFriend(username: string) {
    const user = getUser();
    if (username == user.username) {
        return { ok: false, message: "You cannot add yourself as a friend" } as const;
    }

    if (!validateUsername(username)) {
        return { ok: false, message: "Invalid username" } as const;
    }

    return await dbRetryTx(async (tx) => {
        const userIds = await tx
            .select({ id: t.user.id })
            .from(t.user)
            .where(eq(t.user.username, username));
        if (userIds.length == 0) {
            return { ok: false, message: "This user doesn't exist" } as const;
        }
        const userId = userIds[0].id;

        const friendExists = await tx
            .select({ id: sql`1` })
            .from(t.friend)
            .where(
                or(
                    and(eq(t.friend.user1Id, userId), eq(t.friend.user2Id, user.id)),
                    and(eq(t.friend.user1Id, user.id), eq(t.friend.user2Id, userId))
                )
            )
            .limit(1);
        if (friendExists.length > 0) {
            return { ok: false, message: "You are already friends with this user" } as const;
        }

        await tx.insert(t.friend).values({ id: generateId(), user1Id: user.id, user2Id: userId });

        return { ok: true } as const;
    });
}
