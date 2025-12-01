import { db, dbRetry, t } from "$lib/server/db";
import { decodeId } from "$lib/server/db_utils";
import { getUser } from "$lib/server/telefunc_util";
import { and, eq, or } from "drizzle-orm";

export async function onRemoveFriend(id: string) {
    const user = getUser();
    const otherUserId = decodeId(id);
    await dbRetry(() =>
        db
            .delete(t.friend)
            .where(
                or(
                    and(eq(t.friend.user1Id, otherUserId), eq(t.friend.user2Id, user.id)),
                    and(eq(t.friend.user1Id, user.id), eq(t.friend.user2Id, otherUserId))
                )
            )
    );
}
