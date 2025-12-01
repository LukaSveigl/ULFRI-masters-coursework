import { db, dbRetry, t } from "$lib/server/db";
import { selectGroup, selectUser } from "$lib/server/db_utils";
import { and, eq, or } from "drizzle-orm";

export async function getFriends(userId: Buffer) {
    return dbRetry(() =>
        db
            .select(selectUser(t.user))
            .from(t.friend)
            .innerJoin(
                t.user,
                or(
                    and(eq(t.friend.user1Id, userId), eq(t.friend.user2Id, t.user.id)),
                    and(eq(t.friend.user1Id, t.user.id), eq(t.friend.user2Id, userId))
                )
            )
    );
}

export async function getGroups(userId: Buffer) {
    return dbRetry(() =>
        db
            .select(selectGroup(t.group))
            .from(t.groupMember)
            .innerJoin(t.group, eq(t.groupMember.groupId, t.group.id))
            .where(eq(t.groupMember.userId, userId))
    );
}
