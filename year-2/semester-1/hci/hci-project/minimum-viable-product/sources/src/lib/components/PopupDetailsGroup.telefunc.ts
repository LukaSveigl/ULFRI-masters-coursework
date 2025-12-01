import { db, dbRetry, dbRetryTx, t } from "$lib/server/db";
import { decodeId, encodeUser, selectUser } from "$lib/server/db_utils";
import { getUser } from "$lib/server/telefunc_util";
import { and, eq, not, sql } from "drizzle-orm";
import { Abort } from "telefunc";

export async function onGetMembers(groupId: string) {
    const user = getUser();
    const groupIdB = decodeId(groupId);

    const users = await dbRetryTx(async (tx) => {
        const membershipExists = await tx
            .select({ id: sql`1` })
            .from(t.groupMember)
            .where(and(eq(t.groupMember.groupId, groupIdB), eq(t.groupMember.userId, user.id)))
            .limit(1);
        if (membershipExists.length == 0) {
            throw Abort({ invalidGroup: true });
        }

        return tx
            .select(selectUser(t.user))
            .from(t.groupMember)
            .innerJoin(t.user, eq(t.groupMember.userId, t.user.id))
            .where(
                and(eq(t.groupMember.groupId, groupIdB), not(eq(t.groupMember.userId, user.id)))
            );
    });

    return users.map(encodeUser);
}

export async function onLeaveGroup(groupId: string) {
    const user = getUser();
    await dbRetry(() =>
        db
            .delete(t.groupMember)
            .where(
                and(eq(t.groupMember.groupId, decodeId(groupId)), eq(t.groupMember.userId, user.id))
            )
    );
}
