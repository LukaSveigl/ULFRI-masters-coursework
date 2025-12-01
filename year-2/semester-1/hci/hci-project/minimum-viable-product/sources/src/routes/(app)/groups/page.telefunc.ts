import { dbRetryTx, t } from "$lib/server/db";
import { getFriends } from "$lib/server/db/common_queries";
import { decodeId, encodeUser, generateId } from "$lib/server/db_utils";
import { getUser } from "$lib/server/telefunc_util";
import { decodeBase64urlIgnorePadding } from "@oslojs/encoding";

export async function onRequestFriends() {
    const user = getUser();
    return (await getFriends(user.id)).map(encodeUser);
}

export async function onCreateGroup(config: {
    displayName: string;
    avatar: string | null;
    userIds: string[];
}) {
    const user = getUser();

    const avatar = config.avatar ? Buffer.from(decodeBase64urlIgnorePadding(config.avatar)) : null;
    const userIds = new Set([...config.userIds.map(decodeId), user.id]);

    return dbRetryTx(async (tx) => {
        const groupId = generateId();
        await tx.insert(t.group).values({
            id: groupId,
            displayName: config.displayName,
            avatar,
            avatarModified: new Date(),
            adminUserId: user.id
        });
        await tx.insert(t.groupMember).values(
            userIds
                .values()
                .map((x) => ({
                    id: generateId(),
                    userId: x,
                    groupId
                }))
                .toArray()
        );

        return { ok: true as const };
    });
}
