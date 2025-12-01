import { dbRetryTx, t } from "$lib/server/db";
import { decodeId, encodeUser, generateId, selectUser } from "$lib/server/db_utils";
import { getUser } from "$lib/server/telefunc_util";
import { and, eq, inArray, not, sql } from "drizzle-orm";
import { Abort } from "telefunc";

export async function onGetGroupMembers(groupId: string) {
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

export async function onGetUsers(ids: string[]) {
    getUser();

    const users = await dbRetryTx(async (tx) => {
        return tx
            .select(selectUser(t.user))
            .from(t.user)
            .where(inArray(t.user.id, ids.map(decodeId)));
    });

    return users.map(encodeUser);
}

export async function onIssueLoan(
    groupId: string | null,
    payments: { userId: string; amount: number }[]
) {
    const user = getUser();

    await dbRetryTx(async (tx) => {
        const loanId = generateId();
        await tx.insert(t.loan).values({
            id: loanId,
            issuerUserId: user.id,
            groupId: groupId != null ? decodeId(groupId) : null,
            createdOn: new Date()
        });
        await tx.insert(t.loanPayment).values(
            payments.map((payment) => ({
                id: generateId(),
                loanId,
                userId: decodeId(payment.userId),
                value: ~~(payment.amount * 100),
                valuePaid: 0,
                valuePaidConfirmed: 0
            }))
        );
    });
}
