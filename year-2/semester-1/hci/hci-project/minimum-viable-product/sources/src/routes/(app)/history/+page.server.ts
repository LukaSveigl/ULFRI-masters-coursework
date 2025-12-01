import { dbRetryTx, t } from "$lib/server/db";
import {
    encodeGroup,
    encodeLoan,
    encodeLoanPayment,
    encodeUser,
    selectGroup,
    selectUser
} from "$lib/server/db_utils";
import { eq, getTableColumns, inArray } from "drizzle-orm";
import type { PageServerLoad } from "./$types";

export const load: PageServerLoad = async (event) => {
    await event.parent();
    const user = event.locals.user!;

    const { loans, loanPayments, users, groups } = await dbRetryTx(async (tx) => {
        const loanIdsByMe = tx
            .select({ id: t.loan.id })
            .from(t.loan)
            .where(eq(t.loan.issuerUserId, user.id));
        const loanIdsToMe = tx
            .select({ id: t.loanPayment.loanId })
            .from(t.loanPayment)
            .where(eq(t.loanPayment.userId, user.id));
        const loanIdsRelevant = loanIdsByMe.union(loanIdsToMe).as("loan_ids_releavnt");
        const loans = await tx
            .select(getTableColumns(t.loan))
            .from(t.loan)
            .innerJoin(loanIdsRelevant, eq(t.loan.id, loanIdsRelevant.id));

        const loanPayments = await tx
            .select(getTableColumns(t.loanPayment))
            .from(t.loanPayment)
            .where(
                inArray(
                    t.loanPayment.loanId,
                    loans.map((x) => x.id)
                )
            );

        const userIdsRelevant = new Set(
            loans
                .map((x) => x.issuerUserId)
                .concat(loanPayments.map((x) => x.userId))
                .filter((x) => x != null)
        )
            .values()
            .toArray();
        const users = await tx
            .select(selectUser(t.user))
            .from(t.user)
            .where(inArray(t.user.id, userIdsRelevant));

        const groupIdsRelevant = new Set(loans.map((x) => x.groupId).filter((x) => x != null))
            .values()
            .toArray();
        const groups = await tx
            .select(selectGroup(t.group))
            .from(t.group)
            .where(inArray(t.group.id, groupIdsRelevant));

        return { loans, loanPayments, users, groups };
    });

    return {
        loans: loans.map(encodeLoan),
        loanPayments: loanPayments.map(encodeLoanPayment),
        users: users.map(encodeUser),
        groups: groups.map(encodeGroup)
    };
};
