import { dbRetryTx, t } from "$lib/server/db";
import { and, countDistinct, eq, not, sql, sum, type SQL } from "drizzle-orm";
import type { PageServerLoad } from "./$types";

export const load: PageServerLoad = async (event) => {
    await event.parent();
    const user = event.locals.user!;

    const { unpaidN, unpaidValue, owedN, owedValue } = await dbRetryTx(async (tx) => {
        const [{ n: unpaidN, value: unpaidValue }] = await tx
            .select({
                n: countDistinct(t.loanPayment.loanId),
                value: sum(
                    sql<number>`${t.loanPayment.value} - ${t.loanPayment.valuePaid}`
                ) as any as SQL<number>
            })
            .from(t.loanPayment)
            .where(
                and(
                    eq(t.loanPayment.userId, user.id),
                    not(eq(t.loanPayment.value, t.loanPayment.valuePaid))
                )
            );

        const [{ n: owedN, value: owedValue }] = await tx
            .select({
                n: countDistinct(t.loan.id),
                value: sum(
                    sql<number>`${t.loanPayment.value} - ${t.loanPayment.valuePaidConfirmed}`
                ) as any as SQL<number>
            })
            .from(t.loan)
            .innerJoin(t.loanPayment, eq(t.loan.id, t.loanPayment.loanId))
            .where(
                and(
                    eq(t.loan.issuerUserId, user.id),
                    not(eq(t.loanPayment.value, t.loanPayment.valuePaidConfirmed))
                )
            );

        return { unpaidN, unpaidValue, owedN, owedValue };
    });

    return {
        unpaidN,
        unpaidValue: unpaidValue / 100,
        owedN,
        owedValue: owedValue / 100
    };
};
