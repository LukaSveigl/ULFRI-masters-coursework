import { dbRetryTx, t } from "$lib/server/db";
import { decodeId } from "$lib/server/db_utils";
import { getUser } from "$lib/server/telefunc_util";
import { eq, inArray } from "drizzle-orm";
import { Abort } from "telefunc";

export async function onUpdateLoanPayments(
    payments: {
        id: string;
        valuePaid: number;
        valuePaidConfirmed: number;
    }[]
) {
    const user = getUser();

    await dbRetryTx(async (tx) => {
        const safetyCheck = await tx
            .selectDistinct({ loanId: t.loan.id, userId: t.loan.issuerUserId })
            .from(t.loanPayment)
            .innerJoin(t.loan, eq(t.loanPayment.loanId, t.loan.id))
            .where(
                inArray(
                    t.loanPayment.id,
                    payments.map((x) => decodeId(x.id))
                )
            );

        if (safetyCheck.length != 1 || !safetyCheck[0].userId?.equals(user.id)) throw Abort();

        for (const { id, valuePaid, valuePaidConfirmed } of payments) {
            if (valuePaid < 0 || valuePaidConfirmed < 0 || valuePaid < valuePaidConfirmed) {
                throw Abort();
            }
            await tx
                .update(t.loanPayment)
                .set({
                    valuePaid: ~~(valuePaid * 100),
                    valuePaidConfirmed: ~~(valuePaidConfirmed * 100)
                })
                .where(eq(t.loanPayment.id, decodeId(id)));
        }
    });
}
