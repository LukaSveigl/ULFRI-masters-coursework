import { dbRetryTx, t } from "$lib/server/db";
import { decodeId } from "$lib/server/db_utils";
import { getUser } from "$lib/server/telefunc_util";
import { eq } from "drizzle-orm";
import { Abort } from "telefunc";

export async function onUpdateLoanPayment(id: string, valuePaid: number) {
    const user = getUser();
    const loanId = decodeId(id);

    await dbRetryTx(async (tx) => {
        const loanInfo = await tx
            .select({
                userId: t.loanPayment.userId,
                value: t.loanPayment.value,
                valuePaidConfirmed: t.loanPayment.valuePaidConfirmed
            })
            .from(t.loanPayment)
            .where(eq(t.loanPayment.id, loanId));

        if (loanInfo.length != 1 || !loanInfo[0].userId?.equals(user.id)) throw Abort();

        valuePaid *= 100;
        const { value, valuePaidConfirmed } = loanInfo[0];

        if (valuePaid < valuePaidConfirmed || valuePaid > value) throw Abort();

        await tx.update(t.loanPayment).set({ valuePaid }).where(eq(t.loanPayment.id, loanId));
    });
}
