<script lang="ts">
    import { invalidateAll } from "$app/navigation";
    import LoanSummary from "$lib/components/LoanSummary.svelte";
    import PopupDetailsLoanByMe from "$lib/components/PopupDetailsLoanByMe.svelte";
    import PopupDetailsLoanToMe from "$lib/components/PopupDetailsLoanToMe.svelte";
    import { ScrollArea } from "$lib/shadcn/components/ui/scroll-area/";
    import { sortUsers } from "$lib/sort.js";

    let { data } = $props();

    let idToUser = $derived(new Map(data.users.map((x) => [x.id, x])));
    let idToGroup = $derived(new Map(data.groups.map((x) => [x.id, x])));
    let idToLoanPayment = $derived(
        new Map(
            data.loanPayments.map((x) => [
                x.id,
                {
                    ...x,
                    _user: x.userId != null ? idToUser.get(x.userId)! : null,
                    _status:
                        x.value == x.valuePaidConfirmed
                            ? ("paid" as const)
                            : x.valuePaid == x.valuePaidConfirmed
                              ? ("unpaid" as const)
                              : ("unconfirmed" as const)
                }
            ])
        )
    );
    let loanIdToLoanPayments = $derived.by(() => {
        type V = typeof idToLoanPayment extends Map<any, infer T> ? T : never;
        const ret = new Map<string, V[]>();
        for (const v of idToLoanPayment.values()) {
            let vs = ret.get(v.loanId);
            if (vs == null) {
                vs = [];
                ret.set(v.loanId, vs);
            }
            vs.push(v);
        }
        for (const vs of ret.values()) {
            vs.sort((a, b) => {
                if (a._user == null && b._user == null) {
                    if (a.id < b.id) return -1;
                    if (a.id > b.id) return 1;
                    return 0;
                }
                if (a._user == null && b._user != null) return 1;
                if (a._user != null && b._user == null) return -1;
                return sortUsers(a._user!, b._user!);
            });
        }
        return ret;
    });
    let idToLoan = $derived(
        new Map(
            data.loans.map((x) => {
                let loanType: "byme" | "tome" = x.issuerUserId == data.user.id ? "byme" : "tome";

                const payments = loanIdToLoanPayments.get(x.id)!;
                let status: "paid" | "unpaid" | "unconfirmed" = "paid";

                if (loanType == "byme") {
                    for (const payment of payments) {
                        if (payment._status == "unconfirmed") {
                            status = "unconfirmed";
                            break;
                        } else if (payment._status == "unpaid") {
                            status = "unpaid";
                        }
                    }
                } else {
                    status = payments.find((x) => x.userId == data.user.id)!._status;
                }

                return [
                    x.id,
                    {
                        ...x,
                        _issuerUser: x.issuerUserId != null ? idToUser.get(x.issuerUserId)! : null,
                        _group: x.groupId != null ? idToGroup.get(x.groupId)! : null,
                        _payments: payments,
                        _loanType: loanType,
                        _status: status
                    }
                ];
            })
        )
    );

    type Loan = typeof idToLoan extends Map<any, infer T> ? T : never;

    let displayedLoans = $derived.by(() => {
        const outstanding: Loan[] = [];
        const history: Loan[] = [];

        for (const loan of idToLoan.values()) {
            if (loan._status == "paid") {
                history.push(loan);
            } else {
                outstanding.push(loan);
            }
        }

        outstanding.sort((a, b) => {
            if (a._loanType == "tome" && b._loanType == "byme") return -1;
            if (a._loanType == "byme" && b._loanType == "tome") return 1;
            if (a.createdOn < b.createdOn) return 1;
            if (a.createdOn > b.createdOn) return -1;
            if (a.id < b.id) return -1;
            if (a.id > b.id) return 1;
            return 0;
        });

        history.sort((a, b) => {
            if (a.createdOn < b.createdOn) return 1;
            if (a.createdOn > b.createdOn) return -1;
            if (a.id < b.id) return -1;
            if (a.id > b.id) return 1;
            return 0;
        });

        return { outstanding, history };
    });

    function getFirstUserAvatar(payments: Loan["_payments"]) {
        for (const payment of payments) {
            if (payment._user == null) continue;
            return { user: payment._user, isGroup: false };
        }
        return null;
    }

    function getLoanValue(loan: Loan) {
        if (loan._loanType == "byme") {
            const paid = loan._payments.reduce(
                (acc, payment) => acc + payment.valuePaidConfirmed,
                0
            );
            if (loan._status == "paid") {
                return paid;
            }
            const owed = loan._payments.reduce((acc, payment) => acc + payment.value, 0);
            return owed - paid;
        } else {
            const payment = loan._payments.find((x) => x.userId == data.user.id)!;
            if (payment._status == "paid") return payment.valuePaidConfirmed;
            return payment.value - payment.valuePaid;
        }
    }

    let popupOpen: false | "byme" | "tome" = $state(false);
    let popupLoanId: string = $state("");
</script>

{#snippet displayLoans(name: string, loans: Loan[])}
    {#if loans.length > 0}
        <div class="flex flex-col items-stretch gap-2">
            <p class="text-xl">{name}</p>
            <ul class="flex flex-col items-stretch gap-2">
                {#each loans as loan (loan.id)}
                    <li class="grid">
                        <LoanSummary
                            createdOn={loan.createdOn}
                            avatar={loan._issuerUser != null
                                ? loan._loanType == "tome"
                                    ? { user: loan._issuerUser, isGroup: false }
                                    : loan._group != null
                                      ? { user: loan._group, isGroup: true }
                                      : getFirstUserAvatar(loan._payments)
                                : null}
                            loanType={loan._loanType}
                            badge={loan._status == "unconfirmed"}
                            nPayments={loan._payments.length}
                            value={getLoanValue(loan)}
                            onclick={() => {
                                popupOpen = loan._loanType;
                                popupLoanId = loan.id;
                            }}
                        />
                    </li>
                {/each}
            </ul>
        </div>
    {/if}
{/snippet}

<ScrollArea orientation="both" class="-m-6">
    <div class="m-6 flex flex-col items-stretch gap-4">
        {@render displayLoans("Outstanding", displayedLoans.outstanding)}
        {@render displayLoans("History", displayedLoans.history)}
    </div>
</ScrollArea>

<PopupDetailsLoanByMe
    bind:open={() => popupOpen == "byme",
    (x) => {
        if (x) {
            popupOpen = "byme";
        } else {
            popupOpen = false;
        }
    }}
    loan={idToLoan.get(popupLoanId)}
    onupdate={async () => {
        await invalidateAll();
    }}
/>

<PopupDetailsLoanToMe
    bind:open={() => popupOpen == "tome",
    (x) => {
        if (x) {
            popupOpen = "tome";
        } else {
            popupOpen = false;
        }
    }}
    loan={(() => {
        const loan = idToLoan.get(popupLoanId);
        if (loan == null) return null;
        return {
            ...loan,
            _payment: loan._payments.find((x) => x.userId == data.user.id)!
        };
    })()}
    onupdate={async () => {
        await invalidateAll();
    }}
/>
