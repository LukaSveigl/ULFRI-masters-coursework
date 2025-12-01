<script lang="ts">
    import AppIcon from "$lib/components/AppIcon.svelte";
    import Avatar from "$lib/components/Avatar.svelte";
    import InputEur from "$lib/components/InputEur.svelte";
    import Label from "$lib/components/Label.svelte";
    import Popup from "$lib/components/Popup.svelte";
    import { onUpdateLoanPayment } from "$lib/components/PopupDetailsLoanToMe.telefunc";
    import { formatDateTime, formatMoney } from "$lib/formatting";
    import { Button } from "$lib/shadcn/components/ui/button";
    import * as Dialog from "$lib/shadcn/components/ui/dialog";
    import { mutation } from "$lib/utils.svelte";
    import { ArrowUpToLine, X } from "lucide-svelte";

    let {
        open = $bindable(false),
        loan: loanProp = null,
        onupdate
    }: {
        open?: boolean;
        loan?: {
            createdOn: Date;
            _issuerUser: {
                id: string;
                displayName: string;
                avatar: boolean;
                avatarModified: Date;
                username: string;
            } | null;
            _group: {
                id: string;
                displayName: string;
                avatar: boolean;
                avatarModified: Date;
            } | null;
            _payment: {
                id: string;
                value: number;
                valuePaid: number;
                valuePaidConfirmed: number;
            };
        } | null;
        onupdate?: () => Promise<void>;
    } = $props();

    const mutValuePaid = mutation({
        get() {
            return loanProp?._payment?.valuePaid ?? null;
        },
        async set(valuePaid: number | null) {
            if (valuePaid == null) return "null";
            if (valuePaid < loanProp!._payment.valuePaidConfirmed) return "min";
            if (valuePaid > loanProp!._payment.value) return "max";
            if (!Number.isSafeInteger(valuePaid * 100)) return "precision";
            await onUpdateLoanPayment(loanProp!._payment.id, valuePaid);
            await onupdate?.();
        }
    });

    let loan = $derived(
        loanProp != null
            ? {
                  ...loanProp,
                  _payment: { ...loanProp._payment, valuePaid: mutValuePaid.value }
              }
            : null
    );
</script>

<Popup bind:open>
    <Dialog.Header>
        <Dialog.Title>Loan Details</Dialog.Title>
    </Dialog.Header>
    {#if loan}
        <div class="flex flex-col items-stretch gap-1">
            <div class="flex items-center">
                <p>Issued</p>
                <p class="ml-auto">{formatDateTime(loan.createdOn)}</p>
            </div>
            {#if loan._issuerUser != null}
                <div class="flex items-center">
                    <p>Issued by</p>
                    <div class="ml-auto">
                        <div class="flex items-center gap-2">
                            <Avatar user={loan._issuerUser} size="small" />
                            <p class="leading-none">{loan._issuerUser.displayName}</p>
                        </div>
                    </div>
                </div>
            {/if}
            {#if loan._group != null}
                <div class="flex items-center">
                    <p>To group</p>
                    <div class="ml-auto">
                        <div class="flex items-center gap-2">
                            <Avatar user={loan._group} isGroup size="small" />
                            <p class="leading-none">{loan._group.displayName}</p>
                        </div>
                    </div>
                </div>
            {/if}
            {#if loan._payment.valuePaidConfirmed != loan._payment.value}
                <div class="flex items-center">
                    <p>To pay</p>
                    <p class="ml-auto">
                        {formatMoney(loan._payment.value)}
                    </p>
                </div>
                <div class="flex items-center">
                    <p>Paid, confirmed</p>
                    <p class="ml-auto">
                        {formatMoney(loan._payment.valuePaidConfirmed)}
                    </p>
                </div>
            {:else}
                <div class="flex items-center">
                    <p>Paid</p>
                    <p class="ml-auto">
                        {formatMoney(loan._payment.value)}
                    </p>
                </div>
            {/if}
        </div>
        {#if loan._payment.valuePaidConfirmed != loan._payment.value}
            <div
                class={[
                    "grid grid-cols-[1fr_auto] grid-rows-[auto_auto] gap-x-4 gap-y-1",
                    mutValuePaid.applying && "animate-pulse"
                ]}
            >
                <Label class="row-span-2 grid grid-rows-subgrid" label="Paid">
                    <InputEur
                        min={loan._payment.valuePaidConfirmed}
                        max={loan._payment.value}
                        bind:value={mutValuePaid.value}
                    />
                </Label>
                <Button
                    class="col-start-2 row-start-2 flex-none shadow-md"
                    variant={mutValuePaid.stalled ? "outline" : "default"}
                    size="icon"
                    type="submit"
                    disabled={loan._payment.valuePaid == loan._payment.value}
                    onclick={() => {
                        mutValuePaid.value = loan._payment.value;
                    }}
                >
                    <AppIcon icon={mutValuePaid.stalled ? X : ArrowUpToLine} class="!size-6" />
                </Button>
            </div>
        {/if}
    {/if}
</Popup>
